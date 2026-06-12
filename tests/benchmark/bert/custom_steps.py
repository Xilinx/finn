############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
############################################################################

"""
Custom build steps for the BERT (6-layer MLO + V80) benchmark.

These steps are ported from the Brainsmith BERT flow
(``examples/bert_training`` + ``brainsmith/steps``) into self-contained FINN
``DataflowBuildConfig`` steps. Each step is a plain function with the signature
``step_xxx(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper``,
mirroring the finn-examples ``mobilenet-v1/custom_steps.py`` pattern.

The Brainsmith ``@step`` / ``@transform`` decorators and the
``get_transform(...)`` / ``apply_transforms(...)`` plugin indirection have been
removed; the concrete QONNX/FINN transforms are imported directly.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import qonnx.custom_op.registry as registry
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    SortCommutativeInputsInitializerLast,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

import finn.core.onnx_exec as oxe
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedOp
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

logger = logging.getLogger(__name__)


# ===========================================================================
# Pre-processing
# ===========================================================================


def step_bert_cleanup(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Basic graph cleanup / preparation for BERT models.

    Ported from Brainsmith ``bert_cleanup`` step. Sorts commutative inputs so
    initializers come last and removes identity ops.
    """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model


def step_remove_head(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Remove all nodes up to the first LayerNormalization and rewire the input.

    Ported from the Brainsmith example ``remove_head`` step. After the head is
    removed, the QONNX->FINN pre-processing transforms from Brainsmith's
    ``qonnx_to_finn`` step that the stock ``step_qonnx_to_finn`` does not run
    (ExtractNormScaleBias, FoldConstants, ConvertDivToMul) are applied here.
    They run *after* the head removal because ExtractNormScaleBias appends
    Mul/Add nodes onto LayerNorm outputs; the stock ``step_qonnx_to_finn``
    (ConvertQONNXtoFINN) runs in a later step.
    """
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"

    to_remove = []
    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected a linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes (including the first LayerNorm)
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i, ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    # Clean up after head removal
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())

    # QONNX->FINN pre-processing (ported from Brainsmith qonnx_to_finn step).
    model = model.transform(ExtractNormScaleBias())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul())

    return model


def step_generate_reference_io(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Generate a reference input/output pair for the head-removed model.

    Ported from the Brainsmith example ``generate_reference_io`` step. Saves
    ``input.npy``, ``expected_output.npy`` and ``expected_context.npz`` into the
    build output directory; these feed ``verify_input_npy`` /
    ``verify_expected_output_npy`` and the shell handover.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = np.random.uniform(0, 1000, size=in_shape).astype(np.float32)
    np.save(os.path.join(cfg.output_dir, "input.npy"), in_tensor)

    input_t = {input_m.name: in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(os.path.join(cfg.output_dir, "expected_output.npy"), y_ref[out_name])
    np.savez(os.path.join(cfg.output_dir, "expected_context.npz"), **y_ref)
    return model


# ===========================================================================
# Streamlining
# ===========================================================================


def step_bert_streamlining(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """BERT-specific streamlining.

    Ported from Brainsmith ``bert_streamlining``. Handles the Mul nodes left
    over from the SoftMax transformations: it moves the Mul down the graph so it
    can be merged into a MultiThreshold node.

    ``cfg.preserve_thresh_shape`` is forwarded to the Add/Mul absorb transforms
    (these accept the ``preserve_thresh_shape`` constructor argument in this
    FINN fork).
    """
    preserve = cfg.preserve_thresh_shape
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(AbsorbAddIntoMultiThreshold(preserve_thresh_shape=preserve))
    model = model.transform(AbsorbMulIntoMultiThreshold(preserve_thresh_shape=preserve))
    model = model.transform(RoundAndClipThresholds())

    model = model.transform(MoveOpPastFork(["Mul"]))

    model = model.transform(MoveScalarMulPastMatMul())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(AbsorbAddIntoMultiThreshold(preserve_thresh_shape=preserve))
    model = model.transform(AbsorbMulIntoMultiThreshold(preserve_thresh_shape=preserve))
    model = model.transform(RoundAndClipThresholds())

    model = model.transform(CollapseRepeatedOp("Mul", lambda x, y: y * x))

    # Final cleanup
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())

    return model


# ===========================================================================
# Shell integration metadata
# ===========================================================================


class ExtractShellIntegrationMetadata(Transformation):
    """Walk the ONNX graph and extract metadata for shell-integration handover.

    Inlined verbatim port of the Brainsmith
    ``brainsmith/transforms/post_proc/extract_shell_integration_metadata.py``
    transform (the optional ``dat_file_to_numpy_array`` weight dump is left
    commented out, as in the original).
    """

    def __init__(self, metadata_file: str):
        super().__init__()
        self.metadata_file: str = metadata_file
        self.md = {}

    def apply(self, model):
        graph = model.graph

        # Search for FINNLoop ops (Does not currently support nested FINNLoops)
        finn_loops = {}
        mlo = False
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                finnloop_op = registry.getCustomOp(node)
                finnloop_body = finnloop_op.get_nodeattr("body")

                mvau_hbm_weights = {}
                extern_idx = 0
                for idx, lb_inp in enumerate(finnloop_body.graph.input):
                    downstream = finnloop_body.find_consumer(lb_inp.name)
                    if downstream.op_type.startswith("MVAU"):
                        mlo = True
                        mvau_hbm_weights[idx] = {}
                        mvau_hbm_weights[idx]["name"] = lb_inp.name
                        # datfile retained for reference; weight dump disabled
                        _ = (
                            f"{finnloop_op.get_nodeattr('code_gen_dir_ipgen')}"
                            f"/memblock_MVAU_rtl_id_{idx}.dat"
                        )
                        mvau_hbm_weights[idx]["extern_idx"] = extern_idx
                        mvau_hbm_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
                        mlo_mvau = registry.getCustomOp(downstream)
                        mvau_hbm_weights[idx]["PE"] = mlo_mvau.get_nodeattr("PE")
                        mvau_hbm_weights[idx]["SIMD"] = mlo_mvau.get_nodeattr("SIMD")
                        mvau_hbm_weights[idx]["MH"] = mlo_mvau.get_nodeattr("MH")
                        mvau_hbm_weights[idx]["MW"] = mlo_mvau.get_nodeattr("MW")
                        mvau_hbm_weights[idx]["weightDataType"] = mlo_mvau.get_nodeattr(
                            "weightDataType"
                        )
                        extern_idx += 1
                finn_loops[node.name] = mvau_hbm_weights
        self.md["mlo"] = mlo
        self.md["finn_loops"] = finn_loops

        # Extract instream widths
        instreams = {}
        for input_tensor in graph.input:
            consumer = model.find_consumer(input_tensor.name)
            inst = registry.getCustomOp(consumer)
            instream = {}
            instream["width"] = inst.get_instream_width()
            instreams[input_tensor.name] = instream
            instream["shape"] = inst.get_normal_input_shape()
            instream["datatype"] = inst.get_input_datatype().name
        self.md["insteams"] = instreams

        # Extract outstream widths
        outstreams = {}
        for output_tensor in graph.output:
            producer = model.find_producer(output_tensor.name)
            inst = registry.getCustomOp(producer)
            outstream = {}
            outstream["width"] = inst.get_outstream_width()
            outstreams[output_tensor.name] = outstream
            outstream["shape"] = inst.get_normal_output_shape()
            outstream["datatype"] = inst.get_output_datatype().name
        self.md["outsteams"] = outstreams

        static_matmuls = {}
        for node in graph.node:
            if node.op_type == "MVAU_rtl":
                inst = registry.getCustomOp(node)
                mm = {}
                mm["MH"] = inst.get_nodeattr("MH")
                mm["MW"] = inst.get_nodeattr("MW")
                mm["SIMD"] = inst.get_nodeattr("SIMD")
                mm["PE"] = inst.get_nodeattr("PE")
                static_matmuls[node.name] = mm
        self.md["static_matmuls"] = static_matmuls

        with open(self.metadata_file, "w") as fp:
            json.dump(self.md, fp, indent=4)

        return (model, False)


def step_shell_metadata_handover(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Extract metadata for the V80 shell-integration process.

    Ported from Brainsmith ``shell_metadata_handover``. Writes
    ``stitched_ip/shell_handover.json`` and copies the reference I/O ``*.npy``
    files into the stitched IP directory for handover.
    """
    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        stitched_ip_dir = os.path.join(cfg.output_dir, "stitched_ip")
        if os.path.isdir(stitched_ip_dir):
            model = model.transform(
                ExtractShellIntegrationMetadata(
                    os.path.join(stitched_ip_dir, "shell_handover.json")
                )
            )
            # copy the reference IO *.npy files into the stitched_ip for handover
            shutil.copy(cfg.verify_input_npy, stitched_ip_dir)
            shutil.copy(cfg.verify_expected_output_npy, stitched_ip_dir)
            return model
        else:
            raise RuntimeError(
                "Error: could not find stitched IP directory so unable to create "
                "metadata. Please ensure this is called after the create_stitched_ip step"
            )
    return model


# ===========================================================================
# V80 deployment (CMake-based shell build)
# ===========================================================================


def _find_v80_shell_dir(cfg: Any) -> Path:
    """Locate the V80 shell source directory (``v80_shell/``).

    Resolution priority:
    1. ``cfg.v80_shell_dir`` (explicit config)
    2. ``BWAVE_DIR`` environment variable (legacy compatibility)
    3. ``<this test folder>/v80_shell/`` (default, self-contained)
    """
    if getattr(cfg, "v80_shell_dir", None):
        path = Path(cfg.v80_shell_dir)
        if path.exists():
            return path
        raise RuntimeError(f"v80_shell_dir specified but not found: {path}")

    if "BWAVE_DIR" in os.environ:
        path = Path(os.environ["BWAVE_DIR"])
        if path.exists():
            return path
        logger.warning(f"BWAVE_DIR set but path not found: {path}")

    default_path = Path(__file__).parent / "v80_shell"
    if default_path.exists():
        return default_path

    raise RuntimeError(
        "V80 shell directory (v80_shell/) not found. "
        "Set v80_shell_dir in config or BWAVE_DIR environment variable."
    )


def _check_tool_available(tool: str) -> bool:
    """Check if a tool is available in PATH."""
    try:
        result = subprocess.run(["which", tool], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _check_vivado_available() -> bool:
    """Check if Vivado is available in PATH or XILINX_VIVADO is set."""
    if "XILINX_VIVADO" in os.environ:
        vivado_path = Path(os.environ["XILINX_VIVADO"]) / "bin" / "vivado"
        if vivado_path.exists():
            return True
    return _check_tool_available("vivado")


def _get_torch_cmake_prefix() -> Optional[str]:
    """Get the CMake prefix path for PyTorch/LibTorch, or None if unavailable."""
    try:
        from torch.utils import cmake_prefix_path

        cmake_path = cmake_prefix_path
        if cmake_path and Path(cmake_path).exists():
            logger.debug(f"Found PyTorch CMake prefix: {cmake_path}")
            return cmake_path
    except ImportError:
        logger.debug("PyTorch not installed or cmake_prefix_path not available")
    except Exception as e:
        logger.debug(f"Could not get PyTorch cmake_prefix_path: {e}")
    return None


def _run_cmake_build(
    build_dir: Path,
    target: str,
    cores: int = 4,
    log_file: Optional[Path] = None,
) -> int:
    """Run a make target and stream output. Returns the exit code."""
    cmd = ["make", "-j", str(cores), target]
    logger.info(f"Running: {' '.join(cmd)} in {build_dir}")

    process = subprocess.Popen(
        cmd,
        cwd=build_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_handle = open(log_file, "w") if log_file else None
    try:
        for line in process.stdout:
            logger.debug(line.rstrip())
            if log_handle:
                log_handle.write(line)
        process.wait()
        return process.returncode
    finally:
        if log_handle:
            log_handle.close()


def _setup_v80_build_environment(cfg: Any):
    """Common setup for the V80 build steps.

    Returns a tuple of
    ``(stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir)``
    or ``None`` if STITCHED_IP is not requested.
    """
    if DataflowOutputType.STITCHED_IP not in cfg.generate_outputs:
        return None

    stitched_ip_dir = (Path(cfg.output_dir) / "stitched_ip").resolve()
    if not stitched_ip_dir.exists():
        raise RuntimeError(
            f"Stitched IP directory not found: {stitched_ip_dir}. "
            "Ensure create_stitched_ip and shell_metadata_handover ran successfully."
        )

    handover_file = stitched_ip_dir / "shell_handover.json"
    if not handover_file.exists():
        raise RuntimeError(
            f"shell_handover.json not found in {stitched_ip_dir}. "
            "Ensure shell_metadata_handover step completed."
        )

    if not _check_tool_available("cmake"):
        raise RuntimeError("CMake not found. Install with: apt-get install cmake")
    if not _check_tool_available("make"):
        raise RuntimeError("Make not found. Install with: apt-get install build-essential")

    v80_shell_dir = _find_v80_shell_dir(cfg)
    logger.info(f"Using V80 shell source: {v80_shell_dir}")

    build_dir = Path(cfg.output_dir) / "v80_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # sw/export subdirectories (needed by hw_project even if BUILD_SW/BUILD_PY are OFF)
    sw_export_dir = build_dir / "sw" / "export"
    (sw_export_dir / "include").mkdir(parents=True, exist_ok=True)
    (sw_export_dir / "config").mkdir(parents=True, exist_ok=True)
    (sw_export_dir / "reference").mkdir(parents=True, exist_ok=True)

    deploy_dir = Path(cfg.output_dir) / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    log_dir = deploy_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir


def _run_cmake_configure(
    cfg: Any,
    stitched_ip_dir: Path,
    v80_shell_dir: Path,
    build_dir: Path,
    log_dir: Path,
) -> None:
    """Run CMake configure if not already done."""
    clock_mhz = getattr(cfg, "v80_clock_mhz", 250)
    compile_cores = getattr(cfg, "v80_compile_cores", 4)

    makefile = build_dir / "Makefile"
    if makefile.exists():
        logger.info("CMake already configured, skipping configure step")
        return

    logger.info("Configuring V80 deployment build...")
    cmake_cmd = [
        "cmake",
        "-S",
        str(v80_shell_dir),
        "-B",
        str(build_dir),
        f"-DCORE_PATH={stitched_ip_dir}",
        f"-DBWAVE_DIR={v80_shell_dir}",
        f"-DACLK_F={clock_mhz}",
        f"-DCOMP_CORES={compile_cores}",
        "-DBUILD_HW=ON",
        "-DBUILD_PY=ON",  # enable so both hw and sw targets are available
        "-DBUILD_SW=OFF",  # C++ runtime not needed for Python workflow
    ]

    torch_cmake_prefix = _get_torch_cmake_prefix()
    if not torch_cmake_prefix:
        fallback_path = Path("/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
        if fallback_path.exists():
            torch_cmake_prefix = str(fallback_path)
            logger.info(f"Using fallback PyTorch CMake prefix: {torch_cmake_prefix}")
    if torch_cmake_prefix:
        cmake_cmd.append(f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}")
        logger.info(f"Using PyTorch CMake prefix: {torch_cmake_prefix}")
    else:
        logger.warning(
            "PyTorch cmake_prefix_path not found. If CMake fails to find Torch, "
            "install PyTorch or set CMAKE_PREFIX_PATH manually."
        )

    logger.info(f"CMake command: {' '.join(cmake_cmd)}")

    cmake_env = os.environ.copy()
    if torch_cmake_prefix:
        existing_prefix = cmake_env.get("CMAKE_PREFIX_PATH", "")
        if existing_prefix:
            cmake_env["CMAKE_PREFIX_PATH"] = f"{torch_cmake_prefix}:{existing_prefix}"
        else:
            cmake_env["CMAKE_PREFIX_PATH"] = torch_cmake_prefix
        logger.info(f"Setting CMAKE_PREFIX_PATH={cmake_env['CMAKE_PREFIX_PATH']}")

    cmake_log = log_dir / "cmake_configure.log"
    with open(cmake_log, "w") as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=cfg.output_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=cmake_env,
        )

    if result.returncode != 0:
        logger.error(f"CMake configure failed. See {cmake_log}")
        raise RuntimeError(f"CMake configure failed. Check {cmake_log} for details.")

    logger.info("CMake configuration complete")


def step_v80_hw_build(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Build the V80 hardware (synthesis and implementation) from stitched IP.

    Ported from Brainsmith ``v80_hw_build``. Configures CMake against the
    stitched IP, runs ``hw_project`` -> ``hw_synth`` -> ``hw_compile``, then
    collects bitstreams/reports into the ``deployment`` folder.

    Optional config attributes (read via getattr with defaults):
    ``v80_clock_mhz`` (250), ``v80_compile_cores`` (4), ``v80_shell_dir``.
    """
    if not _check_vivado_available():
        raise RuntimeError(
            "Vivado not found. Ensure Vivado is in PATH or XILINX_VIVADO is set."
        )

    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_hw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir = env
    compile_cores = getattr(cfg, "v80_compile_cores", 4)

    _run_cmake_configure(cfg, stitched_ip_dir, v80_shell_dir, build_dir, log_dir)

    # hw_project (project creation is not parallelizable)
    logger.info("Creating Vivado project (hw_project)...")
    ret = _run_cmake_build(build_dir, "hw_project", cores=1, log_file=log_dir / "hw_project.log")
    if ret != 0:
        raise RuntimeError(f"hw_project failed. Check {log_dir}/hw_project.log")

    # hw_synth
    logger.info("Running synthesis (hw_synth)...")
    ret = _run_cmake_build(
        build_dir, "hw_synth", cores=compile_cores, log_file=log_dir / "hw_synth.log"
    )
    if ret != 0:
        raise RuntimeError(f"hw_synth failed. Check {log_dir}/hw_synth.log")

    # hw_compile (single core for implementation to avoid PLM "Bad file descriptor")
    logger.info("Running implementation (hw_compile)...")
    ret = _run_cmake_build(build_dir, "hw_compile", cores=1, log_file=log_dir / "hw_compile.log")
    if ret != 0:
        raise RuntimeError(f"hw_compile failed. Check {log_dir}/hw_compile.log")

    logger.info("Hardware build complete")

    # Collect hardware artifacts
    hw_root = build_dir / "hw"

    bitstream_src = hw_root / "bitstreams"
    bitstream_dst = deploy_dir / "bitstreams"
    if bitstream_src.exists():
        shutil.copytree(bitstream_src, bitstream_dst, dirs_exist_ok=True)
        logger.info(f"Copied bitstreams to {bitstream_dst}")
    else:
        logger.warning(f"Bitstream directory not found: {bitstream_src}")

    report_src = hw_root / "reports"
    report_dst = deploy_dir / "reports"
    if report_src.exists():
        shutil.copytree(report_src, report_dst, dirs_exist_ok=True)
        logger.info(f"Copied reports to {report_dst}")

    shutil.copy2(handover_file, deploy_dir / "shell_handover.json")
    logger.info(f"Hardware artifacts collected in {deploy_dir}")

    return model


def step_v80_sw_build(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Build the V80 Python bindings from stitched IP.

    Ported from Brainsmith ``v80_sw_build``. Configures CMake against the
    stitched IP, builds ``sw_python``, then collects the python module / config /
    reference artifacts into the ``deployment`` folder.
    """
    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_sw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir = env
    compile_cores = getattr(cfg, "v80_compile_cores", 4)

    _run_cmake_configure(cfg, stitched_ip_dir, v80_shell_dir, build_dir, log_dir)

    logger.info("Building Python bindings (sw_python)...")
    ret = _run_cmake_build(
        build_dir, "sw_python", cores=compile_cores, log_file=log_dir / "sw_python.log"
    )
    if ret != 0:
        raise RuntimeError(f"sw_python failed. Check {log_dir}/sw_python.log")

    logger.info("Python bindings build complete")

    # Collect software artifacts
    sw_root = build_dir / "sw"

    python_src = sw_root / "python"
    python_dst = deploy_dir / "python"
    if python_src.exists():
        shutil.copytree(python_src, python_dst, dirs_exist_ok=True)
        logger.info(f"Copied Python module to {python_dst}")
    else:
        logger.warning(f"Python module directory not found: {python_src}")

    config_src = sw_root / "export" / "config"
    config_dst = deploy_dir / "config"
    if config_src.exists():
        shutil.copytree(config_src, config_dst, dirs_exist_ok=True)
        logger.info(f"Copied config to {config_dst}")

    ref_src = sw_root / "export" / "reference"
    ref_dst = deploy_dir / "reference"
    if ref_src.exists():
        shutil.copytree(ref_src, ref_dst, dirs_exist_ok=True)
        logger.info(f"Copied reference files to {ref_dst}")

    handover_dst = deploy_dir / "shell_handover.json"
    if not handover_dst.exists():
        shutil.copy2(handover_file, handover_dst)

    logger.info(f"Software artifacts collected in {deploy_dir}")

    model.set_metadata_prop("v80_deployment_dir", str(deploy_dir))

    return model

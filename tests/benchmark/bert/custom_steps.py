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

import logging
import math
import numpy as np
import os
import qonnx.custom_op.registry as registry
import shutil
import subprocess
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
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
from typing import Any, Optional

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
from finn.util.basic import launch_process_helper, resolve_xilinx_tool

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


def _extract_shell_metadata(model: ModelWrapper) -> dict:
    """Walk the ONNX graph and extract the V80 shell-integration metadata.

    Returns the stream/core metadata used to drive the hardware build. Inlined
    port of the Brainsmith ``extract_shell_integration_metadata`` transform,
    returning a plain dict so the parameters can be read directly from the model
    at build time (no on-disk JSON handover required).
    """
    graph = model.graph
    md: dict = {}

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
    md["mlo"] = mlo
    md["finn_loops"] = finn_loops

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
    md["insteams"] = instreams

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
    md["outsteams"] = outstreams

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
    md["static_matmuls"] = static_matmuls

    return md


def step_stage_reference_io(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Stage the reference I/O ``*.npy`` files into the stitched IP directory.

    The reference input/output (produced by ``step_generate_reference_io``) are
    copied next to the stitched IP so the software build can pick them up
    (``CORE_PATH``) and stage them into the deployment for on-hardware
    verification.
    """
    if DataflowOutputType.STITCHED_IP not in cfg.generate_outputs:
        return model

    stitched_ip_dir = os.path.join(cfg.output_dir, "stitched_ip")
    if not os.path.isdir(stitched_ip_dir):
        raise RuntimeError(
            "Error: could not find stitched IP directory. Please ensure this is "
            "called after the create_stitched_ip step."
        )

    shutil.copy(cfg.verify_input_npy, stitched_ip_dir)
    shutil.copy(cfg.verify_expected_output_npy, stitched_ip_dir)
    return model


# ===========================================================================
# V80 deployment (CMake-based shell build)
# ===========================================================================


def _find_v80_shell_dir(cfg: Any) -> Path:
    """Locate the V80 shell source directory (``v80_shell/``).

    The shell lives statically inside the FINN repo next to this module, so it is
    resolved relative to this file. ``cfg.v80_shell_dir`` may override it.
    """
    if getattr(cfg, "v80_shell_dir", None):
        path = Path(cfg.v80_shell_dir)
        if path.exists():
            return path
        raise RuntimeError(f"v80_shell_dir specified but not found: {path}")

    default_path = Path(__file__).parent / "v80_shell"
    if default_path.exists():
        return default_path

    raise RuntimeError(f"V80 shell directory not found at {default_path}")


def _check_tool_available(tool: str) -> bool:
    """Check if a tool is available in PATH."""
    try:
        result = subprocess.run(["which", tool], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _get_torch_cmake_prefix() -> Optional[str]:
    """Get the CMake prefix path for PyTorch/LibTorch, or None if unavailable."""
    try:
        from torch.utils import cmake_prefix_path  # noqa

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
    ``(stitched_ip_dir, v80_shell_dir, build_dir, deploy_dir, log_dir)``
    or ``None`` if STITCHED_IP is not requested.
    """
    if DataflowOutputType.STITCHED_IP not in cfg.generate_outputs:
        return None

    stitched_ip_dir = (Path(cfg.output_dir) / "stitched_ip").resolve()
    if not stitched_ip_dir.exists():
        raise RuntimeError(
            f"Stitched IP directory not found: {stitched_ip_dir}. "
            "Ensure create_stitched_ip ran successfully."
        )

    v80_shell_dir = _find_v80_shell_dir(cfg)
    logger.info(f"Using V80 shell source: {v80_shell_dir}")

    build_dir = Path(cfg.output_dir) / "v80_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # sw/export subdirectories: the HW build writes the generated CSR header here
    # (extract_sys.py -> sw/export/include) and the SW (pybind) CMake reuses them.
    sw_export_dir = build_dir / "sw" / "export"
    (sw_export_dir / "include").mkdir(parents=True, exist_ok=True)
    (sw_export_dir / "config").mkdir(parents=True, exist_ok=True)
    (sw_export_dir / "reference").mkdir(parents=True, exist_ok=True)

    deploy_dir = Path(cfg.output_dir) / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    log_dir = deploy_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return stitched_ip_dir, v80_shell_dir, build_dir, deploy_dir, log_dir


# ---------------------------------------------------------------------------
# V80 hardware build (Vivado driven directly, no CMake)
#
# The helpers below port the hardware half of the former CMake module
# (``cmake/FindV80Shell.cmake``: ``gen_internal`` / ``gen_parsed`` /
# ``gen_scripts`` / ``gen_targets``) into plain Python that renders the TCL/SV
# templates and calls Vivado directly, mirroring how the rest of FINN drives
# Vivado (e.g. ``SlashLink`` / ``VitisLink`` in ``alveo_build.py``).
# ---------------------------------------------------------------------------


def _render_template(text: str, variables: dict) -> str:
    """Substitute ``@KEY@`` / ``${KEY}`` for each known variable.

    Faithful, minimal reimplementation of CMake ``configure_file`` for this
    shell: only the *known* variable names are replaced, in both ``@KEY@`` and
    ``${KEY}`` forms. Any other ``$name`` / ``${other}`` token is left intact so
    that TCL runtime variables (``$proj_dir`` etc.) survive untouched.
    """
    for key, val in variables.items():
        sval = str(val)
        text = text.replace(f"@{key}@", sval)
        text = text.replace("${" + key + "}", sval)
    return text


def _build_v80_hw_vars(cfg: Any, stitched_ip_dir: Path, metadata: dict, build_dir: Path) -> dict:
    """Build the template variable map for the rendered HW scripts.

    Combines the static shell configuration, the device/clock derivations
    (device -> ``FPGA_PART``, clock period) and the values extracted from the
    model (``metadata``: stream/core metadata and the HBM-port allocation).
    """
    aclk_f = int(getattr(cfg, "v80_clock_mhz", 250))
    comp_cores = int(getattr(cfg, "v80_compile_cores", 4))

    # ---- static configuration (FindV80Shell.cmake BUILD_HW cache vars) ----
    v: dict = {
        "CMAKE_PROJECT_NAME": "brainsmith",
        "CMAKE_BINARY_DIR": str(build_dir),
        # Platform
        "FDEV_NAME": "v80",
        # Color codes via `tput` need a real terminal; the build now runs as a
        # captured subprocess (no tty), so disable terminal styling.
        "EN_XTERM": 0,
        # Core
        "CORE_INST_NAME": "finn_design_0",
        "CORE_IP_NAME": "finn_design",
        "CORE_PATH": str(stitched_ip_dir),
        "TB_PATH": 0,
        # Optional scripts
        "SCR_PATH": 0,
        "BD_SCR_PATH": 0,
        # Clocks
        "ACLK_F": aclk_f,
        "DCLK_F": 250,
        "UCLK_0_F": 100,
        "UCLK_1_F": 100,
        "UCLK_2_F": 100,
        # Streams
        "EN_STRM": 0,
        # Memory
        "N_HBM_PL_PORTS_MAX": 16,
        "BW_PL_HBM_RD": 8500,
        "BW_PL_HBM_WR": 100,
        # Offsets (hex strings without 0x)
        "CSR_OFFS": "2000000",
        "HBM_OFFS": "4000000000",
        "HBM_RNG": "20000000",
        # Slicing
        "N_PS_PL_CTRL_REGS": 3,
        "N_PS_PL_DATA_REGS": 3,
        "N_HBM_PL_DATA_REGS": 3,
        # Segmented configuration
        "EN_SEG_RECONFIG": 0,
        # Comp cores
        "COMP_CORES": comp_cores,
    }

    # ---- gen_internal: device + clock derivations ----
    v["FPGA_PART"] = "xcv80-lsva4737-2MHP-e-S"
    v["ACLK_P"] = f"{1000.0 / aclk_f:g}"
    v["ACLK_DP_F"] = aclk_f * 2

    # ---- stream / core metadata (extracted directly from the model) ----
    hcfg = metadata

    gi = hcfg["insteams"]["global_in"]
    go = hcfg["outsteams"]["global_out"]
    si = gi.get("shape", [1])[1:]
    so = go.get("shape", [1])[1:]
    v["ILEN_BITS"] = gi["width"]
    v["OLEN_BITS"] = go["width"]
    v["ILEN"] = math.prod(si) if si else 1
    v["OLEN"] = math.prod(so) if so else 1
    v["GI_DTYPE"] = gi.get("datatype", "UNKNOWN")
    v["GO_DTYPE"] = go.get("datatype", "UNKNOWN")

    mlo = 1 if hcfg.get("mlo", False) else 0
    v["MLO"] = mlo

    ids, mh_mw, dtypes = [], [], []
    for _, loop_body in hcfg.get("finn_loops", {}).items():
        for sid in sorted(loop_body.keys(), key=lambda x: int(x)):
            node = loop_body[sid]
            ids.append(int(sid))
            mh_mw.append(int(node.get("MH", 1)) * int(node.get("MW", 1)))
            dtypes.append(node.get("weightDataType", "UNKNOWN"))
    core_id_count = len(ids)
    v["CORE_ID_COUNT"] = core_id_count
    v["CORE_IDS_CSV"] = ", ".join(str(x) for x in ids)
    v["CORE_MHMW_CSV"] = ", ".join(str(x) for x in mh_mw)
    v["CORE_DTYPES_CSV"] = ", ".join(f'"{dt}"' for dt in dtypes)

    # ---- HBM port allocation (ported from gen_parsed) ----
    n_pl = 0 if v["EN_STRM"] else 2
    if mlo:
        n_pl += 1 + core_id_count
    pl_req = n_pl
    n_max = v["N_HBM_PL_PORTS_MAX"]
    if n_max is None or n_max < 0:
        n_pl, n_noc = pl_req, 0
    elif pl_req > n_max:
        n_pl, n_noc = n_max, pl_req - n_max
    else:
        n_pl, n_noc = pl_req, 0
    n_noc = max(n_noc, 0)
    v["N_HBM_PL_PORTS"] = n_pl
    v["N_HBM_NOC_PORTS"] = n_noc
    v["N_HBM_PORTS"] = n_pl + n_noc

    return v


def _render_hw_scripts(v80_shell_dir: Path, build_dir: Path, variables: dict) -> Path:
    """Render the HW TCL/SV/py templates into the build dir (ports ``gen_scripts``).

    Returns the hardware build root (``<build_dir>/hw``) where the rendered
    scripts live and where Vivado is invoked.
    """
    hw_root = build_dir / "hw"
    hdl_dir = hw_root / "hdl"
    for sub in ("checkpoints", "reports", "bitstreams", "hdl", "iprepo"):
        (hw_root / sub).mkdir(parents=True, exist_ok=True)

    # Path variables consumed by the TCL templates. ``FINN_RTLLIB_DIR`` lets the
    # shell pull the reviewed RTL building blocks (Q_srl, AXIS dwc) straight from
    # ``finn-rtllib`` instead of vendoring copies under ``v80_shell/hw/hdl``.
    v80_shell_dir = v80_shell_dir.resolve()
    variables = dict(variables)
    variables["V80_SHELL_DIR"] = str(v80_shell_dir)
    variables["FINN_RTLLIB_DIR"] = str(v80_shell_dir.parents[3] / "finn-rtllib")

    block_design = "host"  # BLOCK_DESIGN_CNFG
    renders = [
        (v80_shell_dir / "scripts" / "base.tcl.in", hw_root / "base.tcl"),
        (v80_shell_dir / "scripts" / "create_project.tcl.in", hw_root / "create_project.tcl"),
        (v80_shell_dir / "scripts" / "inst_ip.tcl.in", hw_root / "inst_ip.tcl"),
        (v80_shell_dir / "scripts" / f"cr_bd_{block_design}.tcl.in", hw_root / "cr_bd.tcl"),
        (v80_shell_dir / "scripts" / "synth.tcl.in", hw_root / "synth.tcl"),
        (v80_shell_dir / "scripts" / "compile.tcl.in", hw_root / "compile.tcl"),
        (v80_shell_dir / "hw" / "hdl" / "intf" / "pkt_types.sv.in", hdl_dir / "pkt_types.sv"),
        (
            v80_shell_dir / "scripts" / "python" / "gen_bd_wrapper.py.in",
            hw_root / "gen_bd_wrapper.py",
        ),
        (
            v80_shell_dir / "scripts" / "python" / "gen_top_wrapper.py.in",
            hw_root / "gen_top_wrapper.py",
        ),
        (v80_shell_dir / "scripts" / "python" / "gen_role.py.in", hw_root / "gen_role.py"),
        (v80_shell_dir / "scripts" / "python" / "extract_sys.py.in", hw_root / "extract_sys.py"),
    ]
    for src, dst in renders:
        dst.write_text(_render_template(src.read_text(), variables))
    logger.info(f"Rendered {len(renders)} HW scripts into {hw_root}")
    return hw_root


def _run_vivado(tcl_file: Path, hw_root: Path, log_file: Path, stage: str) -> None:
    """Run a single Vivado batch stage from a rendered TCL script.

    Mirrors the direct-Vivado pattern used elsewhere in FINN; the rendered TCL
    scripts ``exit 1`` on failure, so a non-zero return code is surfaced as a
    ``RuntimeError`` pointing at the per-stage log.
    """
    vivado = resolve_xilinx_tool("vivado")
    cmd = [vivado, "-mode", "tcl", "-source", str(tcl_file), "-notrace"]
    logger.info(f"[{stage}] {' '.join(cmd)} (cwd={hw_root})")
    try:
        out, err = launch_process_helper(cmd, cwd=str(hw_root), check=True)
    except subprocess.CalledProcessError as e:
        with open(log_file, "w") as f:
            f.write(e.output or "")
            f.write("\n--- stderr ---\n")
            f.write(e.stderr or "")
        raise RuntimeError(f"V80 hw {stage} failed (vivado). See {log_file}") from e
    with open(log_file, "w") as f:
        f.write(out or "")
        if err:
            f.write("\n--- stderr ---\n")
            f.write(err)


def _run_cmake_configure(
    cfg: Any,
    stitched_ip_dir: Path,
    v80_shell_dir: Path,
    build_dir: Path,
    log_dir: Path,
) -> None:
    """Configure the CMake software build (pybind11 module only).

    The hardware is built by ``step_v80_hw_build`` directly via Vivado, so this
    only configures the Python-bindings half of the shell CMake.
    """
    makefile = build_dir / "Makefile"
    if makefile.exists():
        logger.info("CMake already configured, skipping configure step")
        return

    logger.info("Configuring V80 software (pybind11) build...")
    cmake_cmd = [
        "cmake",
        "-S",
        str(v80_shell_dir),
        "-B",
        str(build_dir),
        f"-DCORE_PATH={stitched_ip_dir}",
        f"-DV80_SHELL_DIR={v80_shell_dir}",
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

    Extracts the shell-integration metadata directly from the model, renders the
    shell's TCL/SV templates and runs Vivado directly (project -> synth ->
    compile), then collects bitstreams and reports into the ``deployment``
    folder. This replaces the former CMake-driven hardware flow with plain
    Python, mirroring how the rest of FINN drives Vivado.

    Optional config attributes (read via getattr with defaults):
    ``v80_clock_mhz`` (250), ``v80_compile_cores`` (4), ``v80_shell_dir``.
    """
    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_hw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, v80_shell_dir, build_dir, deploy_dir, log_dir = env

    # Extract the shell metadata from the model, render the TCL/SV/py templates,
    # then drive Vivado.
    metadata = _extract_shell_metadata(model)
    variables = _build_v80_hw_vars(cfg, stitched_ip_dir, metadata, build_dir)
    hw_root = _render_hw_scripts(v80_shell_dir, build_dir, variables)

    logger.info("Creating Vivado project...")
    _run_vivado(hw_root / "create_project.tcl", hw_root, log_dir / "hw_project.log", "project")

    logger.info("Running synthesis...")
    _run_vivado(hw_root / "synth.tcl", hw_root, log_dir / "hw_synth.log", "synth")

    logger.info("Running implementation...")
    _run_vivado(hw_root / "compile.tcl", hw_root, log_dir / "hw_compile.log", "compile")

    logger.info("Hardware build complete")

    # Collect hardware artifacts
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

    logger.info(f"Hardware artifacts collected in {deploy_dir}")

    return model


def step_v80_sw_build(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Build the V80 Python bindings from stitched IP.

    Ported from Brainsmith ``v80_sw_build``. Configures CMake against the
    stitched IP, builds ``sw_python``, then collects the python module / config /
    reference artifacts into the ``deployment`` folder.
    """
    for tool in ("cmake", "make"):
        if not _check_tool_available(tool):
            raise RuntimeError(f"{tool} not found in PATH; required for the V80 software build.")

    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_sw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, v80_shell_dir, build_dir, deploy_dir, log_dir = env
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

    logger.info(f"Software artifacts collected in {deploy_dir}")

    model.set_metadata_prop("v80_deployment_dir", str(deploy_dir))

    return model

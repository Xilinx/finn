# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import contextlib
import fcntl
import functools
import glob
import json
import numpy as np
import os
import shutil
import time
import torch
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_config_paths,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    find_cached_build,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.pytorch import ToTensor


# model
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + ".onnx")


# The BNN-PYNQ nets are trained on ToTensor-normalized images (raw uint8 / 255).
# Merge that division onto the front of the graph so the build runs inference on
# raw uint8 input, and annotate the global input as UINT8.
def custom_step_add_preproc(model, cfg):
    global_inp_name = model.get_first_global_in()
    ishape = model.get_tensor_shape(global_inp_name)
    preproc_file = os.path.join(make_build_dir("bnn_preproc_"), "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), preproc_file, opset_version=13)
    qonnx_cleanup(preproc_file, out_file=preproc_file)
    pre_model = ModelWrapper(preproc_file)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes())
    pre_model = pre_model.transform(FoldConstants())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    return model


# The Brevitas BNN-PYNQ exports end at the final MatMul (raw N-class vector);
# the verification golden and the LabelSelect-keyed folding/specialize configs
# both expect a top-1 class index. Append a TopK(k=1) so the graph matches --
# convert_to_hw turns it into the configured LabelSelect node.
def custom_step_add_postproc(model, cfg):
    model = model.transform(InsertTopK(k=1))
    return model


verif_steps = [
    "finn_onnx_python",
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


# verification parameters
# tfc and lfc are both MNIST MLPs; cnv is CIFAR-10. The verification I/O is a
# (input image, top-1 class label) pair, and the label is robust to the network /
# quantization (both tfc and lfc classify the shared MNIST sample to the same
# digit), so lfc reuses the tfc MNIST I/O directly -- no lfc-specific golden file.
def get_verify_input_npy(model):
    stem = "tfc_mnist" if model.startswith(("tfc-", "lfc-")) else "cnv_cifar10"
    return os.path.join(verification_io_dir(), stem + "_input.npy")


def get_verify_output_npy(model):
    stem = "tfc_mnist" if model.startswith(("tfc-", "lfc-")) else "cnv_cifar10"
    return os.path.join(verification_io_dir(), stem + "_output.npy")


def configure_build(board, model, output_dir, **overrides):
    # The folding/specialize configs are board-agnostic: they are tuned for the
    # smallest target in the suite (AUP-ZU3_8GB, ZU3EG, no URAM), so they also fit
    # the larger KV260_SOM and U55C parts. One config file per (model) serves every
    # board.
    folding, specialize = benchmark_config_paths(
        "bnn-pynq",
        f"{model}_folding_config",
        f"{model}_specialize_layers",
    )
    cfg = dict(
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=10.0,
        stitched_ip_gen_dcp=False,
        inject_steps_before={
            "step_qonnx_to_finn": [custom_step_add_preproc, custom_step_add_postproc]
        },
        verify_steps=get_verify_steps(verif_steps, board_enabled=(board == BASELINE_BOARD)),
        verify_input_npy=get_verify_input_npy(model),
        verify_expected_output_npy=get_verify_output_npy(model),
        default_swg_exception=True,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


# Baseline-board scheme: correctness (streamline/convert/fold + numeric
# verification) is board/part-independent across the suite's UltraScale+ targets,
# so the baseline board builds every (model, datatype) and is the only one that
# verifies. Every other board builds a single representative that exercises its
# distinct shell flow + fabric fit, with verification OFF. The representatives are
# chosen to equal the retired end2end bnn_pynq sanity configs so coverage is
# preserved: KV260_SOM -> cnv-w1a2 == (w1,a2,cnv,KV260_SOM); U55C -> cnv-w2a2 ==
# (w2,a2,cnv,U55C). The baseline additionally covers lfc-w1a1 == (w1,a1,lfc,
# AUP-ZU3_8GB), the last end2end sanity config.
BASELINE_BOARD = "AUP-ZU3_8GB"
_BASELINE_MODELS = [
    "tfc-w1a1",
    "tfc-w1a2",
    "tfc-w2a2",
    "cnv-w1a1",
    "cnv-w1a2",
    "cnv-w2a2",
    "lfc-w1a1",
    "lfc-w1a2",
]
_EXTRA_BUILDS = [
    ("KV260_SOM", "cnv-w1a2"),
    ("U55C", "cnv-w2a2"),
]

BNN_BUILDS = [(BASELINE_BOARD, m) for m in _BASELINE_MODELS] + _EXTRA_BUILDS


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", BNN_BUILDS)
def test_bnnpynq(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_bnn-pynq_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    model_file = get_model_file(model)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one. The bitfile
    # artifacts depend on the shell flow (Zynq boards -> .bit/.hwh, U55C -> .xclbin).
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

# Feature-test helpers, replicated in every model's test file so that each file is
# self-contained (only upstream's benchmark helpers are shared).

# estimate-only builds stop after the phase that writes the estimate reports
ESTIMATE_OUTPUTS = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
ESTIMATE_PHASE = "phase_optimize_hardware"

# Feature builds stop at stitched IP: no bitfile, rtlsim throughput only.
FEATURE_OUTPUTS = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]

# the phase whose output checkpoint the feature builds start from
CHECKPOINT_PHASE = "phase_convert_to_hardware"


def feature_steps(steps=None):
    """The part of a model's step list that a feature build runs (everything
    after ``CHECKPOINT_PHASE``); ``None`` stands for the default phase list."""
    steps = list(build_cfg.default_build_dataflow_steps if steps is None else steps)
    return steps[steps.index(CHECKPOINT_PHASE) + 1 :]


def cached_build(prefix, build_fn, done_file):
    """Return the ``FINN_BUILD_DIR`` build dir ``prefix*`` holding ``done_file``,
    building it with ``build_fn(output_dir)`` if there is none. A file lock keeps
    concurrent tests of the same model from building it twice."""
    build_dir = os.environ["FINN_BUILD_DIR"]
    with open(os.path.join(build_dir, prefix + "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output_dir = find_cached_build(prefix)
        if output_dir is None or not os.path.isfile(os.path.join(output_dir, done_file)):
            output_dir = make_build_dir(prefix)
            build_fn(output_dir)
        return output_dir


def resume_checkpoint(ref_dir, output_dir):
    """Prepare ``output_dir`` to continue ``ref_dir``'s build after
    ``CHECKPOINT_PHASE`` and return the checkpoint to pass as the model file.
    Stitched-IP verification needs the dataflow parent model next to it."""
    os.makedirs(os.path.join(output_dir, "intermediate_models"), exist_ok=True)
    parent = "intermediate_models/dataflow_parent.onnx"
    shutil.copy(os.path.join(ref_dir, parent), os.path.join(output_dir, parent))
    return os.path.join(ref_dir, "intermediate_models", CHECKPOINT_PHASE + ".onnx")


@contextlib.contextmanager
def timed_steps(step_names=("step_target_fps_parallelization", "step_set_fifo_depths")):
    """Record the wall time (s) of the named build steps into the yielded dict.
    The builder only times whole phases, so the steps are wrapped where the
    phases look them up."""
    import finn.builder.build_dataflow_phases as phases

    times, originals = {}, {name: getattr(phases, name) for name in step_names}

    def timed(fn):
        @functools.wraps(fn)
        def wrapper(model, cfg):
            start = time.time()
            try:
                return fn(model, cfg)
            finally:
                times[fn.__name__] = times.get(fn.__name__, 0.0) + time.time() - start

        return wrapper

    for name, fn in originals.items():
        setattr(phases, name, timed(fn))
    try:
        yield times
    finally:
        for name, fn in originals.items():
            setattr(phases, name, fn)


def estimated_throughput(output_dir):
    """Analytic throughput (fps) from the estimate reports, None if missing."""
    est = _load_json(os.path.join(output_dir, "report", "estimate_network_performance.json"))
    return est.get("estimated_throughput_fps") if est else None


def rtlsim_throughput(output_dir):
    """Steady-state stitched-IP rtlsim throughput (fps), None if missing."""
    perf = _load_json(os.path.join(output_dir, "report", "rtlsim_performance.json"))
    return perf.get("stable_throughput[images/s]") if perf else None


def verification_passed(output_dir, step="stitched_ip_rtlsim"):
    """True if ``step`` verified and every verified frame matched the golden output."""
    results = glob.glob(os.path.join(output_dir, "verification_output", f"verify_{step}_*"))
    return bool(results) and all("_SUCCESS." in os.path.basename(r) for r in results)


def feature_metrics(output_dir, ref_dir, **extra):
    """Collect a feature build's numbers (throughput, FIFO storage, padding DWCs,
    folding/FIFO step times), store them as ``report/feature_metrics.json`` and
    return them."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    est, ref = estimated_throughput(output_dir), estimated_throughput(ref_dir)
    rtlsim = rtlsim_throughput(output_dir)
    row = dict(
        output_dir=output_dir,
        est_fps=est,
        ref_est_fps=ref,
        rtlsim_fps=rtlsim,
        rtlsim_vs_est=rtlsim / est if rtlsim and est else None,
        est_vs_ref=est / ref if est and ref else None,
        verified=verification_passed(output_dir),
    )
    for step in ["step_set_fifo_depths", "step_minimize_bit_width"]:
        model_file = os.path.join(output_dir, "intermediate_models", step + ".onnx")
        if os.path.isfile(model_file):
            break
    if os.path.isfile(model_file):
        fifo_bits, pad_dwcs = 0, 0
        for node in ModelWrapper(model_file).graph.node:
            inst = getCustomOp(node) if node.domain.startswith("finn") else None
            if node.op_type.startswith("StreamingFIFO"):
                fifo_bits += inst.get_nodeattr("depth") * inst.get_instream_width()
            elif node.op_type.startswith("StreamingDataWidthConverter"):
                n_in = int(np.prod(inst.get_normal_input_shape()))
                pad_dwcs += n_in != int(np.prod(inst.get_normal_output_shape()))
        row.update(fifo_kbit=fifo_bits / 1024, pad_dwcs=pad_dwcs)
    row.update(extra)
    os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)
    with open(os.path.join(output_dir, "report", "feature_metrics.json"), "w") as f:
        json.dump(row, f, indent=2)
    return row


def _load_json(path):
    """Load a JSON file, None if it is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


FEATURE_BUILDS = [
    pytest.param(board, model, marks=pytest.mark.xdist_group(model)) for board, model in BNN_BUILDS
]


def reference_estimate(board, model):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            model,
            output_dir,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(get_model_file(model), cfg) == 0

    return cached_build(
        f"est_bnn-pynq_{model}_{board}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build(board, model, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board, model)
    output_dir = make_build_dir(f"{name}_bnn-pynq_{model}_{board}_")
    cfg = configure_build(
        board,
        model,
        output_dir,
        steps=feature_steps(None),
        generate_outputs=ESTIMATE_OUTPUTS if estimate_only else FEATURE_OUTPUTS,
        stop_step=ESTIMATE_PHASE if estimate_only else None,
        verify_steps=[] if estimate_only else ["stitched_ip_rtlsim"],
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        **overrides,
    )
    # make_benchmark_cfg drops None values: apply them afterwards (e.g. no folding JSON)
    for key, value in overrides.items():
        if value is None:
            setattr(cfg, key, None)
    with timed_steps() as step_times:
        ret = build.build_dataflow_cfg(resume_checkpoint(ref_dir, output_dir), cfg)
    metrics = feature_metrics(output_dir, ref_dir, test=name, step_times_s=step_times)
    assert ret == 0, f"build failed, see {output_dir}/build_dataflow.log"
    if not estimate_only:
        assert metrics["verified"], "stitched-IP rtlsim output differs from the golden output"
        assert (metrics["rtlsim_vs_est"] or 0) >= 0.98, "rtlsim %s fps, estimate %s fps" % (
            metrics["rtlsim_fps"],
            metrics["est_fps"],
        )
    return metrics


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.fifo_sizing
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board,model", FEATURE_BUILDS)
def test_bnnpynq_fifo_sizing(board, model, strategy):
    run_feature_build(
        board, model, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy
    )

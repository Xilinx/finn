# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import contextlib
import fcntl
import functools
import glob
import json

# custom steps for mobilenetv1
import os
import shutil
import time
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
from custom_steps_mobilenet import step_mobilenet_streamline

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_fd = benchmark_root()

# model
model_name = "mobilenetv1-w4a4"
model_file = os.path.join(build_fd, "models", "%s_pre_post_tidy_opset-11.onnx" % model_name)


# verification parameters
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")


def select_verif_steps(platform):
    steps = [
        "streamlined_python",
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
    ]
    # Skip stitched_ip_rtlsim for ZCU104 due to URAM initialization issues
    if platform == "ZCU104":
        steps.remove("stitched_ip_rtlsim")
    return steps


# Build steps: the MobileNet-specific streamline (handles depthwise convs via
# MoveMulPastDWConv, QuantAvgPool datalayout, flatten reordering, then conv
# lowering) replaces the standard streamline phase. The rest uses the phase-based
# default flow.
build_steps = [
    "phase_prepare_model",
    step_mobilenet_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


# select target clock frequency
def select_clk_period(platform):
    if platform in ["ZCU104"]:
        return 5.4
    elif platform in ["U55C"]:
        return 3.0


def configure_build(board, output_dir, **overrides):
    folding, specialize = benchmark_config_paths(
        "mobilenet_v1",
        f"mobilenet_folding_config_{board}",
        f"mobilenet_specialize_layers_{board}",
    )
    cfg = dict(
        steps=build_steps,
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=select_clk_period(board),
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(select_verif_steps(board)),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize(
    "board",
    [
        "ZCU104",
        "U55C",
    ],
)
def test_mobilenetv1(board, bench_recorder):
    # Create output directory only when test actually runs
    output_dir = make_build_dir(f"build_mobilenet_v1_{board}_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products (and the per-step verification
    # markers) are present, reporting every missing artifact at once instead of
    # aborting on the first one. The bitfile artifacts depend on the shell flow
    # (ZCU104 -> .bit/.hwh via Vivado/Zynq, U55C -> .xclbin via Vitis/Alveo).
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing ----------------------------------------------
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
def timed_steps(step_names=("step_set_fifo_depths",)):
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


def feature_metrics(output_dir, **extra):
    """Collect a feature build's numbers (throughput, FIFO storage, FIFO sizing
    time), store them as ``report/feature_metrics.json`` and return them."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    est, rtlsim = estimated_throughput(output_dir), rtlsim_throughput(output_dir)
    row = dict(
        output_dir=output_dir,
        est_fps=est,
        rtlsim_fps=rtlsim,
        rtlsim_vs_est=rtlsim / est if rtlsim and est else None,
        verified=verification_passed(output_dir),
    )
    for step in ["step_set_fifo_depths", "step_minimize_bit_width"]:
        model_file = os.path.join(output_dir, "intermediate_models", step + ".onnx")
        if os.path.isfile(model_file):
            break
    if os.path.isfile(model_file):
        fifo_bits = 0
        for node in ModelWrapper(model_file).graph.node:
            if node.op_type.startswith("StreamingFIFO"):
                inst = getCustomOp(node)
                fifo_bits += inst.get_nodeattr("depth") * inst.get_instream_width()
        row.update(fifo_kbit=fifo_bits / 1024)
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


# ZCU104 is left out: its stitched-IP rtlsim gives X outputs (URAM init)
FEATURE_BUILDS = ["U55C"]


def reference_estimate(board):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            output_dir,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(model_file, cfg) == 0

    return cached_build(
        f"est_mobilenet_v1_{board}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build(board, name, **overrides):
    """Resume the reference checkpoint with ``overrides`` and check the stitched
    IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board)
    output_dir = make_build_dir(f"{name}_mobilenet_v1_{board}_")
    cfg = configure_build(
        board,
        output_dir,
        steps=feature_steps(build_steps),
        generate_outputs=FEATURE_OUTPUTS,
        verify_steps=["stitched_ip_rtlsim"],
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        **overrides,
    )
    with timed_steps() as step_times:
        ret = build.build_dataflow_cfg(resume_checkpoint(ref_dir, output_dir), cfg)
    metrics = feature_metrics(output_dir, test=name, step_times_s=step_times)
    assert ret == 0, f"build failed, see {output_dir}/build_dataflow.log"
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
@pytest.mark.xdist_group("mobilenet_v1")
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_mobilenetv1_fifo_sizing(board, strategy):
    run_feature_build(board, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy)

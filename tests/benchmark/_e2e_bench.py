# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared helpers for the end-to-end (stitched-IP rtlsim) benchmark suite.

Unlike the estimate-only harnesses (``_feature_bench``), these tests run each
finn-examples model through HLS/RTL codegen, IP synthesis, FIFO sizing and
stitched-IP generation, up to and including ``step_measure_rtlsim_performance``
(no place-and-route, no bitfile). Every flow dumps one JSON per (flow, model)
into a shared results directory; ``e2e_report.py`` assembles the comparison
table (step runtimes, rtlsim throughput, resource estimates).

The flows:

=========  ==================================================================
baseline   committed folding JSON + rtlsim-based FIFO sizing + stock DWCs
fifo       committed folding JSON + ANALYTIC (tree-model) FIFO sizing
folding    SetFolding optimizer (padding enabled) + rtlsim-based FIFO sizing
dwc        committed folding JSON + rtlsim FIFO sizing + generalized DWCs
combined   optimizer folding (padded) + ANALYTIC sizing + generalized DWCs
aligner    baseline flow + align_labels (AlignLabels side-channel output)
=========  ==================================================================

The ``baseline`` and ``dwc`` flows use an *identical* build configuration; they
differ only in which tree they run on (stock DWC vs generalized DWC), which is
why ``test_e2e_baseline`` refuses to run on a tree that carries the generalized
DWC (set ``FINN_E2E_ALLOW_MIXED=1`` to override). Each feature test self-skips
on a tree without its feature, so the whole suite can be pointed at any env/*
assembly and runs exactly the flows that tree supports.
"""

import _feature_bench as fb
import dataclasses
import json
import os
import subprocess
import time

BENCH_DIR = fb.BENCH_DIR
REPO_ROOT = fb.REPO_ROOT

# Everything the e2e comparison runs on. resnet50 stays excluded (deferred:
# core/streamline changes); kws is excluded because it fails dev's dataflow
# contiguity check before any of the features under test are reached; gtsrb is
# covered by the bnn-pynq cnv family.
E2E_MODELS = [
    ("cybersecurity-mlp", "Pynq-Z1", None),
    ("bnn-pynq", "Pynq-Z1", "tfc-w1a1"),
    ("bnn-pynq", "Pynq-Z1", "tfc-w1a2"),
    ("bnn-pynq", "Pynq-Z1", "tfc-w2a2"),
    ("bnn-pynq", "Pynq-Z1", "cnv-w1a1"),
    ("bnn-pynq", "Pynq-Z1", "cnv-w1a2"),
    ("bnn-pynq", "Pynq-Z1", "cnv-w2a2"),
    ("vgg10-radioml", "ZCU104", None),
    ("mobilenet_v1", "ZCU104", None),
]

# rtlsim batch size: >1 so the builder reports steady-state throughput
# (stable_throughput strips the pipeline-fill latency). mobilenet's stitched-IP
# rtlsim is orders of magnitude slower per frame, so it gets the minimum that
# still separates latency from throughput.
RTLSIM_BATCH_DEFAULT = 8
RTLSIM_BATCH = {"mobilenet_v1": 2, "vgg10-radioml": 4}

model_id = fb.model_id


def results_dir():
    d = os.environ.get("FINN_E2E_RESULTS", os.path.join(REPO_ROOT, "e2e_results"))
    os.makedirs(d, exist_ok=True)
    return d


def build_root():
    """Root for build output dirs. The repo lives on the large disk; the
    container / (and /tmp) is nearly full, so never default to /tmp here."""
    d = os.environ.get("FINN_E2E_BUILD_ROOT", os.path.join(REPO_ROOT, "e2e_build"))
    os.makedirs(d, exist_ok=True)
    # FINN scatters scratch dirs (vivado projects, hls) into FINN_BUILD_DIR
    os.environ.setdefault("FINN_BUILD_DIR", os.path.join(d, "tmp"))
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
    return d


def check_tree_is_imported():
    """Guard against the easy-install.pth gotcha: 'finn' can silently resolve to
    a different source tree than this repo. All e2e numbers would then describe
    the wrong code, so fail loudly and say how to fix it."""
    import finn.builder

    mod_file = os.path.abspath(finn.builder.__file__)
    if not mod_file.startswith(REPO_ROOT + os.sep):
        raise RuntimeError(
            f"'finn' imports from {mod_file}, not from this tree ({REPO_ROOT}). "
            f"Run with PYTHONPATH={REPO_ROOT}/src prepended."
        )


# ---- feature-presence guards (each flow runs only where its feature exists) --


def _cfg_fields():
    import finn.builder.build_dataflow_config as build_cfg

    return {f.name for f in dataclasses.fields(build_cfg.DataflowBuildConfig)}


def has_analytic_fifo():
    try:
        import finn.builder.build_dataflow_config as build_cfg

        return hasattr(build_cfg.AutoFIFOSizingMethod, "ANALYTIC") and hasattr(
            build_cfg, "TAVGenerationMethod"
        )
    except ImportError:
        return False


def has_folding_optimizer():
    return "folding_style" in _cfg_fields()


def has_generalized_dwc():
    # the generalized DWC splits the single 'shape' attr into in_shape/out_shape
    # and grows word-count helpers; probe for one of those methods
    from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
        StreamingDataWidthConverter,
    )

    return hasattr(StreamingDataWidthConverter, "get_num_in_words")


def has_label_aligner():
    return "align_labels" in _cfg_fields()


def require(cond, what):
    import pytest

    if not cond:
        pytest.skip(f"{what} not available on this tree")


def forbid_generalized_dwc():
    """The baseline flow must measure the stock DWCs. On a tree that carries the
    generalized DWC the 'baseline' would silently build generalized DWCs and the
    comparison table would compare the feature against itself."""
    import pytest

    if has_generalized_dwc() and os.environ.get("FINN_E2E_ALLOW_MIXED") != "1":
        pytest.skip(
            "tree carries the generalized DWC; run the baseline flow on the "
            "baseline env (or set FINN_E2E_ALLOW_MIXED=1 to force)"
        )


# ---- build configuration -----------------------------------------------------


def make_e2e_cfg(mod, board, model, out):
    """The model's own DataflowBuildConfig, normalised for a cached rtlsim run:
    estimate reports + stitched IP + rtlsim performance, nothing that needs
    synthesis or a board (no dcp, no bitfile, no driver)."""
    import finn.builder.build_dataflow_config as build_cfg

    cfg = fb.make_base_cfg(mod, board, model, out)
    cfg.generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    ]
    cfg.stitched_ip_gen_dcp = False
    cfg.rtlsim_batch_size = RTLSIM_BATCH.get(_model_dir_of(mod), RTLSIM_BATCH_DEFAULT)
    cfg.steps = rtlsim_step_list(cfg)
    return cfg


def _model_dir_of(mod):
    return os.path.basename(os.path.dirname(os.path.abspath(mod.__file__)))


def rtlsim_step_list(cfg):
    """Truncate the model's step list right after rtlsim performance measurement.

    Phase-based lists (bnn/cybersecurity default steps) need no truncation:
    phase_generate_outputs consults generate_outputs and, without BITFILE/OOC
    requests, stops after stitched IP + rtlsim measurement. Explicit lists
    (vgg10/mobilenet) are cut after step_measure_rtlsim_performance.
    """
    steps = cfg.steps or fb._default_steps()
    names = [s if isinstance(s, str) else getattr(s, "__name__", "") for s in steps]
    if "step_measure_rtlsim_performance" in names:
        return list(steps[: names.index("step_measure_rtlsim_performance") + 1])
    if "phase_generate_outputs" in names:
        return list(steps[: names.index("phase_generate_outputs") + 1])
    raise RuntimeError(f"cannot find an rtlsim truncation point in {names}")


# flow mutators. Each takes the normalised cfg and adjusts exactly the knobs
# that define the flow, so any difference between two flows is easy to audit.


def flow_use_json_folding(cfg):
    cfg.target_fps = None  # committed folding_config_file drives the folding


def flow_use_rtlsim_fifo_sizing(cfg):
    import finn.builder.build_dataflow_config as build_cfg

    cfg.auto_fifo_depths = True
    cfg.auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM


def flow_use_analytic_fifo_sizing(cfg):
    import finn.builder.build_dataflow_config as build_cfg

    cfg.auto_fifo_depths = True
    cfg.auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.ANALYTIC
    cfg.tav_generation_strategy = build_cfg.TAVGenerationMethod.TREE_MODEL
    cfg.tav_utilization_strategy = build_cfg.TAVUtilizationMethod.CONSERVATIVE_RELAXATION


def flow_use_folding_optimizer(cfg, target_fps, with_padding=True, fifo_heuristic=False):
    cfg.folding_config_file = None
    cfg.folding_style = "optimizer"
    cfg.folding_effort = int(os.environ.get("FINN_E2E_FOLDING_EFFORT", "100"))
    cfg.folding_maximum_padding = 6 if with_padding else 0
    cfg.enable_folding_dwc_heuristic = True
    cfg.enable_folding_fifo_heuristic = fifo_heuristic
    cfg.target_fps = target_fps
    return cfg


def matched_target_fps(mod, board, model, key):
    """Throughput target for the optimizer flows: the committed folding JSON's
    own estimated fps. Derived from a cheap cached estimate-only build so the
    optimizer competes at matched throughput instead of an arbitrary target."""
    out = os.path.join(build_root(), f"est_json_{key}")
    cfg = fb.make_base_cfg(mod, board, model, out)
    cfg.target_fps = None
    cfg.steps = fb.estimate_step_list(cfg, with_fifo_sizing=False)
    assert fb.run_estimate_build(
        fb.get_model_file(mod, model), cfg
    ), f"estimate build (json folding) failed for {key}"
    json_cycles = fb.network_max_cycles(out)
    clock_hz = 1e9 / cfg.synth_clk_period_ns
    return int(clock_hz / json_cycles), json_cycles


# ---- running + metric collection --------------------------------------------


def run_e2e_build(model_file, cfg):
    """Cached rtlsim build. Returns wall-clock seconds (0.0 for a cache hit)."""
    import finn.builder.build_dataflow as build

    os.chdir(REPO_ROOT)
    report = os.path.join(cfg.output_dir, "report", "rtlsim_performance.json")
    if os.path.isfile(report):
        return 0.0
    os.makedirs(cfg.output_dir, exist_ok=True)
    t0 = time.time()
    rc = build.build_dataflow_cfg(model_file, cfg)
    elapsed = time.time() - t0
    assert rc == 0, f"build failed (rc={rc}) in {cfg.output_dir}"
    return elapsed


def _load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def collect_metrics(out):
    """One metrics dict per finished build: rtlsim performance, estimate-level
    resource totals, per-step runtimes and total FIFO storage."""
    rep = os.path.join(out, "report")
    rtlsim = _load_json(os.path.join(rep, "rtlsim_performance.json"))
    est_res = _load_json(os.path.join(rep, "estimate_layer_resources.json"))
    est_perf = _load_json(os.path.join(rep, "estimate_network_performance.json"))
    step_times = _load_json(os.path.join(out, "time_per_step.json"))

    sized = fb.latest_intermediate(out, "step_set_fifo_depths")
    metrics = {
        "rtlsim": rtlsim,
        "resources_estimate": (est_res or {}).get("total"),
        "estimate_max_cycles": (est_perf or {}).get("max_cycles"),
        "step_times_s": step_times,
        "fifo_kb": round(fb.total_fifo_kb(sized), 3) if sized else None,
    }
    return metrics


def git_describe():
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        return f"{branch}@{head}"
    except Exception:
        return "unknown"


def write_result(flow, key, metrics, extra=None):
    payload = {
        "flow": flow,
        "model": key,
        "tree": git_describe(),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
        **metrics,
    }
    path = os.path.join(results_dir(), f"{flow}__{key}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load_result(flow, key):
    return _load_json(os.path.join(results_dir(), f"{flow}__{key}.json"))


def stable_throughput(metrics_or_result):
    """Steady-state rtlsim throughput. Prefer fclk/interval_cycles: the builder's
    stable_throughput divides by (total - latency) cycles, which degenerates when
    the whole batch drains within the pipeline-fill window (tiny models)."""
    rtlsim = (metrics_or_result or {}).get("rtlsim") or {}
    interval = rtlsim.get("interval_cycles")
    fclk_mhz = rtlsim.get("fclk[mhz]")
    if interval and fclk_mhz and interval > 0:
        return fclk_mhz * 1e6 / interval
    return rtlsim.get("stable_throughput[images/s]") or rtlsim.get("throughput[images/s]")


def assert_no_throughput_regression(flow, key, metrics, rel_tol=0.05):
    """Hard project constraint: a feature flow must never degrade throughput.
    Compares against the baseline flow's recorded result when one exists;
    silently passes otherwise (the report generator flags missing baselines)."""
    base = load_result("baseline", key)
    if base is None:
        return
    base_tp, tp = stable_throughput(base), stable_throughput(metrics)
    if base_tp and tp:
        assert tp >= base_tp * (1 - rel_tol), (
            f"{flow}[{key}]: rtlsim throughput {tp:.1f} img/s regressed vs "
            f"baseline {base_tp:.1f} img/s (tol {rel_tol:.0%})"
        )


def run_flow(entry, flow, mutate, extra=None):
    """The shared body of every e2e test: build, collect, dump, sanity-check."""
    model_dir, board, model = entry
    key = model_id(entry)
    check_tree_is_imported()

    mod = fb.load_bench_module(model_dir)
    model_file = fb.get_model_file(mod, model)
    out = os.path.join(build_root(), f"e2e_{flow}_{key}")

    cfg = make_e2e_cfg(mod, board, model, out)
    mutate(cfg)
    wall = run_e2e_build(model_file, cfg)

    metrics = collect_metrics(out)
    assert metrics["rtlsim"], f"{flow}[{key}]: no rtlsim_performance.json produced"
    assert (stable_throughput(metrics) or 0) > 0, f"{flow}[{key}]: zero rtlsim throughput"
    write_result(flow, key, metrics, extra={**(extra or {}), "build_wall_s": round(wall, 1)})
    return metrics

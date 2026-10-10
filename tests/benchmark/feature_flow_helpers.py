# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared harness for the feature-flow benchmark tests (folding, FIFO sizing, DWC).

Each feature test checks that one transformation keeps a finn-examples model
functionally correct, and records how long the transformation takes:

1. The model is built once per (model, board) up to ``phase_convert_to_hardware``
   and cached in ``FINN_BUILD_DIR`` (found again with ``find_cached_build``),
   together with a golden input/output pair from the streamlined model.
2. Every flow variant copies that checkpoint into a fresh build dir and resumes
   from ``phase_optimize_hardware`` with its own overrides, through stitched IP
   and rtlsim performance measurement (no bitfile).
3. The build verifies against the golden pair (folded cppsim + stitched-IP
   rtlsim), and the folding / FIFO sizing steps are timed individually.

Every run appends one JSON row to ``FINN_BUILD_DIR/feature_flow_results/<flow>.jsonl``.

Environment:
    FEATURE_FLOW_VERIFY: comma-separated verification steps
        (default ``folded_hls_cppsim,stitched_ip_rtlsim``).
    FEATURE_FLOW_FRAMES: number of verification frames (default 1).
    FEATURE_FLOW_REBUILD_CACHE: when truthy, rebuild the cached checkpoints.
    FEATURE_FLOW_IP_CACHE: reuse synthesized HLS IP across runs (default 1).
    FEATURE_FLOW_QUICK: when truthy, stop after phase_optimize_hardware (folding,
        bit width minimization, folded cppsim verification, FIFO sizing, estimates):
        no IP generation or rtlsim, for fast checks of a transformation.
"""

import pytest

import contextlib
import fcntl
import functools
import glob
import hashlib
import importlib.util
import json
import numpy as np
import os
import shutil
import time
from benchmark_helpers import benchmark_root, find_cached_build
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow_phases as build_phases
import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.general import ApplyConfig
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_hls_node

# Target boards: the real FPGAs the tests can eventually run on.
FEATURE_BOARDS = ["ZCU104", "U55C"]

# (benchmark dir, model): small models that build quickly. ``model`` selects the
# network inside multi-model benchmark dirs (bnn-pynq) and is None otherwise.
FEATURE_MODELS = [
    ("bnn-pynq", "tfc-w1a1"),
    ("bnn-pynq", "cnv-w1a1"),
    ("cybersecurity-mlp", None),
]

CACHE_PHASE = "phase_convert_to_hardware"
RESUME_PHASE = "phase_optimize_hardware"

# build steps whose wall time is recorded per run
TIMED_STEPS = [
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_set_fifo_depths",
]

FLOW_OUTPUTS = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]


def _truthy(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def model_key(model_dir, model):
    return model_dir if model is None else model


def feature_cases(models=None):
    """pytest params over (model_dir, model, board)."""
    return [
        pytest.param(model_dir, model, board, id="%s-%s" % (model_key(model_dir, model), board))
        for model_dir, model in (FEATURE_MODELS if models is None else models)
        for board in FEATURE_BOARDS
    ]


def load_bench_module(model_dir):
    """Import ``<model_dir>/test_build_<model>.py`` (hyphenated, so not importable by name)."""
    cands = glob.glob(os.path.join(benchmark_root(), model_dir, "test_build_*.py"))
    assert len(cands) == 1, "expected one test_build_*.py in %s, got %s" % (model_dir, cands)
    spec = importlib.util.spec_from_file_location("bench_" + model_dir.replace("-", "_"), cands[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_file(mod, model):
    return mod.get_model_file(model) if model is not None else mod.model_file


def folding_config_file(model_dir, model, board):
    """The model's committed folding JSON, preferring one for ``board``. Models with
    configs only for a smaller part reuse them on the larger target boards."""
    cands = sorted(glob.glob(os.path.join(benchmark_root(), model_dir, "folding_config", "*.json")))
    if model is not None:
        cands = [c for c in cands if os.path.basename(c).startswith(model + "_")]
    assert cands, "no folding config for %s" % model_key(model_dir, model)
    for cand in cands:
        if board in os.path.basename(cand):
            return cand
    return cands[0]


def configure(mod, model, board, output_dir, **overrides):
    if model is not None:
        return mod.configure_build(board, model, output_dir, **overrides)
    return mod.configure_build(board, output_dir, **overrides)


@contextlib.contextmanager
def _file_lock(path):
    with open(path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _cache_complete(cache_dir):
    return all(
        os.path.isfile(os.path.join(cache_dir, f))
        for f in [
            "intermediate_models/%s.onnx" % CACHE_PHASE,
            "intermediate_models/dataflow_parent.onnx",
            "golden_input.npy",
            "golden_output.npy",
        ]
    )


def cached_checkpoint(model_dir, model, board):
    """Return the cache dir holding the (model, board) checkpoint after
    ``phase_convert_to_hardware`` plus a golden I/O pair, building it if needed."""
    prefix = "featureflow_cache_%s_%s_" % (model_key(model_dir, model), board)
    build_dir = os.environ["FINN_BUILD_DIR"]
    with _file_lock(os.path.join(build_dir, prefix + "lock")):
        cache_dir = find_cached_build(prefix)
        while cache_dir is not None and (
            _truthy("FEATURE_FLOW_REBUILD_CACHE") or not _cache_complete(cache_dir)
        ):
            shutil.rmtree(cache_dir)
            cache_dir = find_cached_build(prefix)
        if cache_dir is not None:
            return cache_dir

        cache_dir = make_build_dir(prefix)
        mod = load_bench_module(model_dir)
        cfg = configure(
            mod,
            model,
            board,
            cache_dir,
            stop_step=CACHE_PHASE,
            folding_config_file=folding_config_file(model_dir, model, board),
            verify_steps=[],
            generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
            save_intermediate_models=True,
        )
        start = time.time()
        assert build.build_dataflow_cfg(model_file(mod, model), cfg) == 0, (
            "cache build failed, see %s/build_dataflow.log" % cache_dir
        )
        cache_time = time.time() - start

        # golden pair from the streamlined model, the reference for every flow
        ref = ModelWrapper(os.path.join(cache_dir, "intermediate_models", "step_streamline.onnx"))
        iname = ref.get_first_global_in()
        ishape = ref.get_tensor_shape(iname)
        frames = int(os.environ.get("FEATURE_FLOW_FRAMES", "1"))
        rng_state = np.random.get_state()
        np.random.seed(42)
        x = gen_finn_dt_tensor(ref.get_tensor_datatype(iname), [frames] + list(ishape[1:]))
        np.random.set_state(rng_state)
        y = np.concatenate(
            [
                oxe.execute_onnx(ref, {iname: x[i : i + 1]})[ref.get_first_global_out()]
                for i in range(frames)
            ]
        )
        np.save(os.path.join(cache_dir, "golden_input.npy"), x)
        np.save(os.path.join(cache_dir, "golden_output.npy"), y)
        with open(os.path.join(cache_dir, "cache_info.json"), "w") as f:
            json.dump({"build_time_s": cache_time, "frames": frames}, f, indent=2)
        return cache_dir


class HLSIPCache:
    """Reuse synthesized HLS IP across flow runs. After step_hw_codegen, a node whose
    generated code (with its own directory name normalized away) matches an entry
    points at that entry's IP, so step_hw_ipgen skips it; after step_hw_ipgen the
    newly synthesized nodes are recorded. Variants with the same folding then pay
    for HLS synthesis once."""

    ATTRS = ["code_gen_dir_ipgen", "ipgen_path", "ip_path", "ip_vlnv"]

    def __init__(self):
        self.index_path = os.path.join(os.environ["FINN_BUILD_DIR"], "featureflow_ipcache.json")
        self.pending = {}
        self.hits = 0

    @staticmethod
    def _code_hash(code_gen_dir):
        h = hashlib.sha256()
        h.update(os.environ.get("XILINX_VIVADO", "").encode())
        for fname in sorted(os.listdir(code_gen_dir)):
            path = os.path.join(code_gen_dir, fname)
            if os.path.isfile(path) and fname != "vitis_hls.log":
                h.update(fname.encode())
                with open(path, "rb") as f:
                    h.update(f.read().replace(code_gen_dir.encode(), b"<DIR>"))
        return h.hexdigest()

    def _load(self):
        if not os.path.isfile(self.index_path):
            return {}
        with open(self.index_path) as f:
            return json.load(f)

    def lookup(self, model, cfg):
        index = self._load()
        for node in model.graph.node:
            if not is_hls_node(node):
                continue
            inst = getCustomOp(node)
            code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
            if not code_gen_dir or not os.path.isdir(code_gen_dir):
                continue
            key = self._code_hash(code_gen_dir)
            entry = index.get(key)
            if (
                entry is not None
                and all(entry.get(attr) for attr in self.ATTRS)
                and os.path.isdir(entry["ip_path"])
            ):
                for attr in self.ATTRS:
                    inst.set_nodeattr(attr, entry[attr])
                self.hits += 1
            else:
                self.pending[node.name] = key
        return model

    def store(self, model, cfg):
        with _file_lock(self.index_path + ".lock"):
            index = self._load()
            for node in model.graph.node:
                if node.name in self.pending:
                    inst = getCustomOp(node)
                    if os.path.isdir(inst.get_nodeattr("ip_path")):
                        index[self.pending[node.name]] = {
                            attr: inst.get_nodeattr(attr) for attr in self.ATTRS
                        }
            with open(self.index_path, "w") as f:
                json.dump(index, f, indent=1)
        return model

    def inject(self, cfg):
        def ip_cache_lookup(model, cfg):
            return self.lookup(model, cfg)

        def ip_cache_store(model, cfg):
            return self.store(model, cfg)

        after = {k: list(v) for k, v in cfg.inject_steps_after.items()}
        after.setdefault("step_hw_codegen", []).append(ip_cache_lookup)
        after.setdefault("step_hw_ipgen", []).append(ip_cache_store)
        cfg.inject_steps_after = after


def reference_cycles(cache_dir, folding_json):
    """Estimated cycles/frame of the checkpoint folded by the committed folding JSON."""
    model = ModelWrapper(os.path.join(cache_dir, "intermediate_models", CACHE_PHASE + ".onnx"))
    model = model.transform(ApplyConfig(folding_json))
    model = model.transform(AnnotateCycles())
    return int(model.analysis(dataflow_performance)["max_cycles"])


def matched_target_fps(cfg, ref_cycles):
    """Largest target_fps whose cycle budget is no tighter than ``ref_cycles``."""
    return int((10**9 / cfg.synth_clk_period_ns) // ref_cycles)


@contextlib.contextmanager
def time_steps(step_names):
    """Record the wall time of the named build steps, as the phase builder only
    reports per-phase times. Steps are wrapped where the phases look them up."""
    times = {}
    originals = {}

    def wrap(fn):
        @functools.wraps(fn)
        def timed(model, cfg):
            start = time.time()
            try:
                return fn(model, cfg)
            finally:
                times[fn.__name__] = times.get(fn.__name__, 0.0) + time.time() - start

        return timed

    for name in step_names:
        originals[name] = getattr(build_phases, name)
        setattr(build_phases, name, wrap(originals[name]))
    try:
        yield times
    finally:
        for name, fn in originals.items():
            setattr(build_phases, name, fn)


def verification_results(output_dir):
    """{verification step: True/False} from the verify_<step>_<frame>_<RESULT> files."""
    results = {}
    for path in glob.glob(os.path.join(output_dir, "verification_output", "verify_*.np[yz]")):
        stem = os.path.splitext(os.path.basename(path))[0]
        step, result = stem[len("verify_") :].rsplit("_", 2)[0], stem.rsplit("_", 1)[1]
        results[step] = results.get(step, True) and result == "SUCCESS"
    return results


def _load_report(output_dir, name):
    path = os.path.join(output_dir, "report", name)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def hw_model_stats(model):
    """FIFO storage and DWC usage of a dataflow (child) model."""
    fifo_bits = 0
    fifo_depths = []
    dwcs = []
    for node in model.graph.node:
        if node.op_type.startswith("StreamingFIFO"):
            inst = getCustomOp(node)
            depth = inst.get_nodeattr("depth")
            fifo_depths.append(depth)
            fifo_bits += depth * inst.get_instream_width()
        elif node.op_type.startswith("StreamingDataWidthConverter"):
            inst = getCustomOp(node)
            dwcs.append(
                {
                    "name": node.name,
                    "impl": node.op_type.rsplit("_", 1)[-1],
                    "inWidth": inst.get_nodeattr("inWidth"),
                    "outWidth": inst.get_nodeattr("outWidth"),
                    "in_shape": list(inst.get_normal_input_shape()),
                    "out_shape": list(inst.get_normal_output_shape()),
                    # padding/cropping DWCs carry different element counts
                    "pads_or_crops": bool(
                        np.prod(inst.get_normal_input_shape())
                        != np.prod(inst.get_normal_output_shape())
                    ),
                }
            )
    return {
        "num_fifos": len(fifo_depths),
        "fifo_total_depth": int(sum(fifo_depths)),
        "fifo_total_kbit": fifo_bits / 1024.0 if fifo_depths else None,
        "num_dwcs": len(dwcs),
        "num_pad_crop_dwcs": sum(d["pads_or_crops"] for d in dwcs),
        "dwcs": dwcs,
    }


def run_flow(flow, model_dir, model, board, **overrides):
    """Resume the cached (model, board) checkpoint with ``overrides`` through stitched
    IP + rtlsim, verifying against the golden pair. Returns the metrics row."""
    cache_dir = cached_checkpoint(model_dir, model, board)
    key = model_key(model_dir, model)
    output_dir = make_build_dir("featureflow_%s_%s_%s_" % (flow, key, board))
    shutil.copytree(
        os.path.join(cache_dir, "intermediate_models"),
        os.path.join(output_dir, "intermediate_models"),
    )

    quick = _truthy("FEATURE_FLOW_QUICK")
    verify = os.environ.get("FEATURE_FLOW_VERIFY", "folded_hls_cppsim,stitched_ip_rtlsim")
    verify_steps = [s.strip() for s in verify.split(",") if s.strip()]
    if quick:
        verify_steps = [s for s in verify_steps if s == "folded_hls_cppsim"]
    mod = load_bench_module(model_dir)
    settings = dict(
        start_step=RESUME_PHASE,
        generate_outputs=list(FLOW_OUTPUTS),
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        verify_steps=verify_steps,
        verify_input_npy=os.path.join(cache_dir, "golden_input.npy"),
        verify_expected_output_npy=os.path.join(cache_dir, "golden_output.npy"),
        folding_config_file=folding_config_file(model_dir, model, board),
    )
    if quick:
        settings.update(
            stop_step=RESUME_PHASE,
            generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        )
    settings.update(overrides)
    # make_benchmark_cfg drops None values (keeping the model's own setting), so
    # fields that must be unset (e.g. no folding JSON for auto-folding) are cleared after
    unset = [k for k, v in settings.items() if v is None]
    cfg = configure(
        mod, model, board, output_dir, **{k: v for k, v in settings.items() if v is not None}
    )
    for k in unset:
        setattr(cfg, k, None)
    ip_cache = None
    if os.environ.get("FEATURE_FLOW_IP_CACHE", "1") != "0":
        ip_cache = HLSIPCache()
        ip_cache.inject(cfg)

    start = time.time()
    with time_steps(TIMED_STEPS) as step_times:
        ret = build.build_dataflow_cfg(model_file(mod, model), cfg)
    total_time = time.time() - start

    row = {
        "flow": flow,
        "model": key,
        "board": board,
        "output_dir": output_dir,
        "build_ok": ret == 0,
        "quick": quick,
        "total_time_s": total_time,
        "step_time_s": step_times,
        "verification": verification_results(output_dir),
        "verify_steps": verify_steps,
        "hls_ip_cache_hits": ip_cache.hits if ip_cache else None,
        "settings": {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in overrides.items()
            if isinstance(v, (str, int, float, bool, type(None))) or hasattr(v, "value")
        },
    }
    est = _load_report(output_dir, "estimate_network_performance.json")
    row["est_max_cycles"] = est.get("max_cycles")
    row["est_throughput_fps"] = est.get("estimated_throughput_fps")
    rtlsim = _load_report(output_dir, "rtlsim_performance.json")
    row["rtlsim_throughput_fps"] = rtlsim.get("stable_throughput[images/s]")
    row["rtlsim_latency_cycles"] = rtlsim.get("latency_cycles")
    res = _load_report(output_dir, "estimate_layer_resources.json").get("total", {})
    row["est_resources"] = {k: res.get(k) for k in ("LUT", "BRAM_18K", "URAM", "DSP")}
    # FIFO/DWC stats from the latest checkpoint that has them (FIFO sizing may be skipped)
    for step in ["step_set_fifo_depths", "step_minimize_bit_width"]:
        ckpt = os.path.join(output_dir, "intermediate_models", step + ".onnx")
        if os.path.isfile(ckpt):
            row.update(hw_model_stats(ModelWrapper(ckpt)))
            break

    results_dir = os.path.join(os.environ["FINN_BUILD_DIR"], "feature_flow_results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, flow + ".jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def latest_row(flow, model, board):
    """The latest complete (not quick, built) result of ``flow`` for (model, board)."""
    path = os.path.join(os.environ["FINN_BUILD_DIR"], "feature_flow_results", flow + ".jsonl")
    if not os.path.isfile(path):
        return None
    found = None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if (row["model"], row["board"]) == (model, board):
                if row["build_ok"] and not row.get("quick"):
                    found = row
    return found


def record_target(row, target_cycles):
    """Add the folding target to a result row and its results file entry."""
    row["target_cycles"] = target_cycles
    row["target_met"] = row["est_max_cycles"] is not None and row["est_max_cycles"] <= target_cycles
    path = os.path.join(
        os.environ["FINN_BUILD_DIR"], "feature_flow_results", row["flow"] + ".jsonl"
    )
    with open(path) as f:
        lines = f.readlines()
    for i in range(len(lines) - 1, -1, -1):
        if json.loads(lines[i])["output_dir"] == row["output_dir"]:
            lines[i] = json.dumps(row) + "\n"
            break
    with open(path, "w") as f:
        f.writelines(lines)


def assert_flow_correct(row):
    """The build finished and every requested verification step passed."""
    assert row["build_ok"], "build failed, see %s/build_dataflow.log" % row["output_dir"]
    for step in row["verify_steps"]:
        assert step in row["verification"], "verification %s did not run (%s)" % (
            step,
            row["output_dir"],
        )
        assert row["verification"][step], "verification %s failed (%s)" % (
            step,
            row["output_dir"],
        )
    if not row["quick"]:
        assert row["rtlsim_throughput_fps"], "no rtlsim throughput (deadlock?) in %s" % (
            row["output_dir"]
        )

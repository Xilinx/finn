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

"""Shared helpers for the per-PR benchmark feature harnesses.

The three feature harnesses (folding optimizer, generalized DWC, analytical FIFO
sizing) each build every supported finn-examples model through the ESTIMATE-ONLY
flow (no Vivado synthesis), extract one metric vector, and compare it against a
checked-in reference JSON with per-metric tolerance.

Each feature lives on its own ``dev-<feature>`` branch; on a tree without a given
feature the corresponding harness self-skips (see the ``require_*`` guards in the
individual test modules). The reference JSONs are seeded on first run: a model
whose key is missing (or when ``FINN_BENCH_RECORD=1``) is recorded and skipped,
so an initial pass on a feature branch populates the expected numbers, which are
then committed and asserted against on subsequent runs.
"""

import importlib.util
import inspect
import json
import os

# tests/benchmark/_feature_bench.py -> repo root
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "..", ".."))
REFERENCE_DIR = os.path.join(BENCH_DIR, "reference")

# (model_dir, board, model) tuples for every finn-examples family the base branch
# builds. ``board`` is chosen so a committed folding config exists and no Vivado is
# needed (estimate-only). ``model`` is only used by the multi-model bnn-pynq family.
# resnet50 is intentionally excluded (deferred: needs core/streamline changes).
SUPPORTED_MODELS = [
    ("gtsrb", "Pynq-Z1", None),
    ("kws", "Pynq-Z1", None),
    ("cybersecurity-mlp", "Pynq-Z1", None),
    ("vgg10-radioml", "ZCU104", None),
    ("bnn-pynq", "Pynq-Z1", "cnv-w1a1"),
    ("mobilenet_v1", "ZCU104", None),
]


def model_id(entry):
    """Short pytest-id for a SUPPORTED_MODELS entry."""
    model_dir, board, model = entry
    return model_dir if model is None else f"{model_dir}-{model}"


def load_bench_module(model_dir):
    """Import the per-model ``test_build_<model>.py`` benchmark module by path.

    These files have hyphens in their names and live outside the import path, so
    they are loaded via importlib rather than a normal import.
    """
    import glob

    cands = glob.glob(os.path.join(BENCH_DIR, model_dir, "test_build_*.py"))
    assert len(cands) == 1, f"expected one test_build_*.py in {model_dir}, got {cands}"
    mod_name = "bench_" + model_dir.replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, cands[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model_file(mod, model):
    """Path to the ONNX model for a benchmark module (handles multi-model bnn)."""
    if hasattr(mod, "model_file"):
        return mod.model_file
    # multi-model families (bnn-pynq) expose get_model_file(model) instead
    assert model is not None, "multi-model benchmark module needs an explicit model"
    return mod.get_model_file(model)


def make_base_cfg(mod, board, model, out):
    """Build the model's own DataflowBuildConfig, then normalise it for a cached,
    estimate-only run (no verification, no pdb, intermediate models saved)."""
    import finn.builder.build_dataflow_config as build_cfg

    sig = inspect.signature(mod.configure_build)
    if "model" in sig.parameters:
        cfg = mod.configure_build(board, model, out)
    else:
        cfg = mod.configure_build(board, out)
    cfg.output_dir = out
    cfg.verify_steps = []
    cfg.enable_build_pdb_debug = False
    cfg.save_intermediate_models = True
    cfg.generate_outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
    return cfg


def estimate_step_list(cfg, with_fifo_sizing=False):
    """Truncate the model's build step list to the estimate-only boundary.

    Two flavours of step list occur across the benchmark models:

    * phase-based (gtsrb/kws/cybersecurity/bnn): ``phase_optimize_hardware``
      bundles folding, (analytic) FIFO sizing and estimate-report generation and
      already runs sizing synthesis-free for estimate-only builds -- so we simply
      keep everything up to and including that phase.
    * explicit (vgg10/mobilenet): a flat list where ``step_set_fifo_depths`` sits
      *after* ``step_hw_ipgen``. For metric extraction that does not need FIFOs we
      truncate at ``step_generate_estimate_reports``; for the FIFO harness we keep
      the folding/bit-width prefix and then run ``step_set_fifo_depths`` followed by
      ``step_generate_estimate_reports``, dropping codegen/ipgen so it stays
      synthesis-free (analytic sizing works on the graph, no built IP needed).
    """
    steps = cfg.steps or _default_steps()
    names = [s if isinstance(s, str) else getattr(s, "__name__", "") for s in steps]

    if "phase_optimize_hardware" in names:
        idx = names.index("phase_optimize_hardware")
        return list(steps[: idx + 1])

    if "step_generate_estimate_reports" not in names:
        raise RuntimeError(f"no estimate-report step to truncate at in {names}")
    est_idx = names.index("step_generate_estimate_reports")
    if not with_fifo_sizing:
        return list(steps[: est_idx + 1])
    return list(steps[:est_idx]) + ["step_set_fifo_depths", "step_generate_estimate_reports"]


def _default_steps():
    import finn.builder.build_dataflow_config as build_cfg

    return list(build_cfg.default_build_dataflow_steps)


def run_estimate_build(model_file, cfg):
    """Run a cached estimate-only build; return True on success (rc == 0).

    Skips the build if the estimate reports already exist under ``cfg.output_dir``
    so repeated pytest runs reuse the cached result.
    """
    import finn.builder.build_dataflow as build

    os.chdir(REPO_ROOT)
    report = os.path.join(cfg.output_dir, "report", "estimate_network_performance.json")
    if os.path.isfile(report):
        return True
    os.makedirs(cfg.output_dir, exist_ok=True)
    return build.build_dataflow_cfg(model_file, cfg) == 0


# ---- metric extraction ------------------------------------------------------


def total_estimate_lut(out):
    with open(os.path.join(out, "report", "estimate_layer_resources.json")) as f:
        return json.load(f)["total"]["LUT"]


def network_max_cycles(out):
    with open(os.path.join(out, "report", "estimate_network_performance.json")) as f:
        return json.load(f)["max_cycles"]


def latest_intermediate(out, step_name):
    p = os.path.join(out, "intermediate_models", step_name + ".onnx")
    return p if os.path.isfile(p) else None


def total_fifo_kb(model_path):
    """Total FIFO storage (KiB) of a sized model.

    Prefers inserted ``StreamingFIFO`` nodes (depth x stream width); falls back to
    the per-node ``outFIFODepths`` attributes if no FIFO nodes are present.
    """
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    model = ModelWrapper(model_path)
    fifo_nodes = [n for n in model.graph.node if n.op_type.startswith("StreamingFIFO")]
    bits = 0
    if fifo_nodes:
        for n in fifo_nodes:
            inst = getCustomOp(n)
            bits += inst.get_nodeattr("depth") * inst.get_instream_width()
    else:
        for n in model.graph.node:
            inst = getCustomOp(n)
            try:
                depths = inst.get_nodeattr("outFIFODepths")
            except (AttributeError, Exception):  # noqa: B014 - attr may be absent
                continue
            width = inst.get_outstream_width()
            for d in depths:
                bits += int(d) * width
    return bits / 8.0 / 1024.0


# ---- reference compare / record ---------------------------------------------


def compare_or_record(feature, key, metrics, tolerances):
    """Compare ``metrics`` against ``reference/<feature>_reference.json[key]``.

    Missing key (or ``FINN_BENCH_RECORD=1``) records the measured metrics and skips
    the test with a note to rerun. Otherwise asserts each metric is within its
    per-metric relative tolerance (default 5%). Integer-count metrics with tol 0
    must match exactly.
    """
    import pytest

    ref_path = os.path.join(REFERENCE_DIR, feature + "_reference.json")
    ref = {}
    if os.path.isfile(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)

    record = os.environ.get("FINN_BENCH_RECORD") == "1" or key not in ref
    if record:
        ref[key] = metrics
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2, sort_keys=True)
        pytest.skip(f"recorded {feature}[{key}]={metrics}; commit reference and rerun to assert")

    expected = ref[key]
    errors = []
    for name, val in metrics.items():
        exp = expected.get(name)
        if exp is None:
            continue
        tol = tolerances.get(name, 0.05)
        if tol == 0 or exp == 0:
            ok = val == exp
        else:
            ok = abs(val - exp) <= abs(exp) * tol
        if not ok:
            errors.append(f"{name}: measured {val} vs expected {exp} (tol {tol:.0%})")
    assert not errors, f"{feature}[{key}] regressed -> " + "; ".join(errors)

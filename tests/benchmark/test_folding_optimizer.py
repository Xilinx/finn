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

"""Resource-aware folding-optimizer benchmark harness.

For every supported finn-examples model this runs two estimate-only builds:
a ``json`` baseline (the committed reference folding config) and the ``optimizer``
targeting the json baseline's achieved throughput. It records the achieved cycles
of both plus the optimizer's total estimated LUTs, and compares them against a
checked-in reference with tolerance.

NOTE: the optimizer uses simulated annealing and its result is not fully
deterministic across runs (SA convergence is non-monotonic). Tolerances here are
therefore wide; if you need a tight regression pin, set a fixed optimizer seed
before relying on the optimizer-specific numbers.

Runs green only on a tree that has the folding optimizer
(``feature/set-folding-optimizer``); otherwise self-skips.

Seed the reference on first run (per model, on the feature branch):
    FINN_BENCH_RECORD=1 pytest tests/benchmark/test_folding_optimizer.py
then commit tests/benchmark/reference/folding_reference.json and rerun.
"""

import pytest

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _feature_bench as fb  # noqa: E402

FEATURE = "folding"
FOLDING_EFFORT = 100
# SA is stochastic -> keep tolerances generous (see module docstring)
TOLERANCES = {"json_cycles": 0.02, "opt_cycles": 0.15, "opt_lut": 0.25}


def require_folding_optimizer():
    """Skip unless the folding optimizer (folding_style) is present on this tree."""
    try:
        import finn.builder.build_dataflow_config as build_cfg

        fields = {f.name for f in dataclasses.fields(build_cfg.DataflowBuildConfig)}
        assert "folding_style" in fields
    except (ImportError, AttributeError, AssertionError):
        pytest.skip("folding optimizer (folding_style) not available on this tree")


def _run(mod, board, model, out, method, target_fps):
    cfg = fb.make_base_cfg(mod, board, model, out)
    if method == "json":
        cfg.target_fps = None  # rely on the committed folding_config_file
    else:
        cfg.folding_config_file = None
        cfg.folding_style = "optimizer"
        cfg.folding_effort = FOLDING_EFFORT
        cfg.target_fps = target_fps
    cfg.steps = fb.estimate_step_list(cfg, with_fifo_sizing=False)
    assert fb.run_estimate_build(fb.get_model_file(mod, model), cfg), f"{method} build failed"
    return out


@pytest.mark.slow
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", fb.SUPPORTED_MODELS, ids=fb.model_id)
def test_folding_optimizer(entry):
    require_folding_optimizer()

    model_dir, board, model = entry
    key = fb.model_id(entry)

    mod = fb.load_bench_module(model_dir)
    build_dir = os.environ.get("FINN_BUILD_DIR", "/tmp")
    base = os.path.join(build_dir, f"bench_{FEATURE}_{key}_{board}")

    # 1) json baseline sets the throughput target for the optimizer
    json_out = _run(mod, board, model, base + "_json", "json", None)
    json_cycles = fb.network_max_cycles(json_out)
    assert json_cycles and json_cycles > 0, "json baseline produced no cycle estimate"

    # 2) optimizer targeting the json baseline's throughput
    cfg_tmp = fb.make_base_cfg(mod, board, model, base + "_opt")
    clock_hz = 1e9 / cfg_tmp.synth_clk_period_ns
    target_fps = int(clock_hz / json_cycles)
    opt_out = _run(mod, board, model, base + "_opt", "optimizer", target_fps)

    metrics = {
        "json_cycles": json_cycles,
        "opt_cycles": fb.network_max_cycles(opt_out),
        "opt_lut": fb.total_estimate_lut(opt_out),
    }
    fb.compare_or_record(FEATURE, key, metrics, TOLERANCES)

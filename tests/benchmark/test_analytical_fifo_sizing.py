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

"""Analytical (tree-model) FIFO-sizing benchmark harness.

For every supported finn-examples model this builds the estimate-only flow with
the analytic tree-model FIFO sizer engaged (no rtlsim, no synthesis) and checks
the resulting total FIFO storage (KiB) against a checked-in reference with
tolerance. The comparison target is the analytic sizer's own regression baseline;
the paper / rtlsim ground truths are tracked separately in the claude-tools notes.

Runs green only on a tree that has the analytic FIFO-sizing feature
(``feature/analytical-fifo-sizing``); otherwise self-skips.

Seed the reference on first run (per model, on the feature branch):
    FINN_BENCH_RECORD=1 pytest tests/benchmark/test_analytical_fifo_sizing.py
then commit tests/benchmark/reference/fifo_sizing_reference.json and rerun.
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _feature_bench as fb  # noqa: E402

FEATURE = "fifo_sizing"
# mobilenet_v1 is a known analytic over-size gap (isolated-characterization
# ignores backpressure) -- tracked, not yet matching ground truth.
XFAIL_MODELS = {"mobilenet_v1"}
TOLERANCES = {"fifo_kb": 0.10}


def require_analytic_fifo():
    """Skip unless the analytic tree-model FIFO sizer is present on this tree."""
    try:
        import finn.builder.build_dataflow_config as build_cfg

        assert hasattr(build_cfg.AutoFIFOSizingMethod, "ANALYTIC")
        assert hasattr(build_cfg, "TAVGenerationMethod")
        assert hasattr(build_cfg, "TAVUtilizationMethod")
    except (ImportError, AttributeError, AssertionError):
        pytest.skip("analytic tree-model FIFO sizing not available on this tree")


def _configure_analytic_fifo(cfg, build_cfg):
    cfg.auto_fifo_depths = True
    cfg.auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.ANALYTIC
    cfg.tav_generation_strategy = build_cfg.TAVGenerationMethod.TREE_MODEL
    cfg.tav_utilization_strategy = build_cfg.TAVUtilizationMethod.CONSERVATIVE_RELAXATION
    # fixed reference folding (never auto-fold during a sizing run)
    cfg.target_fps = None


@pytest.mark.slow
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", fb.SUPPORTED_MODELS, ids=fb.model_id)
def test_analytical_fifo_sizing(entry):
    require_analytic_fifo()
    import finn.builder.build_dataflow_config as build_cfg

    model_dir, board, model = entry
    key = fb.model_id(entry)
    if model_dir in XFAIL_MODELS:
        pytest.xfail(f"{model_dir}: known analytic FIFO over-size gap")

    mod = fb.load_bench_module(model_dir)
    model_file = fb.get_model_file(mod, model)
    build_dir = os.environ.get("FINN_BUILD_DIR", "/tmp")
    out = os.path.join(build_dir, f"bench_{FEATURE}_{key}_{board}")

    cfg = fb.make_base_cfg(mod, board, model, out)
    _configure_analytic_fifo(cfg, build_cfg)
    cfg.steps = fb.estimate_step_list(cfg, with_fifo_sizing=True)

    assert fb.run_estimate_build(model_file, cfg), f"estimate build failed for {key}"

    sized = fb.latest_intermediate(out, "step_set_fifo_depths")
    assert sized is not None, "step_set_fifo_depths did not produce a sized model"
    metrics = {"fifo_kb": round(fb.total_fifo_kb(sized), 3)}
    fb.compare_or_record(FEATURE, key, metrics, TOLERANCES)

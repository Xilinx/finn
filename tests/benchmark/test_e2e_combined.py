# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end COMBINED flow: all three features together -- SetFolding
optimizer with padded folding configs + analytic tree-model FIFO sizing (with
the folding FIFO-cost heuristic enabled, as recommended when rtlsim is never
invoked) + generalized DWCs, measured up to stitched-IP rtlsim.

This is the headline row of the comparison table: the fully-analytical flow at
the committed folding JSON's throughput target versus the classic
JSON + rtlsim-sizing + stock-DWC baseline. Requires all three features on the
tree; self-skips otherwise.

Run (e2e env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_combined.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402
import _feature_bench as fb  # noqa: E402

FLOW = "combined"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_combined(entry):
    eb.require(eb.has_folding_optimizer(), "SetFolding optimizer (folding_style)")
    eb.require(eb.has_generalized_dwc(), "generalized DWC")
    eb.require(eb.has_analytic_fifo(), "analytic tree-model FIFO sizing")

    model_dir, board, model = entry
    key = eb.model_id(entry)
    mod = fb.load_bench_module(model_dir)
    target_fps, json_cycles = eb.matched_target_fps(mod, board, model, key)

    # the optimizer's in-loop FIFO scoring re-sizes every candidate fold; even
    # synthesis-free that multiplies sizing cost by folding_effort, so it is
    # opt-in for the benchmark matrix
    fifo_heuristic = os.environ.get("FINN_E2E_FOLDING_FIFO_HEURISTIC") == "1"

    def mutate(cfg):
        eb.flow_use_folding_optimizer(
            cfg, target_fps, with_padding=True, fifo_heuristic=fifo_heuristic
        )
        eb.flow_use_analytic_fifo_sizing(cfg)

    metrics = eb.run_flow(
        entry, FLOW, mutate, extra={"target_fps": target_fps, "json_cycles": json_cycles}
    )
    eb.assert_no_throughput_regression(FLOW, key, metrics)

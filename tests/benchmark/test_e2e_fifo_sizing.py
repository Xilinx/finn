# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end ANALYTIC FIFO SIZING flow: committed folding JSON + tree-model
analytic FIFO sizer, measured up to stitched-IP rtlsim.

Differs from the baseline flow in exactly one knob (auto_fifo_strategy:
largefifo_rtlsim -> analytical), so the table isolates the sizer's effect on
FIFO storage, sizing runtime and throughput. Requires
feature/analytical-fifo-sizing on the tree; self-skips otherwise.

Run (e2e env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_fifo_sizing.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402

FLOW = "fifo"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_fifo_sizing(entry):
    eb.require(eb.has_analytic_fifo(), "analytic tree-model FIFO sizing")

    def mutate(cfg):
        eb.flow_use_json_folding(cfg)
        eb.flow_use_analytic_fifo_sizing(cfg)

    metrics = eb.run_flow(entry, FLOW, mutate)
    eb.assert_no_throughput_regression(FLOW, eb.model_id(entry), metrics)

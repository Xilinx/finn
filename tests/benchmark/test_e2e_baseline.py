# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end BASELINE flow: committed folding JSON + rtlsim-based FIFO sizing
(largefifo_rtlsim) + stock HLS/RTL DWCs, measured up to stitched-IP rtlsim.

This is the reference row of the e2e comparison table: every feature flow
(fifo/folding/dwc/combined) is judged against these numbers. It must therefore
run on a tree WITHOUT the generalized DWC (the baseline env); on a feature env
it self-skips so it can never accidentally rebrand a feature build as baseline.

Run (baseline env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_baseline.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402

FLOW = "baseline"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_baseline(entry):
    eb.forbid_generalized_dwc()

    def mutate(cfg):
        eb.flow_use_json_folding(cfg)
        eb.flow_use_rtlsim_fifo_sizing(cfg)

    eb.run_flow(entry, FLOW, mutate)

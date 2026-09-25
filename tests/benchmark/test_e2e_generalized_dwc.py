# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end GENERALIZED DWC flow: the exact baseline build configuration
(committed folding JSON + rtlsim FIFO sizing) run on a tree that carries the
generalized DWC, measured up to stitched-IP rtlsim.

The build config is byte-identical to test_e2e_baseline's; the only difference
is the DWC implementation the tree provides, so baseline-vs-dwc rows isolate
the DWC swap (LUT estimates, throughput). Requires
feature/generalized-datawidthconverter; self-skips otherwise.

Run (e2e env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_generalized_dwc.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402

FLOW = "dwc"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_generalized_dwc(entry):
    eb.require(eb.has_generalized_dwc(), "generalized DWC")

    def mutate(cfg):
        eb.flow_use_json_folding(cfg)
        eb.flow_use_rtlsim_fifo_sizing(cfg)

    metrics = eb.run_flow(entry, FLOW, mutate)
    eb.assert_no_throughput_regression(FLOW, eb.model_id(entry), metrics)

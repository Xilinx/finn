# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end FOLDING OPTIMIZER flow: SetFolding optimizer with padded folding
configs (folding_maximum_padding=6, which relies on the generalized DWC to
realize non-divisor stream widths) at the committed folding JSON's throughput
target, plus the baseline rtlsim FIFO sizing, measured up to stitched-IP rtlsim.

FIFO sizing stays on the baseline method so the table isolates the folding
change; the 'combined' flow is where optimizer + analytic sizing + generalized
DWC all engage together. Requires feature/set-folding-optimizer AND
feature/generalized-datawidthconverter (for padding); self-skips otherwise.

Run (e2e env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_folding.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402
import _feature_bench as fb  # noqa: E402

FLOW = "folding"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_folding(entry):
    eb.require(eb.has_folding_optimizer(), "SetFolding optimizer (folding_style)")
    eb.require(eb.has_generalized_dwc(), "generalized DWC (needed for padded folding)")

    model_dir, board, model = entry
    key = eb.model_id(entry)
    mod = fb.load_bench_module(model_dir)
    target_fps, json_cycles = eb.matched_target_fps(mod, board, model, key)

    def mutate(cfg):
        eb.flow_use_folding_optimizer(cfg, target_fps, with_padding=True)
        eb.flow_use_rtlsim_fifo_sizing(cfg)

    metrics = eb.run_flow(
        entry, FLOW, mutate, extra={"target_fps": target_fps, "json_cycles": json_cycles}
    )
    eb.assert_no_throughput_regression(FLOW, key, metrics)

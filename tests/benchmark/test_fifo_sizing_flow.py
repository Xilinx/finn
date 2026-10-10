# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FIFO sizing flow: the committed folding JSON, FIFOs sized by each strategy.

Checks that the sized accelerator still computes the golden output in stitched-IP
rtlsim and records the time of step_set_fifo_depths and the resulting FIFO storage.
largefifo_rtlsim (dev's default: simulate with oversized FIFOs, trim to the observed
occupancy) is the reference: every other strategy must reach its rtlsim throughput
within THROUGHPUT_TOL, as an undersized FIFO stalls the pipeline.
"""

import pytest

from feature_flow_helpers import (
    assert_flow_correct,
    feature_cases,
    latest_row,
    run_flow,
)

import finn.builder.build_dataflow_config as build_cfg

REFERENCE = "largefifo_rtlsim"
FIFO_STRATEGIES = [REFERENCE, "heuristic_analytical", "heuristic_rtlsim"]
# largest accepted rtlsim throughput loss against the reference sizing
THROUGHPUT_TOL = 0.02


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("strategy", FIFO_STRATEGIES)
@pytest.mark.parametrize("model_dir,model,board", feature_cases())
def test_fifo_sizing_flow(model_dir, model, board, strategy):
    known = [m.value for m in build_cfg.AutoFIFOSizingMethod]
    if strategy not in known:
        pytest.skip("FIFO sizing strategy %s not available on this tree" % strategy)
    row = run_flow(
        "fifo_" + strategy,
        model_dir,
        model,
        board,
        auto_fifo_depths=True,
        auto_fifo_strategy=strategy,
    )
    assert_flow_correct(row)
    if strategy == REFERENCE or row["quick"]:
        return
    ref = latest_row("fifo_" + REFERENCE, row["model"], board)
    if ref is None or not ref["rtlsim_throughput_fps"]:
        pytest.skip("correct, but no %s result to compare throughput against" % REFERENCE)
    ratio = row["rtlsim_throughput_fps"] / ref["rtlsim_throughput_fps"]
    assert ratio >= 1 - THROUGHPUT_TOL, "%s loses %.1f%% throughput against %s (%s)" % (
        strategy,
        100 * (1 - ratio),
        REFERENCE,
        row["output_dir"],
    )

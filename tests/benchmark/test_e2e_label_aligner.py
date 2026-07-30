# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end LABEL ALIGNER flow: the baseline build with align_labels=True,
which inserts an AlignLabels node giving the accelerator a second output stream
carrying each input aligned with its prediction, measured up to stitched-IP
rtlsim.

Uses the analytic FIFO sizer when the tree has it: the AlignLabels bypass
buffer must be sized to the model latency for full throughput (the aligner env
carries feature/analytical-fifo-sizing for exactly this), and falls back to
rtlsim sizing otherwise. The test asserts the aligner does not degrade rtlsim
throughput versus the recorded baseline. Requires feature/label_aligner;
self-skips otherwise.

Run (aligner env):
    PYTHONPATH=$PWD/src pytest -x tests/benchmark/test_e2e_label_aligner.py
"""

import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _e2e_bench as eb  # noqa: E402
import _feature_bench as fb  # noqa: E402

FLOW = "aligner"

# mobilenet ends in non-dataflow TopK/Flatten before its output; InsertAlignLabels
# attaches at the graph output, sandwiching them inside the dataflow block
# (contiguity violation). Supporting this topology needs an output-side
# skip-non-dataflow rule in InsertAlignLabels (and a label stream carrying the
# pre-TopK scores, whose fully-folded width is impractical at 1000 classes).
XFAIL_MODELS = {"mobilenet_v1"}


def _has_alignlabels_node(model_path):
    from qonnx.core.modelwrapper import ModelWrapper

    model = ModelWrapper(model_path)
    return any(n.op_type.startswith("AlignLabels") for n in model.graph.node)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", eb.E2E_MODELS, ids=eb.model_id)
def test_e2e_label_aligner(entry):
    eb.require(eb.has_label_aligner(), "label aligner (align_labels)")

    key = eb.model_id(entry)
    if entry[0] in XFAIL_MODELS:
        pytest.xfail(f"{entry[0]}: trailing non-dataflow TopK/Flatten breaks AlignLabels insertion")

    def mutate(cfg):
        eb.flow_use_json_folding(cfg)
        if eb.has_analytic_fifo():
            eb.flow_use_analytic_fifo_sizing(cfg)
        else:
            eb.flow_use_rtlsim_fifo_sizing(cfg)
        cfg.align_labels = True

    metrics = eb.run_flow(entry, FLOW, mutate)

    out = os.path.join(eb.build_root(), f"e2e_{FLOW}_{key}")
    sized = fb.latest_intermediate(out, "step_set_fifo_depths")
    assert sized and _has_alignlabels_node(
        sized
    ), f"{FLOW}[{key}]: no AlignLabels node in the built model"
    eb.assert_no_throughput_regression(FLOW, key, metrics)

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

"""Generalized StreamingDataWidthConverter benchmark harness.

For every supported finn-examples model this builds the estimate-only flow (folded
+ specialized, no synthesis), then applies ``InsertDWC`` and inspects the DWC nodes
the generalized transform produces: how many are inserted, how many route to the
optimal RTL variant vs the generalized HLS variant, and their aggregate estimated
LUTs. Those counts are the direct signature of the PR's ``InsertDWC`` routing
logic, so they make a stable regression signal even at estimate level (where the
per-DWC LUT estimate is coarse -- hence its loose tolerance).

Runs green only on a tree that has the generalized DWC feature
(``feature/generalized-datawidthconverter``); otherwise self-skips.

Seed the reference on first run (per model, on the feature branch):
    FINN_BENCH_RECORD=1 pytest tests/benchmark/test_generalized_dwc.py
then commit tests/benchmark/reference/dwc_reference.json and rerun.
"""

import pytest

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _feature_bench as fb  # noqa: E402

FEATURE = "dwc"
# exact-match the routing decision; the LUT estimate is coarse at estimate level
TOLERANCES = {"num_dwc": 0, "num_rtl": 0, "num_hls": 0, "dwc_lut": 0.15}


def require_generalized_dwc():
    """Skip unless the generalized (padding/cropping) DWC op is present."""
    try:
        from finn.custom_op.fpgadataflow import streamingdatawidthconverter as dwc_mod

        # the generalized abstract op carries in_shape/out_shape (vs a single shape)
        if "in_shape" not in inspect.getsource(dwc_mod):
            pytest.skip("generalized DWC (in_shape/out_shape) not available on this tree")
    except ImportError:
        pytest.skip("streamingdatawidthconverter op not importable on this tree")


def _dwc_metrics(model_path):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC

    model = ModelWrapper(model_path)
    model = model.transform(InsertDWC())
    dwcs = [n for n in model.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    num_rtl = num_hls = 0
    dwc_lut = 0
    for n in dwcs:
        inst = getCustomOp(n)
        style = inst.get_nodeattr("preferred_impl_style")
        if style == "rtl":
            num_rtl += 1
        else:
            num_hls += 1
        try:
            dwc_lut += int(inst.lut_estimation())
        except Exception:
            pass
    return {"num_dwc": len(dwcs), "num_rtl": num_rtl, "num_hls": num_hls, "dwc_lut": dwc_lut}


@pytest.mark.slow
@pytest.mark.finn_examples
@pytest.mark.parametrize("entry", fb.SUPPORTED_MODELS, ids=fb.model_id)
def test_generalized_dwc(entry):
    require_generalized_dwc()

    model_dir, board, model = entry
    key = fb.model_id(entry)

    mod = fb.load_bench_module(model_dir)
    model_file = fb.get_model_file(mod, model)
    build_dir = os.environ.get("FINN_BUILD_DIR", "/tmp")
    out = os.path.join(build_dir, f"bench_{FEATURE}_{key}_{board}")

    cfg = fb.make_base_cfg(mod, board, model, out)
    cfg.steps = fb.estimate_step_list(cfg, with_fifo_sizing=False)
    assert fb.run_estimate_build(model_file, cfg), f"estimate build failed for {key}"

    folded = fb.latest_intermediate(out, "step_generate_estimate_reports")
    assert folded is not None, "no folded/estimate model to insert DWCs into"
    metrics = _dwc_metrics(folded)
    fb.compare_or_record(FEATURE, key, metrics, TOLERANCES)

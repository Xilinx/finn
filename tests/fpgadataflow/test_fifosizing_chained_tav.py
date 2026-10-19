# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

"""Depth regression of the chained-TAV FIFO sizer on small graphs whose token
access vectors all come from tree models, so no synthesis or rtlsim is needed.
The expected depths are the sizer's own output and pin its behaviour; see
docs/finn/fifo_sizing_heuristics.rst for what each rule does."""

import pytest

import numpy as np
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveFIFOSizes,
    DeriveTokenAccessVectors,
    burst_above_rate,
    causal_writes,
    local_clock,
    peak_occupancy,
    periodic_peak,
    refill_floor,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

PART = "xc7z020clg400-1"
IDT = DataType["INT4"]


def mvau_node(name_in, name_out, weights, mw, mh, simd, pe, nvec):
    return helper.make_node(
        "MVAU",
        [name_in, weights],
        [name_out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=IDT.name,
        weightDataType=IDT.name,
        outputDataType=IDT.name,
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=[1, nvec],
        mem_mode="internal_decoupled",
        resType="lut",
    )


def finish(model, tensors, weights):
    for name, (mw, mh) in weights.items():
        model.set_initializer(name, gen_finn_dt_tensor(IDT, (mw, mh)))
        model.set_tensor_datatype(name, IDT)
    for t in tensors:
        model.set_tensor_datatype(t, IDT)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def make_mlp(layers, nvec):
    """A chain of MVAUs, ``layers`` = [(mw, mh, simd, pe)]; width converters are
    inserted where the foldings differ."""
    nodes, vi = [], []
    cur = "inp"
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, nvec, layers[0][0]])
    for i, (mw, mh, simd, pe) in enumerate(layers):
        out = "outp" if i == len(layers) - 1 else "t%d" % i
        nodes.append(mvau_node(cur, out, "w%d" % i, mw, mh, simd, pe, nvec))
        if out != "outp":
            vi.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, nvec, mh]))
        cur = out
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nvec, layers[-1][1]])
    model = ModelWrapper(
        qonnx_make_model(helper.make_graph(nodes, "mlp", [inp], [outp], value_info=vi))
    )
    weights = {"w%d" % i: (mw, mh) for i, (mw, mh, _, _) in enumerate(layers)}
    return finish(model, ["inp", "outp"] + [v.name for v in vi], weights)


def make_residual(nvec, ch, simd, pe):
    """Fork -> (shortcut | MVAU) -> join."""
    shape = [1, nvec, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    vi = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("short", "l0", "l1")]
    nodes = [
        helper.make_node(
            "DuplicateStreams",
            ["inp"],
            ["short", "l0"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=ch,
            NumOutputStreams=2,
            PE=simd,
            inputDataType=IDT.name,
            numInputVectors=[1, nvec],
        ),
        mvau_node("l0", "l1", "w", ch, ch, simd, pe, nvec),
        helper.make_node(
            "ElementwiseAdd",
            ["short", "l1"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=shape,
            out_shape=shape,
            lhs_dtype=IDT.name,
            rhs_dtype=IDT.name,
            out_dtype=IDT.name,
            lhs_style="input",
            rhs_style="input",
            PE=pe,
        ),
    ]
    model = ModelWrapper(
        qonnx_make_model(helper.make_graph(nodes, "residual", [inp], [outp], value_info=vi))
    )
    return finish(model, ["inp", "outp", "short", "l0", "l1"], {"w": (ch, ch)})


def size_fifos(model):
    """The sizing part of step_set_fifo_depths with tree-model TAVs. Returns
    ``{"producer -> consumer": depth}``."""
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateCycles())
    period = int(model.analysis(dataflow_performance)["max_cycles"])
    model = model.transform(DeriveTokenAccessVectors(model, period, "tree_model", PART, 10.0))
    period = int(
        model.analysis(partial(dataflow_performance, use_characterized=True))["max_cycles"]
    )
    model = model.transform(DeriveFIFOSizes(period=period))
    depths = {}
    for node in model.graph.node:
        out_depths = getCustomOp(node).get_nodeattr("outFIFODepths")
        for i, t in enumerate(node.output):
            cons = model.find_consumer(t)
            if cons is None:
                continue
            in_depths = getCustomOp(cons).get_nodeattr("inFIFODepths")
            depth = max(out_depths[i], in_depths[list(cons.input).index(t)])
            depths["%s -> %s" % (node.name, cons.name)] = depth
    return depths


@pytest.mark.fpgadataflow
def test_chained_tav_mlp_chain():
    # fast layer -> slow pacer (bursty reads) -> fast layer
    layers = [(64, 64, 8, 8), (64, 64, 2, 64), (64, 64, 64, 64)]
    depths = size_fifos(make_mlp(layers, nvec=16))
    assert depths == {
        "MVAU_hls_0 -> StreamingDataWidthConverter_rtl_0": 2,
        # the pacer reads 32 tokens in a burst at the start of each window:
        # the burst floor keeps 4 ahead of its average rate line
        "StreamingDataWidthConverter_rtl_0 -> MVAU_hls_1": 4,
        "MVAU_hls_1 -> MVAU_hls_2": 2,
    }


@pytest.mark.fpgadataflow
def test_chained_tav_residual():
    depths = size_fifos(make_residual(nvec=16, ch=64, simd=8, pe=64))
    # the shortcut's width converter and the MVAU are inside the branch: fill
    # frame only, never relaxed; the join waits for the MVAU's latency
    assert depths == {
        "DuplicateStreams_hls_0 -> StreamingDataWidthConverter_rtl_0": 2,
        "DuplicateStreams_hls_0 -> MVAU_hls_0": 2,
        "StreamingDataWidthConverter_rtl_0 -> ElementwiseAdd_hls_0": 2,
        "MVAU_hls_0 -> ElementwiseAdd_hls_0": 2,
    }


@pytest.mark.fpgadataflow
def test_periodic_peak_and_floors():
    # one write per cycle, one read per cycle 10 cycles later: 10 in flight
    w = np.arange(100)
    assert periodic_peak(w, w + 10, 1000) == 10
    # repeated every 100 cycles the frames overlap and the backlog stays 10
    assert periodic_peak(w, w + 10, 100) == 10
    # fill frame vs steady frame: the fill frame carries the latency, the
    # steady frame the run-ahead; a chain edge takes the larger
    writes = np.concatenate([w, w + 100])
    reads = np.concatenate([w + 10, w + 100 + 2])
    assert peak_occupancy(writes, reads, 1000, 100, fill_only=True) == 10
    assert peak_occupancy(writes, reads, 1000, 100, fill_only=False) == 10
    # demand of 8 tokens in 8 cycles against a rate of 0.25/cycle: 6 above the line
    assert burst_above_rate(np.arange(8), 0.25) == 6
    # a 4:1 compactor cannot emit output j before input 4(j+1) has arrived
    in_times = np.arange(0, 16) * 10
    held = causal_writes(np.zeros(4, dtype=np.int64), {"x": (in_times, None)})
    assert list(held[:2]) == [in_times[3], in_times[7]]
    # the max-plus clock: a node stalls as long as its latest input is late
    curve = np.array([1, 2, 3, 4])
    clock = local_clock(4, {"x": (np.array([0, 0, 10, 10]), curve)})
    assert list(clock) == [0, 1, 10, 11]
    # writes pause from cycle 29 to 130 while the consumer reads one token per
    # 4 cycles: 25 reads in the gap, 15 beyond a consumer slack of 40 cycles
    w_gap = np.concatenate([np.arange(30), np.arange(10) + 130])
    r_gap = np.arange(40) * 4 + 2
    assert refill_floor(w_gap, r_gap, 0) == 25
    assert refill_floor(w_gap, r_gap, 40) == 15

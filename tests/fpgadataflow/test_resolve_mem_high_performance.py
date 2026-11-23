# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import os
import re
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.resolve_mem_high_performance import (
    ResolveMemHighPerformance,
)
from finn.util.basic import make_build_dir, robust_rmtree


def make_mvau_fifo_model(mem_high_performance="auto", pumped_memory=0):
    """A decoupled MVAU_hls followed by a StreamingFIFO_rtl, which has no memstream."""
    mw, mh = 8, 4
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, mh])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    mvau = helper.make_node(
        "MVAU_hls",
        ["inp", "weights"],
        ["mid"],
        name="MVAU_hls_0",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=2,
        PE=2,
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT32",
        noActivation=1,
        mem_mode="internal_decoupled",
        pumpedMemory=pumped_memory,
        mem_high_performance=mem_high_performance,
    )
    fifo = helper.make_node(
        "StreamingFIFO_rtl",
        ["mid"],
        ["outp"],
        name="StreamingFIFO_rtl_0",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=2,
        folded_shape=[1, mh],
        normal_shape=[1, mh],
        dataType="INT32",
    )
    graph = helper.make_graph([mvau, fifo], "mvau_fifo", [inp], [outp], value_info=[mid])
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("weights", DataType["INT4"])
    model.set_initializer("weights", np.ones((mw, mh), dtype=np.float32))
    return model


# target clock period in ns, memstream clock doubled by pumped memory
@pytest.mark.parametrize(
    "clk_ns,pumped,expected",
    [(10.0, 0, 0), (3.0, 0, 0), (2.5, 0, 1), (1.6, 0, 1), (5.0, 1, 1), (6.0, 1, 0)],
)
@pytest.mark.fpgadataflow
def test_mem_high_performance_follows_clock(clk_ns, pumped, expected):
    inst = getCustomOp(make_mvau_fifo_model().graph.node[0])
    assert inst.get_mem_high_performance(clk_ns, pumped) == expected
    model = make_mvau_fifo_model(pumped_memory=pumped)
    model = model.transform(ResolveMemHighPerformance(clk_ns))
    mvau, fifo = (getCustomOp(n) for n in model.graph.node)
    assert mvau.get_nodeattr("mem_high_performance") == ("on" if expected else "off")
    # layers without memstream support are left alone
    assert fifo.get_nodeattr("mem_high_performance") == "auto"


# build config override (None = by clock)
@pytest.mark.parametrize("force", [None, False, True])
# node attribute
@pytest.mark.parametrize("attr", ["auto", "on", "off"])
@pytest.mark.fpgadataflow
def test_mem_high_performance_override(force, attr):
    model = make_mvau_fifo_model(mem_high_performance=attr)
    model = model.transform(ResolveMemHighPerformance(10.0, force))
    resolved = getCustomOp(model.graph.node[0]).get_nodeattr("mem_high_performance")
    if attr != "auto":
        # explicit node attributes take precedence
        assert resolved == attr
    elif force is None:
        assert resolved == "off"
    else:
        assert resolved == ("on" if force else "off")


# node attribute
@pytest.mark.parametrize("attr", ["auto", "on", "off"])
# target clock period in ns
@pytest.mark.parametrize("clk_ns", [5.0, 2.0])
@pytest.mark.fpgadataflow
def test_mem_high_performance_wrapper_parameter(attr, clk_ns):
    inst = getCustomOp(make_mvau_fifo_model(mem_high_performance=attr).graph.node[0])
    code_gen_dir = make_build_dir("test_mem_high_performance_")
    try:
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        inst.generate_hdl_memstream("xczu7ev-ffvc1156-2-e", clk_ns)
        with open(os.path.join(code_gen_dir, "MVAU_hls_0_memstream_wrapper.v")) as f:
            wrapper = f.read()
        m = re.search(r"parameter\s+HIGH_PERFORMANCE\s*=\s*(\d+)", wrapper)
        assert m is not None, "HIGH_PERFORMANCE missing from memstream wrapper"
        expected = {"on": 1, "off": 0, "auto": int(clk_ns <= 2.5)}[attr]
        assert int(m.group(1)) == expected
        assert ".HIGH_PERFORMANCE(HIGH_PERFORMANCE)" in wrapper
    finally:
        robust_rmtree(code_gen_dir)

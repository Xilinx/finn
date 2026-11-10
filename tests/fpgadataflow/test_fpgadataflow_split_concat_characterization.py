# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree model vs node rtlsim token access vectors of StreamingSplit_hls and
StreamingConcat_hls."""

import pytest

from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.util.test import tree_model_test_up_to_rotation

# ChannelsPerStream, SIMD, numInputVectors
CONFIGS = [
    ([8, 8, 8, 8], 1, [1]),
    ([8, 8, 8, 8], 2, [1]),
    ([8, 8, 8, 8], 4, [1]),
    ([4, 8, 12], 2, [1, 3]),
    ([16, 4], 1, [1, 2, 2]),
    ([6, 2, 10], 2, [1, 4]),
    ([2, 6], 2, [1, 5]),
]

# The rate, the token counts and the read-to-write relationship are exact. The
# model does not claim where rtlsim's window was cut, hence the rotation.
TOLERANCES = dict(
    max_allowed_volume_frac=0.005,
    max_allowed_length_frac=0.005,
    volume_const=2,
    length_const=2,
)
PART = "xc7z020clg400-1"
TARGET_CLK_NS = 4


def make_split_hw_model(chans, simd, nvec, dt=DataType["INT8"]):
    total = sum(chans)
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, nvec + [total])
    outs = [
        oh.make_tensor_value_info("out%d" % i, TensorProto.FLOAT, nvec + [c])
        for i, c in enumerate(chans)
    ]
    node = oh.make_node(
        "StreamingSplit",
        ["inp"],
        [o.name for o in outs],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=chans,
        inputDataType=dt.name,
        numInputVectors=nvec,
        preferred_impl_style="hls",
    )
    graph = oh.make_graph([node], "split_graph", [inp], outs)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="split-model"))
    for t in [inp] + outs:
        model.set_tensor_datatype(t.name, dt)
    return model


def make_concat_hw_model(chans, simd, nvec, dt=DataType["INT8"]):
    total = sum(chans)
    inps = [
        oh.make_tensor_value_info("inp%d" % i, TensorProto.FLOAT, nvec + [c])
        for i, c in enumerate(chans)
    ]
    out = oh.make_tensor_value_info("outp", TensorProto.FLOAT, nvec + [total])
    node = oh.make_node(
        "StreamingConcat",
        [i.name for i in inps],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=chans,
        inputDataTypes=[dt.name] * len(chans),
        numInputVectors=nvec,
        preferred_impl_style="hls",
    )
    graph = oh.make_graph([node], "concat_graph", inps, [out])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="concat-model"))
    for t in inps + [out]:
        model.set_tensor_datatype(t.name, dt)
    return model


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_split(config):
    chans, simd, nvec = config
    model = make_split_hw_model(chans, simd, nvec)
    assert tree_model_test_up_to_rotation(
        model, ("StreamingSplit", config), PART, TARGET_CLK_NS, **TOLERANCES
    ), "characterized TAV does not match RTLsim'd one!"


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_concat(config):
    chans, simd, nvec = config
    model = make_concat_hw_model(chans, simd, nvec)
    assert tree_model_test_up_to_rotation(
        model, ("StreamingConcat", config), PART, TARGET_CLK_NS, **TOLERANCES
    ), "characterized TAV does not match RTLsim'd one!"

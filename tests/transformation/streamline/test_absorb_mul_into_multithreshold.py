# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as ox
from finn.transformation.streamline.absorb import AbsorbMulIntoMultiThreshold


def make_mul_mt_model(scale, thresholds):
    """Build a Mul(per-channel) -> MultiThreshold model with NC layout."""
    num_ch, num_steps = thresholds.shape
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, num_ch])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [num_ch])
    mul_out = oh.make_tensor_value_info("mul_out", TensorProto.FLOAT, [1, num_ch])
    thres = oh.make_tensor_value_info("thres", TensorProto.FLOAT, [num_ch, num_steps])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, num_ch])

    mul_node = oh.make_node("Mul", ["inp", "mul_param"], ["mul_out"])
    mt_node = oh.make_node(
        "MultiThreshold",
        ["mul_out", "thres"],
        ["outp"],
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(
                name="mul-mt",
                inputs=[inp],
                outputs=[outp],
                value_info=[mul_param, mul_out, thres],
                nodes=[mul_node, mt_node],
            )
        )
    )
    model.set_initializer("mul_param", scale)
    model.set_initializer("thres", thresholds)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


@pytest.mark.streamline
def test_absorb_mul_into_multithreshold_zero_scale():
    # Channels with a zero scale (e.g. dead BatchNorm channels) feed a constant 0
    # into the MultiThreshold. Their absorbed thresholds must stay sorted (RTL
    # thresholding uses binary search) and keep the result: thresholds <= 0 always
    # met, all others never. Channel 1 straddles 0 and has a threshold exactly at 0.
    num_ch, num_steps = 4, 15
    thresholds = np.sort(
        np.random.uniform(-4.0, 4.0, size=(num_ch, num_steps)).astype(np.float32), axis=1
    )
    thresholds[1] = np.linspace(-1.0, 1.0, num_steps, dtype=np.float32)
    scale = np.array([0.5, 0.0, 2.0, 0.0], dtype=np.float32)
    model = make_mul_mt_model(scale, thresholds)

    inp = {"inp": np.random.uniform(-8.0, 8.0, size=(1, num_ch)).astype(np.float32)}
    expected = ox.execute_onnx(model, inp)["outp"]

    model = model.transform(AbsorbMulIntoMultiThreshold())
    assert len(model.get_nodes_by_op_type("Mul")) == 0
    new_thresholds = model.get_initializer(model.graph.node[0].input[1])
    assert new_thresholds.dtype == np.float32
    assert not np.isnan(new_thresholds).any()
    # compare against np.sort, np.diff would turn adjacent infs into NaN
    assert (new_thresholds == np.sort(new_thresholds, axis=-1)).all()
    assert (ox.execute_onnx(model, inp)["outp"] == expected).all()

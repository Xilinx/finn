# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveMulPastJoinMul


def create_mul_model(param_shape):
    """(x*A) * (y*B) with A, B distinct constants, joined by an elementwise Mul."""
    in_shape = [1, 64, 10, 9]

    mul0 = oh.make_node("Mul", inputs=["in1", "mul0_param"], outputs=["mul0_out"])
    mul1 = oh.make_node("Mul", inputs=["in2", "mul1_param"], outputs=["mul1_out"])
    join = oh.make_node("Mul", inputs=["mul0_out", "mul1_out"], outputs=["out_join"])

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, in_shape)
    in2 = oh.make_tensor_value_info("in2", TensorProto.FLOAT, in_shape)
    mul0_out = oh.make_tensor_value_info("mul0_out", TensorProto.FLOAT, in_shape)
    mul1_out = oh.make_tensor_value_info("mul1_out", TensorProto.FLOAT, in_shape)
    out_join = oh.make_tensor_value_info("out_join", TensorProto.FLOAT, in_shape)

    graph = oh.make_graph(
        nodes=[mul0, mul1, join],
        name="test_graph",
        inputs=[in1, in2],
        outputs=[out_join],
        value_info=[mul0_out, mul1_out],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test_model"))

    # distinct constants (A != B) to exercise the A*B folding
    rng = np.random.default_rng(0)
    model.set_initializer("mul0_param", rng.uniform(0.5, 2.0, param_shape).astype(np.float32))
    model.set_initializer("mul1_param", rng.uniform(0.5, 2.0, param_shape).astype(np.float32))
    return model


@pytest.mark.streamline
# scalar constant vs per-channel (NCHW channel dim) constant
@pytest.mark.parametrize("param_shape", [[1], [1, 64, 1, 1]])
def test_move_mul_past_join_mul(param_shape):
    model = create_mul_model(param_shape)

    in0 = model.get_first_global_in()
    in1 = model.graph.input[1].name
    input_shape = model.get_tensor_shape(in0)
    input_dtype = model.get_tensor_datatype(in0)
    input_dict = {
        in0: gen_finn_dt_tensor(input_dtype, input_shape),
        in1: gen_finn_dt_tensor(input_dtype, input_shape),
    }

    model_transformed = model.transform(MoveMulPastJoinMul())

    # numerically equivalent
    assert oxe.compare_execution(model, model_transformed, input_dict)

    # one of the two producer Muls has been folded away
    assert [n.op_type for n in model.graph.node].count("Mul") == 3
    assert [n.op_type for n in model_transformed.graph.node].count("Mul") == 2

    # both graph inputs now feed the join Mul directly, and the surviving Mul
    # is the last node (moved past the join)
    assert model_transformed.find_consumers(in0)[0].op_type == "Mul"
    assert model_transformed.find_consumers(in0)[0] == model_transformed.find_consumers(in1)[0]
    assert (
        model_transformed.find_producer(model_transformed.get_first_global_out()).op_type == "Mul"
    )

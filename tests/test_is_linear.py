import onnx.helper as oh
from onnx import TensorProto

import finn.analysis.topology as ta
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_is_linear_linear():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul_param],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform_single(si.infer_shapes)
    ret = model.analysis(ta.is_linear)
    assert ret["is_linear"] is True


def test_is_linear_forked_node_output():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [2])
    mul0_param = oh.make_tensor_value_info("mul0_param", TensorProto.FLOAT, [2])
    mul1_param = oh.make_tensor_value_info("mul1_param", TensorProto.FLOAT, [2])
    mul0_res = oh.make_tensor_value_info("mul0_res", TensorProto.FLOAT, [2])
    mul1_res = oh.make_tensor_value_info("mul1_res", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul0_param, mul1_param, mul0_res, mul1_res],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul0_param"], ["mul0_res"]),
                oh.make_node("Mul", ["middle", "mul1_param"], ["mul1_res"]),
                oh.make_node("Add", ["mul0_res", "mul1_res"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform_single(si.infer_shapes)
    ret = model.analysis(ta.is_linear)
    assert ret["is_linear"] is False

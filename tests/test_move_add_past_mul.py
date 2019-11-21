import numpy as np
import onnx.helper as oh
from onnx import TensorProto

import finn.core.onnx_exec as ox
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import MoveAddPastMul


def test_move_add_past_mul_single():
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
    model = model.transform(InferShapes())
    model.set_initializer("add_param", np.asarray([1, 3], dtype=np.float32))
    model.set_initializer("mul_param", np.asarray([2, 4], dtype=np.float32))
    new_model = model.transform(MoveAddPastMul())
    inp_dict = {"top_in": np.asarray([-1.0, 1.0], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)


def test_move_add_past_mul_multi():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param_0 = oh.make_tensor_value_info("add_param_0", TensorProto.FLOAT, [2])
    mul_param_0 = oh.make_tensor_value_info("mul_param_0", TensorProto.FLOAT, [2])
    add_param_1 = oh.make_tensor_value_info("add_param_1", TensorProto.FLOAT, [2])
    mul_param_1 = oh.make_tensor_value_info("mul_param_1", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param_0, mul_param_0, add_param_1, mul_param_1],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param_0"], ["middle_0"]),
                oh.make_node("Mul", ["middle_0", "mul_param_0"], ["middle_1"]),
                oh.make_node("Add", ["middle_1", "add_param_1"], ["middle_2"]),
                oh.make_node("Mul", ["middle_2", "mul_param_1"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("add_param_0", np.asarray([1, 3], dtype=np.float32))
    model.set_initializer("mul_param_0", np.asarray([2, 4], dtype=np.float32))
    model.set_initializer("add_param_1", np.asarray([-1, 3], dtype=np.float32))
    model.set_initializer("mul_param_1", np.asarray([2, -4], dtype=np.float32))
    new_model = model.transform(MoveAddPastMul())
    inp_dict = {"top_in": np.asarray([-1.0, 1.0], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)

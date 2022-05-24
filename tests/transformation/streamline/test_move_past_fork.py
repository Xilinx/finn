import pytest

import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.reorder import MoveLinearPastFork


@pytest.mark.streamline
@pytest.mark.parametrize("ch", [64, 1])
# ifmdim
@pytest.mark.parametrize("ifmdim", [-1, 7])
def test_move_past_fork(ch, ifmdim):
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_shape = (1, ch)
    else:
        input_shape = (1, ch, ifmdim, ifmdim)

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)

    num_of_params = 8
    value_info = []
    for i in range(num_of_params):
        value_info += [
            helper.make_tensor_value_info("p" + str(i), TensorProto.FLOAT, input_shape)
        ]

    add_1_to_move = helper.make_node("Add", ["top_in", "p0"], ["fork1"])
    mul_1_to_move = helper.make_node("Mul", ["t5", "p4"], ["fork2"])
    add_2_to_move = helper.make_node("Add", ["fork2", "p5"], ["t6"])
    mul_1_not_to_move = helper.make_node("Mul", ["t8", "p7"], ["fork3"])
    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                # fork1
                add_1_to_move,
                helper.make_node("Mul", ["fork1", "p1"], ["t2"]),
                helper.make_node("Mul", ["fork1", "p2"], ["t3"]),
                helper.make_node("Add", ["t2", "t3"], ["t4"]),
                helper.make_node("Add", ["t4", "p3"], ["t5"]),
                # fork2
                mul_1_to_move,
                add_2_to_move,
                helper.make_node("Add", ["fork2", "p6"], ["t7"]),
                helper.make_node("Add", ["t6", "t7"], ["t8"]),
                # empty branches: do nothing
                mul_1_not_to_move,
                helper.make_node("Add", ["fork3", "fork3"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    for i in range(num_of_params):
        model.set_initializer(
            "p" + str(i), np.random.rand(*input_shape).astype(np.float32)
        )

    # Transform
    new_model = model.transform(MoveLinearPastFork())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    # Test
    assert oxe.compare_execution(model, new_model, inp_dict)
    assert not new_model.is_fork_node(add_1_to_move)
    assert not new_model.is_fork_node(mul_1_to_move)
    assert not new_model.is_fork_node(add_2_to_move)
    assert new_model.is_fork_node(mul_1_not_to_move)
    assert len(new_model.graph.node) == 14

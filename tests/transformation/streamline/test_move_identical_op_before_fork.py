import pytest

import numpy as np
from onnx import helper as oh
from onnx import TensorProto

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.streamline.reorder import MoveTransposeBeforeFork
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe


def create_model(perm, default_data_layout):
    if perm == [0, 3, 1, 2]:
        in_shape = [1, 128, 1, 256]
        out_shape = [1, 256, 128, 1]
        data_layout = "NHWC"
    if perm == [0, 2, 3, 1]:
        in_shape = [1, 256, 128, 1]
        out_shape = [1, 128, 1, 256]
        data_layout = "NCHW"

    if (
        default_data_layout and data_layout == "NCHW"
    ):  # meaning that we will not set the data_layout attribute
        Fork1_node = oh.make_node(
            "MultiThreshold",
            inputs=["in1_multithreshold1", "in2_multithreshold1"],
            outputs=["out_multithreshold1"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
        )
    else:
        Fork1_node = oh.make_node(
            "MultiThreshold",
            inputs=["in1_multithreshold1", "in2_multithreshold1"],
            outputs=["out_multithreshold1"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
            data_layout=data_layout,
        )

    Transpose1_node = oh.make_node(
        "Transpose",
        inputs=["out_multithreshold1"],
        outputs=["out_transpose1"],
        perm=perm,
    )

    Transpose2_node = oh.make_node(
        "Transpose",
        inputs=["out_multithreshold1"],
        outputs=["out_transpose2"],
        perm=perm,
    )

    Add1_node = oh.make_node(
        "Add", inputs=["out_transpose1", "in2_add1"], outputs=["out_add1"]
    )

    Mul1_node = oh.make_node(
        "Mul", inputs=["out_transpose2", "in2_mul1"], outputs=["out_mul1"]
    )

    in1_multithreshold1 = oh.make_tensor_value_info(
        "in1_multithreshold1", TensorProto.FLOAT, in_shape
    )

    in2_multithreshold1 = oh.make_tensor_value_info(
        "in2_multithreshold1", TensorProto.FLOAT, [256, 15]
    )
    out_multithreshold1 = oh.make_tensor_value_info(
        "out_multithreshold1", TensorProto.FLOAT, in_shape
    )
    out_transpose1 = oh.make_tensor_value_info(
        "out_transpose1", TensorProto.FLOAT, out_shape
    )
    out_transpose2 = oh.make_tensor_value_info(
        "out_transpose2", TensorProto.FLOAT, out_shape
    )
    in2_add1 = oh.make_tensor_value_info("in2_add1", TensorProto.FLOAT, [1])
    in2_mul1 = oh.make_tensor_value_info("in2_mul1", TensorProto.FLOAT, [1])

    out_add1 = oh.make_tensor_value_info("out_add1", TensorProto.FLOAT, out_shape)
    out_mul1 = oh.make_tensor_value_info("out_mul1", TensorProto.FLOAT, out_shape)

    graph = oh.make_graph(
        nodes=[Fork1_node, Transpose1_node, Transpose2_node, Add1_node, Mul1_node],
        name="test_graph",
        inputs=[in1_multithreshold1],
        outputs=[out_add1, out_mul1],
        value_info=[
            in2_multithreshold1,
            in2_add1,
            in2_mul1,
            out_multithreshold1,
            out_transpose1,
            out_transpose2,
        ],
    )

    onnx_model = oh.make_model(graph, producer_name="test_model")
    model = ModelWrapper(onnx_model)

    mt_weights = np.random.randint(low=-1000, high=1000, size=[256, 15])
    mt_weights = np.sort(mt_weights, 1)
    model.set_initializer("in2_multithreshold1", mt_weights)

    add_init = np.random.randint(low=-1000, high=1000, size=[1]).astype(np.float32)
    model.set_initializer("in2_add1", add_init)

    mul_init = np.random.randint(low=-1000, high=1000, size=[1]).astype(np.float32)
    model.set_initializer("in2_mul1", mul_init)

    return model


# permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 3, 1, 2], [0, 2, 3, 1]])
# default data layout variable
@pytest.mark.parametrize("default_data_layout", [True, False])
def test_move_identical_op_before_fork(perm, default_data_layout):
    model = create_model(perm, default_data_layout)

    # Create input data
    input0_tensor_name = model.graph.input[0].name
    input_shape = model.get_tensor_shape(input0_tensor_name)
    input_dtype = model.get_tensor_datatype(input0_tensor_name)
    input_val = gen_finn_dt_tensor(input_dtype, input_shape)

    input_dict = {}
    input_dict[input0_tensor_name] = input_val

    model_transformed = model.transform(MoveTransposeBeforeFork())

    assert oxe.compare_execution(model, model_transformed, input_dict)

    # Check if transpose node is before fork, i.e. whether the transpose node is a fork
    transpose_node = [
        n for n in model_transformed.graph.node if n.op_type == "Transpose"
    ]
    assert model_transformed.is_fork_node(transpose_node[0])

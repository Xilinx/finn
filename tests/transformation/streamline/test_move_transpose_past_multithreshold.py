import pytest

import numpy as np
from onnx import helper as oh
from onnx import TensorProto

from finn.core.modelwrapper import ModelWrapper
import finn.core.data_layout as DataLayout
from finn.transformation.streamline.reorder import MoveTransposePastMultiThreshold
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe


def create_model(permutation, default_data_layout):
    if permutation == [0, 3, 1, 2]:
        in_shape = [1, 128, 1, 256]
        out_shape = [1, 256, 128, 1]
        data_layout = "NCHW"
    if permutation == [0, 2, 3, 1]:
        in_shape = [1, 256, 128, 1]
        out_shape = [1, 128, 1, 256]
        data_layout = "NHWC"

    Transpose1_node = oh.make_node(
        "Transpose",
        inputs=["in_transpose1"],
        outputs=["out_transpose1"],
        perm=permutation,
    )

    Transpose2_node = oh.make_node(
        "Transpose",
        inputs=["in_transpose2"],
        outputs=["out_transpose2"],
        perm=permutation,
    )

    if (
        default_data_layout is True and data_layout == "NCHW"
    ):  # meaning that we will not set the data_layout attribute
        Multithreshold1_node = oh.make_node(
            "MultiThreshold",
            inputs=["out_transpose1", "in2_multithreshold1"],
            outputs=["out_multithreshold1"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
        )

        Multithreshold2_node = oh.make_node(
            "MultiThreshold",
            inputs=["out_transpose2", "in2_multithreshold2"],
            outputs=["out_multithreshold2"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
        )
    else:  # we set the data_layout attribute
        Multithreshold1_node = oh.make_node(
            "MultiThreshold",
            inputs=["out_transpose1", "in2_multithreshold1"],
            outputs=["out_multithreshold1"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
            data_layout=data_layout,
        )

        Multithreshold2_node = oh.make_node(
            "MultiThreshold",
            inputs=["out_transpose2", "in2_multithreshold2"],
            outputs=["out_multithreshold2"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
            data_layout=data_layout,
        )

    Add1_node = oh.make_node(
        "Add",
        inputs=["out_multithreshold1", "out_multithreshold2"],
        outputs=["out_add1"],
    )

    in_transpose1 = oh.make_tensor_value_info(
        "in_transpose1", TensorProto.FLOAT, in_shape
    )
    in_transpose2 = oh.make_tensor_value_info(
        "in_transpose2", TensorProto.FLOAT, in_shape
    )
    out_add1 = oh.make_tensor_value_info("out_add1", TensorProto.FLOAT, out_shape)

    out_transpose1 = oh.make_tensor_value_info(
        "out_transpose1", TensorProto.FLOAT, out_shape
    )
    out_transpose2 = oh.make_tensor_value_info(
        "out_transpose2", TensorProto.FLOAT, out_shape
    )
    out_multithreshold1 = oh.make_tensor_value_info(
        "out_multithreshold1", TensorProto.FLOAT, out_shape
    )
    out_multithreshold2 = oh.make_tensor_value_info(
        "out_multithreshold2", TensorProto.FLOAT, out_shape
    )

    in2_multithreshold1 = oh.make_tensor_value_info(
        "in2_multithreshold1", TensorProto.FLOAT, [256, 15]
    )
    in2_multithreshold2 = oh.make_tensor_value_info(
        "in2_multithreshold2", TensorProto.FLOAT, [256, 15]
    )

    graph = oh.make_graph(
        nodes=[
            Transpose1_node,
            Transpose2_node,
            Multithreshold1_node,
            Multithreshold2_node,
            Add1_node,
        ],
        name="test_graph",
        inputs=[in_transpose1, in_transpose2],
        outputs=[out_add1],
        value_info=[
            out_transpose1,
            out_transpose2,
            out_multithreshold1,
            out_multithreshold2,
            in2_multithreshold1,
            in2_multithreshold2,
        ],
    )

    onnx_model = oh.make_model(graph, producer_name="test_model")
    model = ModelWrapper(onnx_model)

    mt_weights = np.random.randint(low=-1000, high=1000, size=[256, 15])
    mt_weights = np.sort(mt_weights, 1)
    model.set_initializer("in2_multithreshold1", mt_weights)
    model.set_initializer("in2_multithreshold2", mt_weights)

    return model


# permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 3, 1, 2], [0, 2, 3, 1]])
# default data layout variable
@pytest.mark.parametrize("default_data_layout", [True, False])
def test_move_transpose_past_multithreshold(perm, default_data_layout):
    model = create_model(perm, default_data_layout)

    # Create input data
    input0_tensor_name = model.graph.input[0].name
    input1_tensor_name = model.graph.input[1].name

    # Note: it is assumed that both tensors have the same shape and data type
    input_shape = model.get_tensor_shape(input0_tensor_name)
    input_dtype = model.get_tensor_datatype(input0_tensor_name)
    input_val = gen_finn_dt_tensor(input_dtype, input_shape)
    input_dict = {}
    input_dict[input0_tensor_name] = input_val
    input_dict[input1_tensor_name] = input_val

    model_transformed = model.transform(MoveTransposePastMultiThreshold())

    assert oxe.compare_execution(model, model_transformed, input_dict)

    # Check if order changed
    node0_input0_model = model.find_consumer(model.graph.input[0].name).op_type
    node1_input1_model = model.find_consumer(model.graph.input[1].name).op_type
    node0_input0_model_transformed = model_transformed.find_consumer(
        model_transformed.graph.input[0].name
    ).op_type
    node1_input1_model_transformed = model_transformed.find_consumer(
        model_transformed.graph.input[1].name
    ).op_type
    assert node0_input0_model != node0_input0_model_transformed
    assert node1_input1_model != node1_input1_model_transformed

    # Check if data_layout is set correctly
    mt0_input = model_transformed.graph.node[0].input[0]
    mt1_input = model_transformed.graph.node[1].input[0]
    mt0_output = model_transformed.graph.node[0].output[0]
    mt1_output = model_transformed.graph.node[1].output[0]
    if perm == [0, 3, 1, 2]:
        assert model_transformed.get_tensor_layout(mt0_input) == DataLayout.NHWC
        assert model_transformed.get_tensor_layout(mt1_input) == DataLayout.NHWC
        assert model_transformed.get_tensor_layout(mt0_output) == DataLayout.NHWC
        assert model_transformed.get_tensor_layout(mt1_output) == DataLayout.NHWC
    if perm == [0, 2, 3, 1]:
        assert model_transformed.get_tensor_layout(mt0_input) == DataLayout.NCHW
        assert model_transformed.get_tensor_layout(mt1_input) == DataLayout.NCHW
        assert model_transformed.get_tensor_layout(mt0_output) == DataLayout.NCHW
        assert model_transformed.get_tensor_layout(mt1_output) == DataLayout.NCHW

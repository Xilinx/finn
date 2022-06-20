import pytest

import onnx
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC


def create_maxpool(ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt):
    ofm_dim_h = compute_pool_output_dim(
        ifm_dim[0], kernel_shape[0], strides[0], pads[0], ceil_mode
    )
    ofm_dim_w = compute_pool_output_dim(
        ifm_dim[1], kernel_shape[1], strides[1], pads[1], ceil_mode
    )
    inp = oh.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim[0], ifm_dim[1]]
    )
    outp_mp = oh.make_tensor_value_info(
        "outp_mp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim_h, ofm_dim_w]
    )
    outp = oh.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    maxpool_node = oh.make_node(
        "MaxPool",
        inputs=["inp"],
        outputs=["out_mp"],
        ceil_mode=ceil_mode,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=["out_mp"],
        outputs=["outp"],
        name="Transpose1",
        perm=[0, 2, 3, 1],
    )

    graph = oh.make_graph(
        nodes=[maxpool_node, transpose_node],
        name="maxpool_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_mp],
    )

    model = oh.make_model(graph, producer_name="maxpool_model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    model = model.transform(InferShapes())

    return model


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("ifm_dim", [[8, 8], [9, 9]])
# input channels
@pytest.mark.parametrize("ifm_ch", [3])
# kernel shape
@pytest.mark.parametrize("kernel_shape", [[2, 2]])
# padding
@pytest.mark.parametrize("pads", [[0, 0, 0, 0], [1, 1, 1, 1]])
# strides
@pytest.mark.parametrize("strides", [[2, 2]])
# ceil_mode
@pytest.mark.parametrize("ceil_mode", [0, 1])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
def test_maxpool_nhwc(ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt):
    # create MaxPool node
    maxpool_model = create_maxpool(
        ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt
    )

    # generate input tensor for testing
    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim[0], ifm_dim[1]])
    input_dict = {"inp": input_tensor}

    # execute first model
    output_dict = oxe.execute_onnx(maxpool_model, input_dict)
    expected = output_dict["outp"]

    # transform MaxPool into MaxPoolNHWC
    maxpool_model = maxpool_model.transform(MakeMaxPoolNHWC())

    # execute transformed model
    output_node_name = maxpool_model.graph.output[0].name
    output_dict = oxe.execute_onnx(
        maxpool_model, input_dict, return_full_exec_context=False
    )
    output = output_dict[output_node_name]

    # compare outputs
    assert (expected == output).all()

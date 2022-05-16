import pytest

import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.reorder import MoveMulPastMaxPool
from finn.util.basic import gen_finn_dt_tensor


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("ifm_dim", [4, 7])
# input channels
@pytest.mark.parametrize("ifm_ch", [1, 3])
# kernel size
@pytest.mark.parametrize("k", [2, 3])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("pad", [0, 1])
# channelwise or scalar mul
@pytest.mark.parametrize("cw", [0, 1])
# negative mul
@pytest.mark.parametrize("negative", [0, 1])
def test_move_mul_past_maxpool(ifm_dim, ifm_ch, k, stride, pad, cw, negative):
    if cw == 1:
        mul_shape = [1, ifm_ch, 1, 1]
    else:
        mul_shape = [1, 1, 1, 1]

    ofm_ch = ifm_ch
    ofm_dim = compute_pool_output_dim(ifm_dim, k, stride, pad)

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    mul = helper.make_tensor_value_info("mul", TensorProto.FLOAT, mul_shape)
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim, ofm_dim]
    )

    Mul_node = helper.make_node("Mul", ["inp", "mul"], ["mul_out"])

    Maxpool_node = helper.make_node(
        "MaxPool",
        ["mul_out"],
        ["outp"],
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        strides=[stride, stride],
    )

    graph = helper.make_graph(
        nodes=[Mul_node, Maxpool_node],
        name="mulpastmaxpool_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mul],
    )

    model = helper.make_model(graph, producer_name="mulpastmaxpool-model")
    model = ModelWrapper(model)
    inp_values = gen_finn_dt_tensor(DataType["INT2"], [1, ifm_ch, ifm_dim, ifm_dim])
    mul_values = np.random.random_sample(mul_shape).astype(np.float32)
    if negative == 1:
        mul_values = mul_values * (-1)
    model.set_initializer("mul", mul_values)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    idict = {"inp": inp_values}
    odict = oxe.execute_onnx(model, idict, True)
    out_before = odict["outp"]

    # perform transformation
    model_transformed = model.transform(MoveMulPastMaxPool())
    odict = oxe.execute_onnx(model_transformed, idict, True)
    out_after = odict["outp"]

    assert (out_before == out_after).all()

    if negative == 1:
        assert model.graph.node[0].op_type == model_transformed.graph.node[0].op_type
        assert model.graph.node[1].op_type == model_transformed.graph.node[1].op_type
    else:
        assert model.graph.node[0].op_type == model_transformed.graph.node[1].op_type
        assert model.graph.node[1].op_type == model_transformed.graph.node[0].op_type

# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferConvInpGen,
    InferQuantizedMatrixVectorActivation,
)
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import (
    InferPixelPaddingDeconv,
)


def make_convtranspose_model(idt, wdt, k, idim, ifm_ch, ofm_ch, stride, pads, output_padding, dil):
    """A single ConvTranspose with INT32 output; the output shape comes from InferShapes."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, *idim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, None)
    node = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        dilations=dil,
        group=1,
        kernel_shape=(k, k),
        pads=pads,
        output_padding=output_padding,
        strides=stride,
    )
    graph = helper.make_graph([node], "convtranspose_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="convtranspose-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("W", wdt)
    model.set_tensor_datatype("outp", DataType["INT32"])
    model.set_initializer("W", gen_finn_dt_tensor(wdt, [ifm_ch, ofm_ch, k, k]))
    return model.transform(InferShapes())


# kernel size
@pytest.mark.parametrize("k", [1, 2, 3])
# stride [h, w]
@pytest.mark.parametrize("stride", [[1, 1], [2, 3]])
# ConvTranspose pads [H_begin, W_begin, H_end, W_end]; beyond dilation * (k - 1) they crop
@pytest.mark.parametrize("pads", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 1, 0, 2]])
# output_padding [h, w]
@pytest.mark.parametrize("output_padding", [[0, 0], [1, 0]])
# dilation [h, w]
@pytest.mark.parametrize("dil", [[1, 1], [2, 1]])
@pytest.mark.transform
def test_infer_pixel_padding_deconv(k, stride, pads, output_padding, dil):
    """Lowering a ConvTranspose, including pads that crop (pads > dilation * (k - 1),
    issue #1484), keeps the output and the SWG + MVAU conversion of the lowered graph
    consistent with the ConvTranspose."""
    if output_padding[0] >= stride[0]:
        pytest.skip("output_padding must be smaller than the stride")
    if k == 1 and dil != [1, 1]:
        pytest.skip("dilation has no effect for k=1")
    idt = wdt = DataType["INT4"]
    ifm_ch, ofm_ch = 2, 3
    model = make_convtranspose_model(
        idt, wdt, k, [7, 6], ifm_ch, ofm_ch, stride, pads, output_padding, dil
    )
    oshape = model.get_tensor_shape("outp")
    if min(oshape[2:]) < 1:
        pytest.skip("pads crop the whole output")
    input_dict = {"inp": gen_finn_dt_tensor(idt, [1, ifm_ch, 7, 6])}
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(InferPixelPaddingDeconv())
    conv_extent = [dil[0] * (k - 1), dil[1] * (k - 1)]
    need_crop = (
        max(pads[0], pads[2] - output_padding[0]) > conv_extent[0]
        or max(pads[1], pads[3] - output_padding[1]) > conv_extent[1]
    )
    assert len(model.get_nodes_by_op_type("Crop")) == int(need_crop)
    for node in model.get_nodes_by_op_type("Im2Col"):
        assert all(p >= 0 for p in getCustomOp(node).get_nodeattr("pad_amount"))
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert y_produced.shape == y_expected.shape
    assert (y_produced == y_expected).all()

    model = model.transform(InferConvInpGen())
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    assert not model.get_nodes_by_op_type("Im2Col")
    assert not model.get_nodes_by_op_type("MatMul")
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

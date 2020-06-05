import os

import onnx  # noqa
import torch
import numpy as np
import brevitas.onnx as bo
from brevitas.nn import QuantAvgPool2d
from brevitas.quant_tensor import pack_quant_tensor
from brevitas.core.quant import QuantType
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe

import pytest

export_onnx_path = "test_avg_pool.onnx"


@pytest.mark.parametrize("kernel_size", [7])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("signed", [False])
@pytest.mark.parametrize("bit_width", [4])
def test_brevitas_avg_pool_export(kernel_size, stride, signed, bit_width):
    ch = 4
    ishape = (1, ch, 7, 7)
    input_bit_width = 32
    ibw_tensor = torch.Tensor([input_bit_width])

    b_avgpool = QuantAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        signed=signed,
        min_overall_bit_width=bit_width,
        max_overall_bit_width=bit_width,
        quant_type=QuantType.INT,
    )
    # call forward pass manually once to cache scale factor and bitwidth
    input_tensor = torch.from_numpy(np.zeros(ishape)).float()
    output_scale = torch.from_numpy(np.ones((1, ch, 1, 1))).float()
    input_quant_tensor = pack_quant_tensor(
        tensor=input_tensor, scale=output_scale, bit_width=ibw_tensor
    )
    bo.export_finn_onnx(b_avgpool, ishape, export_onnx_path, input_t=input_quant_tensor)
    model = ModelWrapper(export_onnx_path)
    # set FINN datatype
    if signed is True:
        prefix = "INT"
    else:
        prefix = "UINT"
    dt_name = prefix + str(input_bit_width)
    dtype = DataType[dt_name]
    model.set_tensor_datatype(model.graph.input[0].name, dtype)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # calculate golden output
    inp = gen_finn_dt_tensor(dtype, ishape)
    input_tensor = torch.from_numpy(inp).float()
    input_quant_tensor = pack_quant_tensor(
        tensor=input_tensor, scale=output_scale, bit_width=ibw_tensor
    )
    b_avgpool.eval()
    expected = b_avgpool.forward(input_quant_tensor).tensor.detach().numpy()

    # finn execution
    idict = {model.graph.input[0].name: inp}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    assert (expected == produced).all()

    os.remove(export_onnx_path)

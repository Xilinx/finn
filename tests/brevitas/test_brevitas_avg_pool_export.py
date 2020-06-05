import onnx  # noqa
import torch
import numpy as np
import brevitas.onnx as bo
from brevitas.nn import QuantAvgPool2d
from brevitas.quant_tensor import pack_quant_tensor
from brevitas.core.quant import QuantType

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

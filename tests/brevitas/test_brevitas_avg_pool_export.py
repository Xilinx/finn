import onnx  # noqa
import brevitas.onnx as bo
from brevitas.nn import QuantAvgPool2d
from brevitas.core.quant import QuantType
import pytest

export_onnx_path = "test_avg_pool.onnx"


@pytest.mark.parametrize("kernel_size", [7])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("signed", [False])
@pytest.mark.parametrize("bit_width", [4])
def test_brevitas_avg_pool_export(kernel_size, stride, signed, bit_width):
    ishape = (1, 1024, 7, 7)

    b_avgpool = QuantAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        signed=signed,
        min_overall_bit_width=bit_width,
        max_overall_bit_width=bit_width,
        quant_type=QuantType.INT,
    )
    bo.export_finn_onnx(b_avgpool, ishape, export_onnx_path)

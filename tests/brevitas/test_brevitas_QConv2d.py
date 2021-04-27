import pytest
import os
import numpy as np
import torch
import brevitas.onnx as bo
from brevitas.nn import QuantConv2d
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor

export_onnx_path = "test_brevitas_conv.onnx"


@pytest.mark.parametrize("dw", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("in_channels", [32])
def test_brevitas_QConv2d(dw, bias, in_channels):
    ishape = (1, 32, 111, 111)
    if dw is True:
        groups = in_channels
        out_channels = in_channels
        kernel_size = 3
        padding = 1
        stride = 1
        w_shape = (32, 1, 3, 3)

    else:
        groups = 1
        out_channels = 64
        kernel_size = 1
        padding = 0
        stride = 1
        w_shape = (64, 32, 1, 1)

    b_conv = QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias,
        bias_quant_type=QuantType.FP,
        compute_output_bit_width=False,
        compute_output_scale=False,
        weight_bit_width=4,
        weight_quant_type=QuantType.INT,
        weight_scaling_impl_type=ScalingImplType.STATS,
        weight_scaling_stats_op=StatsOp.MAX,
        weight_scaling_per_output_channel=True,
        weight_restrict_scaling_type=RestrictValueType.LOG_FP,
        weight_narrow_range=True,
        weight_scaling_min_val=2e-16,
    )
    weight_tensor = gen_finn_dt_tensor(DataType.INT4, w_shape)
    b_conv.weight = torch.nn.Parameter(torch.from_numpy(weight_tensor).float())

    bo.export_finn_onnx(b_conv, ishape, export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=-1.0, high=1.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_conv.eval()
    expected = b_conv.forward(inp_tensor).detach().numpy()

    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)

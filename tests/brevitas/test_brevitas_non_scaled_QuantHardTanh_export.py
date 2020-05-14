import os
import onnx  # noqa
import numpy as np
import torch
import brevitas.onnx as bo
from brevitas.nn import QuantHardTanh
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import pytest
from finn.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes
from brevitas.core.quant import QuantType

export_onnx_path = "test_act.onnx"


@pytest.mark.parametrize("abits", [1, 2, 4, 8])
@pytest.mark.parametrize("narrow_range", [False, True])
@pytest.mark.parametrize("max_val", [1.0, 1 - 2 ** (-7)])
def test_brevitas_act_export_qhardtanh_nonscaled(abits, narrow_range, max_val):
    def get_quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT

    act_quant_type = get_quant_type(abits)
    min_val = -1.0
    ishape = (1, 10)
    b_act = QuantHardTanh(
        bit_width=abits,
        quant_type=act_quant_type,
        max_val=max_val,
        min_val=min_val,
        restrict_scaling_type=RestrictValueType.LOG_FP,
        scaling_impl_type=ScalingImplType.CONST,
        narrow_range=narrow_range,
    )
    bo.export_finn_onnx(b_act, ishape, export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=min_val, high=max_val, size=ishape).astype(
        np.float32
    )
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    expected = b_act.forward(inp_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)

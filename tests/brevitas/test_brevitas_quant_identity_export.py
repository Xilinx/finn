# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import onnx  # noqa
import os
import torch
from brevitas.core.scaling import ScalingImplType
from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Uint8ActPerTensorFloat
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_preferred_onnx_opset
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir, robust_rmtree


@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("ishape", [(1, 15), (1, 32, 1, 1)])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("quant", [Int8ActPerTensorFloat, Uint8ActPerTensorFloat])
def test_brevitas_quant_identity_export(abits, ishape, narrow, quant):
    build_dir = make_build_dir("test_brevitas_quant_identity_export_")
    export_path = os.path.join(build_dir, "quant_identity.onnx")
    b_act = QuantIdentity(act_quant=quant, bit_width=abits, narrow_range=narrow)

    export_qonnx(
        b_act,
        torch.randn(ishape),
        export_path,
        opset_version=get_preferred_onnx_opset(),
    )
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())

    inp_tensor = np.random.uniform(low=-10.0, high=10.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_act.eval()
    expected = b_act.forward(inp_tensor).detach().numpy()

    # kept for diagnosis on failure: robust_rmtree below is skipped if the
    # assert raises (see tests/README.md)
    assert np.isclose(produced, expected, atol=1e-3).all()
    robust_rmtree(build_dir)


@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("ishape", [(1, 15, 4, 4), (1, 32, 1, 1)])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("quant", [Int8ActPerTensorFloat, Uint8ActPerTensorFloat])
def test_brevitas_quant_identity_export_per_channel(abits, ishape, narrow, quant):
    ch = ishape[1]
    build_dir = make_build_dir("test_brevitas_quant_identity_export_per_channel_")
    export_path = os.path.join(build_dir, "quant_identity.onnx")
    b_act = QuantIdentity(
        act_quant=quant,
        bit_width=abits,
        narrow_range=narrow,
        min_val=-6.0,
        max_val=6.0,
        scaling_impl_type=ScalingImplType.CONST,
        scaling_per_output_channel=True,
        per_channel_broadcastable_shape=(1, ch, 1, 1),
    )

    export_qonnx(
        b_act,
        torch.randn(ishape),
        export_path,
        opset_version=get_preferred_onnx_opset(),
    )
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())

    inp_tensor = np.random.uniform(low=-10.0, high=10.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_act.eval()
    expected = b_act.forward(inp_tensor).detach().numpy()

    # kept for diagnosis on failure: robust_rmtree below is skipped if the
    # assert raises (see tests/README.md)
    assert np.isclose(produced, expected, atol=1e-3).all()
    robust_rmtree(build_dir)

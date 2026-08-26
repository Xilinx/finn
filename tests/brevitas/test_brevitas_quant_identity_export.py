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
# The per-channel scale shape decides the channel axis (and thus the derived
# MultiThreshold data layout), so cover channels-first and channels-last in
# both 3D and 4D as well as the 2D-collapsed case.
@pytest.mark.parametrize(
    "ishape, channel_shape",
    [
        pytest.param((1, 8, 4, 4), (1, 8, 1, 1), id="nchw"),
        pytest.param((1, 4, 4, 8), (1, 1, 1, 8), id="nhwc"),
        pytest.param((1, 4, 8), (1, 1, 8), id="nwc"),
        pytest.param((1, 8, 4), (1, 8, 1), id="ncw"),
        pytest.param((1, 32, 1, 1), (1, 32, 1, 1), id="nchw-singleton"),
    ],
)
# When True, drop leading unit dims from the scale (e.g. (1,1,1,8) -> (8,)) to
# cover the broadcasting right-alignment path in _get_channel_axis.
@pytest.mark.parametrize("squeeze_leading", [False, True])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("quant", [Int8ActPerTensorFloat, Uint8ActPerTensorFloat])
def test_brevitas_quant_identity_export_per_channel(
    abits, ishape, channel_shape, narrow, quant, squeeze_leading
):
    build_dir = make_build_dir("test_brevitas_quant_identity_export_per_channel_")
    export_path = os.path.join(build_dir, "quant_identity.onnx")
    b_act = QuantIdentity(
        act_quant=quant,
        bit_width=abits,
        narrow_range=narrow,
        min_val=-6.0,
        max_val=6.0,
        scaling_impl_type=ScalingImplType.PARAMETER,
        scaling_per_output_channel=True,
        per_channel_broadcastable_shape=channel_shape,
    )

    export_qonnx(
        b_act,
        torch.randn(ishape),
        export_path,
        opset_version=get_preferred_onnx_opset(),
    )
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    # brevitas must export the per-channel scale with the intended (broadcast)
    # shape, otherwise this layout case would not actually exercise the target
    # channel axis in the Quant op
    quant_scale = model.get_initializer(model.get_nodes_by_op_type("Quant")[0].input[1])
    padded_scale_shape = (1,) * (len(ishape) - quant_scale.ndim) + tuple(quant_scale.shape)
    assert padded_scale_shape == tuple(channel_shape), (
        "brevitas did not preserve the per-channel scale shape in the Quant node: "
        f"{quant_scale.shape} vs expected {channel_shape}"
    )
    if squeeze_leading:
        # Reduce the Quant scale to a leading-1s-omitted shape so its channel axis
        # can only be recovered via broadcasting right-alignment in
        # _get_channel_axis (e.g. nhwc (1,1,1,8) -> (8,), nchw (1,8,1,1) ->
        # (8,1,1)). Only leading unit dims are dropped; trailing unit dims are
        # significant for the channel position and must be kept.
        q_scale_name = model.get_nodes_by_op_type("Quant")[0].input[1]
        scale = model.get_initializer(q_scale_name)
        reduced_shape = tuple(scale.shape)
        while len(reduced_shape) > 1 and reduced_shape[0] == 1:
            reduced_shape = reduced_shape[1:]
        model.set_initializer(q_scale_name, scale.reshape(reduced_shape))
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

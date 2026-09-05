# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test that PWPolyFActivation can be exported alongside Brevitas quantized layers.
"""

import pytest

import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantLinear
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPWPolyFLayer
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.torch_hw_modules import PWPolyFActivation


class QuantLinearWithPWPolyF(nn.Module):
    """A simple model mixing Brevitas quantized linear with PWPolyF activation."""

    def __init__(self, in_features, out_features, func="gelu"):
        super().__init__()
        self.quant_inp = QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.linear = QuantLinear(
            in_features,
            out_features,
            bias=False,
            weight_bit_width=4,
            return_quant_tensor=False,
        )
        self.act = PWPolyFActivation(func=func, K=3, degree=2)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.linear(x)
        x = self.act(x)
        return x


@pytest.mark.parametrize("func", ["gelu", "silu"])
@pytest.mark.brevitas_export
def test_brevitas_pwpolyf_mixed_export(func):
    """Test that a model with both Brevitas and PWPolyF layers exports correctly."""
    in_features = 8
    out_features = 16

    model = QuantLinearWithPWPolyF(in_features, out_features, func=func)
    model.eval()

    x = torch.randn(1, in_features)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name

    try:
        export_qonnx(model, x, export_path)

        onnx_model = ModelWrapper(export_path)

        # Check that PWPolyFunction node is present
        pwp_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyFunction"]
        assert len(pwp_nodes) == 1, "Expected one PWPolyFunction node in exported model"

        node = pwp_nodes[0]
        func_attr = {a.name: a for a in node.attribute}
        assert func_attr["func"].s.decode("utf-8") == func
        assert func_attr["K"].i == 3
        assert func_attr["degree"].i == 2

        # Check that Quant nodes are also present (from Brevitas)
        quant_nodes = [n for n in onnx_model.graph.node if n.op_type == "Quant"]
        assert len(quant_nodes) >= 1, "Expected at least one Quant node from Brevitas"

    finally:
        os.unlink(export_path)


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.brevitas_export
def test_brevitas_pwpolyf_convert_to_finn(func):
    """Test the full flow: export mixed model, convert to FINN, and execute."""
    in_features = 8
    out_features = 16

    model = QuantLinearWithPWPolyF(in_features, out_features, func=func)
    model.eval()

    x = torch.randn(1, in_features)

    with torch.no_grad():
        y_pytorch = model(x).numpy()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name

    try:
        export_qonnx(model, x, export_path)

        onnx_model = ModelWrapper(export_path)
        onnx_model = onnx_model.transform(ConvertQONNXtoFINN())
        onnx_model = onnx_model.transform(InferShapes())
        onnx_model = onnx_model.transform(InferPWPolyFLayer())

        # Check that PWPolyF HW node is present after inference
        pwp_hw_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyF"]
        assert len(pwp_hw_nodes) == 1, "Expected one PWPolyF HW node after InferPWPolyFLayer"

        # Execute and compare
        input_dict = {onnx_model.graph.input[0].name: x.numpy()}
        output_dict = oxe.execute_onnx(onnx_model, input_dict)
        y_finn = output_dict[onnx_model.graph.output[0].name]

        # Note: tolerance is higher due to quantization effects
        assert np.allclose(y_pytorch, y_finn, atol=0.5), "FINN execution differs from PyTorch"

    finally:
        os.unlink(export_path)

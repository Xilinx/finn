# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
import os
import tempfile
import torch
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.util.pwpolyf import PiecewisePolyActivation
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPWPolyFLayer
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xczu3eg-sbva484-1-e"


def make_pwpolyf_modelwrapper(func, K, num_channels, num_input_vecs):
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, num_input_vecs + [num_channels]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, num_input_vecs + [num_channels]
    )

    pwpolyf_node = helper.make_node(
        "PWPolyF",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        func=func,
        K=K,
        NumChannels=num_channels,
        PE=1,
        inputDataType="FLOAT32",
        outputDataType="FLOAT32",
        numInputVectors=num_input_vecs,
        name="PWPolyF_0",
    )

    graph = helper.make_graph(
        nodes=[pwpolyf_node],
        name="pwpolyf_graph",
        inputs=[inp],
        outputs=[outp],
    )
    model = helper.make_model(graph, producer_name="pwpolyf-test")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    return model


@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.parametrize("num_channels", [4, 16])
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.fpgadataflow
def test_pwpolyf_cppsim(func, num_channels, num_input_vecs, fold):
    K = 3
    if fold == -1:
        fold = num_channels
    pe = num_channels // fold
    if num_channels % pe != 0:
        pytest.skip("Invalid folding configuration.")

    model = make_pwpolyf_modelwrapper(func, K, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    input_shape = tuple(num_input_vecs + [num_channels])
    x = np.random.uniform(-10, 10, input_shape).astype(np.float32)

    ref_mod = PiecewisePolyActivation(func, K=K)
    with torch.no_grad():
        y_expected = ref_mod(torch.from_numpy(x)).numpy()

    input_dict = {"inp": x}
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    assert y_produced.shape == y_expected.shape
    assert np.allclose(y_produced, y_expected, atol=1e-6)


@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.fpgadataflow
def test_pwpolyf_onnx_export(func):
    K = 3
    num_channels = 32
    mod = PiecewisePolyActivation(func, K=K)
    mod.eval()
    dummy = torch.randn(1, num_channels)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmpf = f.name
    try:
        torch.onnx.export(
            mod, dummy, tmpf,
            input_names=["input"], output_names=["output"],
            opset_version=13, dynamo=False,
        )
        import onnx
        onnx_model = onnx.load(tmpf)
    finally:
        os.unlink(tmpf)

    pwp_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyF"]
    assert len(pwp_nodes) == 1
    node = pwp_nodes[0]
    func_attr = {a.name: a for a in node.attribute}
    assert func_attr["func"].s.decode("utf-8") == func
    assert func_attr["K"].i == K


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_transform(func):
    K = 3
    num_channels = 16
    mod = PiecewisePolyActivation(func, K=K)
    mod.eval()
    dummy = torch.randn(1, num_channels)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmpf = f.name
    try:
        torch.onnx.export(
            mod, dummy, tmpf,
            input_names=["inp"], output_names=["outp"],
            opset_version=13, dynamo=False,
        )
        model = ModelWrapper(tmpf)
    finally:
        os.unlink(tmpf)

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain != "finn.custom_op.fpgadataflow"

    model = model.transform(InferPWPolyFLayer())

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == func
    assert inst.get_nodeattr("K") == K
    assert inst.get_nodeattr("NumChannels") == num_channels
    assert inst.get_nodeattr("PE") == 1
    assert inst.get_nodeattr("inputDataType") == "FLOAT32"

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {"inp": x}
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    ref_mod = PiecewisePolyActivation(func, K=K)
    with torch.no_grad():
        y_expected = ref_mod(torch.from_numpy(x)).numpy()
    assert np.allclose(y_produced, y_expected, atol=1e-6)


@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.fpgadataflow
def test_pwpolyf_specialize_rtl(func):
    K = 3
    num_channels = 8
    model = make_pwpolyf_modelwrapper(func, K, num_channels, [1])
    model = model.transform(SpecializeLayers(test_fpga_part))

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    assert node.domain == "finn.custom_op.fpgadataflow.rtl"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == func
    assert inst.get_nodeattr("K") == K


@pytest.mark.parametrize("func", ["gelu", "tanh"])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.fpgadataflow
def test_pwpolyf_resource_estimates(func, pe):
    K = 3
    num_channels = 8
    model = make_pwpolyf_modelwrapper(func, K, num_channels, [1])
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    assert inst.dsp_estimation() == 2 * pe
    assert inst.lut_estimation() == 200 * pe
    assert inst.bram_estimation() == 0
    assert inst.uram_estimation() == 0


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.fpgadataflow
def test_pwpolyf_folded_shape(func):
    K = 3
    num_channels = 12
    num_input_vecs = [1, 3, 3]
    model = make_pwpolyf_modelwrapper(func, K, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)

    # PE=1
    assert inst.get_normal_input_shape() == (1, 3, 3, 12)
    assert inst.get_normal_output_shape() == (1, 3, 3, 12)
    assert inst.get_folded_input_shape() == (1, 3, 3, 12, 1)
    assert inst.get_folded_output_shape() == (1, 3, 3, 12, 1)

    # PE=4
    inst.set_nodeattr("PE", 4)
    assert inst.get_folded_input_shape() == (1, 3, 3, 3, 4)
    assert inst.get_folded_output_shape() == (1, 3, 3, 3, 4)
    assert inst.get_instream_width() == 4 * 32
    assert inst.get_outstream_width() == 4 * 32


@pytest.mark.parametrize("func", ["gelu", "silu"])
@pytest.mark.fpgadataflow
def test_pwpolyf_exp_cycles(func):
    """Verify expected cycle count estimation."""
    K = 3
    num_channels = 8
    pe = 2
    num_input_vecs = [1, 4, 4]
    model = make_pwpolyf_modelwrapper(func, K, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    # folded shape = (1, 4, 4, 4, 2), exp_cycles = prod of all but last = 1*4*4*4 = 64
    exp = inst.get_exp_cycles()
    assert exp == 1 * 4 * 4 * (num_channels // pe)

    # exp_cycles_per_layer analysis only runs on specialized (rtl/hls) nodes
    model = model.transform(SpecializeLayers(test_fpga_part))
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    exp_dict = model.analysis(exp_cycles_per_layer)
    assert node.name in exp_dict
    assert exp_dict[node.name] == exp

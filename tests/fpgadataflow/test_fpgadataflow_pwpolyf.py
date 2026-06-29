# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

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
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPWPolyFLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.torch_hw_modules import PWPolyFActivation

test_fpga_part = "xcvc1902-vsva2197-2MP-e-S"
non_versal_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


def make_pwpolyf_modelwrapper(func, K, num_channels, num_input_vecs):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, num_input_vecs + [num_channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [num_channels])

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


def make_pwpolyf_rtl_inst(K=3, degree=2):
    model = make_pwpolyf_modelwrapper("gelu", K, 4, [1])
    model = model.transform(SpecializeLayers(test_fpga_part))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("degree", degree)
    return inst


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

    ref_mod = PWPolyFActivation(func, K=K)
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
    degree = 3
    num_channels = 32
    mod = PWPolyFActivation(func, K=K, degree=degree)
    mod.eval()
    dummy = torch.randn(1, num_channels)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmpf = f.name
    try:
        torch.onnx.export(
            mod,
            dummy,
            tmpf,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        import onnx  # noqa: PLC0415

        onnx_model = onnx.load(tmpf)
    finally:
        os.unlink(tmpf)

    pwp_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyF"]
    assert len(pwp_nodes) == 1
    node = pwp_nodes[0]
    func_attr = {a.name: a for a in node.attribute}
    assert func_attr["func"].s.decode("utf-8") == func
    assert func_attr["K"].i == K
    assert func_attr["degree"].i == degree


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_transform(func):
    K = 3
    degree = 3
    num_channels = 16
    mod = PWPolyFActivation(func, K=K, degree=degree)
    mod.eval()
    dummy = torch.randn(1, num_channels)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmpf = f.name
    try:
        torch.onnx.export(
            mod,
            dummy,
            tmpf,
            input_names=["inp"],
            output_names=["outp"],
            opset_version=13,
            dynamo=False,
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
    assert inst.get_nodeattr("degree") == degree
    assert inst.get_nodeattr("NumChannels") == num_channels
    assert inst.get_nodeattr("PE") == 1
    assert inst.get_nodeattr("inputDataType") == "FLOAT32"

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {"inp": x}
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    ref_mod = PWPolyFActivation(func, K=K, degree=degree)
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


@pytest.mark.fpgadataflow
def test_pwpolyf_specialize_rejects_non_versal():
    model = make_pwpolyf_modelwrapper("gelu", 3, 8, [1])

    with pytest.raises(Exception, match="Versal"):
        model.transform(SpecializeLayers(non_versal_fpga_part))


@pytest.mark.parametrize("func", ["gelu", "tanh"])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("K, bram18_per_coeff_rom", [(3, 1), (6, 2)])
@pytest.mark.fpgadataflow
def test_pwpolyf_resource_estimates(func, pe, degree, K, bram18_per_coeff_rom):
    num_channels = 8
    model = make_pwpolyf_modelwrapper(func, K, num_channels, [1])
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("degree", degree)

    assert inst.dsp_estimation() == degree * pe
    assert inst.lut_estimation() == 100 * degree * pe
    assert inst.bram_estimation() == max(degree - 1, 0) * pe * bram18_per_coeff_rom
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


# ---------- helpers for standard ONNX op inference tests ----------


def make_standard_activation_model(op_type, num_channels, num_input_vecs):
    """Build an ONNX model with a single standard activation op."""
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    act_node = helper.make_node(op_type, ["inp"], ["outp"], name=op_type + "_0")
    graph = helper.make_graph([act_node], "test_graph", [inp], [outp])
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 20
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def make_silu_pattern_model(num_channels, num_input_vecs):
    """Build ONNX model with Sigmoid + Mul pattern (SiLU)."""
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    sig_out = helper.make_tensor_value_info("sig_out", TensorProto.FLOAT, shape)

    sigmoid_node = helper.make_node("Sigmoid", ["inp"], ["sig_out"], name="Sigmoid_0")
    mul_node = helper.make_node("Mul", ["inp", "sig_out"], ["outp"], name="Mul_0")

    graph = helper.make_graph(
        [sigmoid_node, mul_node],
        "silu_graph",
        [inp],
        [outp],
    )
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    model.graph.value_info.append(sig_out)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def make_erf_gelu_model(num_channels, num_input_vecs):
    """Build ONNX model with the Erf-based GELU decomposition.

    Pattern: x * 0.5 * (1 + erf(x / sqrt(2)))
    Nodes: Div(x, sqrt(2)) -> Erf -> Add(_, 1) -> Mul(0.5, _) -> Mul(x, _)
    """
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    sqrt2 = helper.make_tensor("sqrt2", TensorProto.FLOAT, [], [np.float32(np.sqrt(2))])
    one = helper.make_tensor("one", TensorProto.FLOAT, [], [np.float32(1.0)])
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [np.float32(0.5)])

    div_node = helper.make_node("Div", ["inp", "sqrt2"], ["div_out"], name="Div_0")
    erf_node = helper.make_node("Erf", ["div_out"], ["erf_out"], name="Erf_0")
    add_node = helper.make_node("Add", ["erf_out", "one"], ["add_out"], name="Add_0")
    mul_half_node = helper.make_node("Mul", ["half", "add_out"], ["mul_half_out"], name="Mul_0")
    mul_x_node = helper.make_node("Mul", ["inp", "mul_half_out"], ["outp"], name="Mul_1")

    graph = helper.make_graph(
        [div_node, erf_node, add_node, mul_half_node, mul_x_node],
        "erf_gelu_graph",
        [inp],
        [outp],
        initializer=[sqrt2, one, half],
    )
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


# ---------- standard ONNX op inference tests ----------


@pytest.mark.parametrize(
    "op_type,expected_func",
    [
        ("Gelu", "gelu"),
        ("Sigmoid", "sigmoid"),
        ("Tanh", "tanh"),
    ],
)
@pytest.mark.parametrize("num_channels", [4, 16])
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_standard_op(op_type, expected_func, num_channels, num_input_vecs):
    model = make_standard_activation_model(op_type, num_channels, num_input_vecs)

    assert model.graph.node[0].op_type == op_type

    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == expected_func
    assert inst.get_nodeattr("K") == 3
    assert inst.get_nodeattr("NumChannels") == num_channels
    assert inst.get_nodeattr("PE") == 1
    assert inst.get_nodeattr("inputDataType") == "FLOAT32"


@pytest.mark.parametrize("num_channels", [4, 16])
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_silu_pattern(num_channels, num_input_vecs):
    model = make_silu_pattern_model(num_channels, num_input_vecs)

    assert len(model.graph.node) == 2
    assert model.graph.node[0].op_type == "Sigmoid"
    assert model.graph.node[1].op_type == "Mul"

    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == "silu"
    assert inst.get_nodeattr("K") == 3
    assert inst.get_nodeattr("NumChannels") == num_channels


@pytest.mark.fpgadataflow
def test_pwpolyf_infer_silu_reversed_mul_inputs():
    """SiLU detection works regardless of Mul input order."""
    num_channels = 8
    shape = [1, num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    sig_out = helper.make_tensor_value_info("sig_out", TensorProto.FLOAT, shape)

    sigmoid_node = helper.make_node("Sigmoid", ["inp"], ["sig_out"], name="Sigmoid_0")
    mul_node = helper.make_node("Mul", ["sig_out", "inp"], ["outp"], name="Mul_0")

    graph = helper.make_graph([sigmoid_node, mul_node], "silu_graph", [inp], [outp])
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    model.graph.value_info.append(sig_out)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("func") == "silu"


@pytest.mark.fpgadataflow
def test_pwpolyf_sigmoid_multi_consumer_no_silu():
    """Sigmoid with multiple consumers becomes standalone sigmoid, not silu."""
    num_channels = 8
    shape = [1, num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp1 = helper.make_tensor_value_info("outp1", TensorProto.FLOAT, shape)
    outp2 = helper.make_tensor_value_info("outp2", TensorProto.FLOAT, shape)
    sig_out = helper.make_tensor_value_info("sig_out", TensorProto.FLOAT, shape)

    sigmoid_node = helper.make_node("Sigmoid", ["inp"], ["sig_out"], name="Sigmoid_0")
    mul_node = helper.make_node("Mul", ["inp", "sig_out"], ["outp1"], name="Mul_0")
    identity_node = helper.make_node("Identity", ["sig_out"], ["outp2"], name="Id_0")

    graph = helper.make_graph(
        [sigmoid_node, mul_node, identity_node],
        "test_graph",
        [inp],
        [outp1, outp2],
    )
    model = helper.make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    model.graph.value_info.append(sig_out)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(InferPWPolyFLayer())

    pwp_nodes = [n for n in model.graph.node if n.op_type == "PWPolyF"]
    assert len(pwp_nodes) == 1
    inst = getCustomOp(pwp_nodes[0])
    assert inst.get_nodeattr("func") == "sigmoid"
    # Mul and Identity should remain
    assert any(n.op_type == "Mul" for n in model.graph.node)
    assert any(n.op_type == "Identity" for n in model.graph.node)


@pytest.mark.parametrize(
    "op_type,expected_func",
    [
        ("Gelu", "gelu"),
        ("Sigmoid", "sigmoid"),
        ("Tanh", "tanh"),
    ],
)
@pytest.mark.fpgadataflow
def test_pwpolyf_standard_op_execution(op_type, expected_func):
    num_channels = 16
    model = make_standard_activation_model(op_type, num_channels, [1])
    model = model.transform(InferPWPolyFLayer())

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]

    ref_mod = PWPolyFActivation(expected_func, K=3)
    with torch.no_grad():
        y_expected = ref_mod(torch.from_numpy(x)).numpy()
    assert np.allclose(y_produced, y_expected, atol=1e-6)


@pytest.mark.fpgadataflow
def test_pwpolyf_silu_pattern_execution():
    num_channels = 16
    model = make_silu_pattern_model(num_channels, [1])
    model = model.transform(InferPWPolyFLayer())

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]

    ref_mod = PWPolyFActivation("silu", K=3)
    with torch.no_grad():
        y_expected = ref_mod(torch.from_numpy(x)).numpy()
    assert np.allclose(y_produced, y_expected, atol=1e-6)


# ---------- Erf-based GELU inference tests ----------


@pytest.mark.parametrize("num_channels", [4, 16])
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_erf_gelu_pattern(num_channels, num_input_vecs):
    """Erf-based GELU decomposition (opset < 20) is converted to PWPolyF."""
    model = make_erf_gelu_model(num_channels, num_input_vecs)

    assert len(model.graph.node) == 5
    assert model.graph.node[1].op_type == "Erf"

    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == "gelu"
    assert inst.get_nodeattr("K") == 3
    assert inst.get_nodeattr("NumChannels") == num_channels
    assert inst.get_nodeattr("PE") == 1
    assert inst.get_nodeattr("inputDataType") == "FLOAT32"


@pytest.mark.fpgadataflow
def test_pwpolyf_erf_gelu_execution():
    """Erf-based GELU produces same output as PWPolyFActivation."""
    num_channels = 16
    model = make_erf_gelu_model(num_channels, [1])
    model = model.transform(InferPWPolyFLayer())

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]

    ref_mod = PWPolyFActivation("gelu", K=3)
    with torch.no_grad():
        y_expected = ref_mod(torch.from_numpy(x)).numpy()
    assert np.allclose(y_produced, y_expected, atol=1e-6)


# ---------- coefficient package smoketests ----------


@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.fpgadataflow
def test_pwpolyf_generate_coeffs_pkg(K):
    """Verify PWPolyF_rtl coefficient generation produces valid SystemVerilog."""
    pkg = make_pwpolyf_rtl_inst(K=K)._generate_coeffs_pkg()

    assert "package pwpolyf_pkg" in pkg
    assert "endpackage" in pkg
    # localparam lines use padded alignment in the generated SV
    assert "DEGREE      = 2;" in pkg
    assert "K           = %d;" % K in pkg

    num_segs = 1 + 2 * 5 * (1 << K)
    assert "NUM_SEGS    = %d;" % num_segs in pkg

    for func_label in ["GELU", "SILU", "SIGMOID", "TANH"]:
        assert func_label + " = '{" in pkg

    seg_lines = [line for line in pkg.split("\n") if "// seg" in line]
    # Each function has num_segs segments, 4 functions total
    assert len(seg_lines) == 4 * num_segs


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.fpgadataflow
def test_pwpolyf_generate_coeffs_pkg_degree(degree):
    """Verify PWPolyF_rtl coefficient generation respects degree parameter."""
    K = 3
    pkg = make_pwpolyf_rtl_inst(K=K, degree=degree)._generate_coeffs_pkg()

    assert "DEGREE      = %d;" % degree in pkg
    # Each segment line should have degree+1 coefficient values
    seg_lines = [line for line in pkg.split("\n") if "// seg 0" in line]
    for line in seg_lines:
        hex_vals = [s for s in line.split() if s.startswith("32'h")]
        assert len(hex_vals) == degree + 1


# ---------- generate_hdl smoketests ----------


@pytest.mark.parametrize("func", ["gelu", "tanh"])
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_pwpolyf_generate_hdl(func, pe):
    """Verify generate_hdl produces expected RTL files."""
    num_channels = 4
    model = make_pwpolyf_modelwrapper(func, 3, num_channels, [1])
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))

    # Re-fetch node after transform (PrepareIP returns a new model)
    node = model.graph.node[0]
    inst = getCustomOp(node)

    code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
    assert code_gen_dir, "code_gen_dir_ipgen not set after PrepareIP"
    assert os.path.isfile(os.path.join(code_gen_dir, "pwpolyf_pkg.sv"))
    assert os.path.isfile(os.path.join(code_gen_dir, "pwpolyf.sv"))
    assert os.path.isfile(os.path.join(code_gen_dir, "queue.sv"))

    topname = inst.get_nodeattr("gen_top_module")
    assert os.path.isfile(os.path.join(code_gen_dir, topname + ".v"))

    # Verify package content
    with open(os.path.join(code_gen_dir, "pwpolyf_pkg.sv"), "r") as f:
        pkg_content = f.read()
    assert "DEGREE      = 2;" in pkg_content
    assert "K           = 3;" in pkg_content
    assert func.upper() + " = '{" in pkg_content


# ---------- RTL simulation tests ----------


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.parametrize("num_channels", [4, 8])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pwpolyf_rtlsim(func, num_channels, pe):
    """Node-by-node RTL simulation of PWPolyF_rtl."""
    if num_channels % pe != 0:
        pytest.skip("PE does not divide NumChannels")

    K = 3
    model = make_pwpolyf_modelwrapper(func, K, num_channels, [1])

    # Get cppsim reference output
    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {"inp": x}
    y_ref = oxe.execute_onnx(model, input_dict)["outp"]

    # Specialize to RTL and set PE
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    # RTL simulation pipeline
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_rtl = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.allclose(y_ref, y_rtl, atol=1e-4), "RTL output does not match cppsim reference"

    # Verify cycle count (re-fetch node after transforms)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pwpolyf_rtlsim_stitched_ip(func, pe):
    """Stitched IP RTL simulation of PWPolyF_rtl."""
    K = 3
    num_channels = 4
    model = make_pwpolyf_modelwrapper(func, K, num_channels, [1])

    # Get cppsim reference output
    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {model.graph.input[0].name: x}
    y_ref = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]

    # Specialize to RTL and set PE
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    # Stitched IP pipeline
    model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")

    input_dict = {model.graph.input[0].name: x}
    y_rtl = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert np.allclose(
        y_ref, y_rtl, atol=1e-4
    ), "Stitched IP output does not match cppsim reference"

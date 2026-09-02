# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import os
import re
import tempfile
import torch
from onnx import TensorProto, helper, load
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.custom_op.general.pwpolyfunction import EXP_CLAMP, NUM_OCTAVES
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPWPolyFLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import get_finn_root
from finn.util.torch_hw_modules import PWPolyFActivation

TEST_FPGA_PART = "xcvc1902-vsva2197-2MP-e-S"
NON_VERSAL_FPGA_PART = "xczu3eg-sbva484-1-e"
TARGET_CLK_NS = 5

ACTIVATION_PATTERNS = {
    "Gelu": (["Gelu"], "gelu"),
    "Sigmoid": (["Sigmoid"], "sigmoid"),
    "Tanh": (["Tanh"], "tanh"),
    "silu": (["Sigmoid", "Mul"], "silu"),
    "erf_gelu": (["Div", "Erf", "Add", "Mul", "Mul"], "gelu"),
}


def make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, num_input_vecs + [num_channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [num_channels])

    pwpolyf_node = helper.make_node(
        "PWPolyF",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        func=func,
        K=k,
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
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pwpolyf-test"))
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    return model


def make_standard_activation_modelwrapper(op_type, num_channels, num_input_vecs):
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    act_node = helper.make_node(op_type, ["inp"], ["outp"], name=op_type + "_0")
    graph = helper.make_graph([act_node], "test_graph", [inp], [outp])
    model = qonnx_make_model(graph, producer_name="pwpolyf-test")
    model.opset_import[0].version = 20
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    return model


def make_silu_pattern_modelwrapper(
    num_channels, num_input_vecs, reverse_mul_inputs=False, extra_consumer=False
):
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    sig_out = helper.make_tensor_value_info("sig_out", TensorProto.FLOAT, shape)

    sigmoid_node = helper.make_node("Sigmoid", ["inp"], ["sig_out"], name="Sigmoid_0")
    mul_inputs = ["sig_out", "inp"] if reverse_mul_inputs else ["inp", "sig_out"]
    mul_node = helper.make_node("Mul", mul_inputs, ["outp"], name="Mul_0")
    nodes = [sigmoid_node, mul_node]
    outputs = [outp]
    if extra_consumer:
        outp2 = helper.make_tensor_value_info("outp2", TensorProto.FLOAT, shape)
        identity_node = helper.make_node("Identity", ["sig_out"], ["outp2"], name="Id_0")
        nodes.append(identity_node)
        outputs.append(outp2)

    graph = helper.make_graph(nodes, "silu_graph", [inp], outputs)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pwpolyf-test"))
    model.graph.value_info.append(sig_out)
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    model.set_tensor_datatype("sig_out", DataType["FLOAT32"])
    if extra_consumer:
        model.set_tensor_datatype("outp2", DataType["FLOAT32"])
    return model


def make_erf_gelu_modelwrapper(num_channels, num_input_vecs):
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
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pwpolyf-test"))
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    return model


def make_activation_modelwrapper(pattern, num_channels, num_input_vecs):
    if pattern in ["Gelu", "Sigmoid", "Tanh"]:
        return make_standard_activation_modelwrapper(pattern, num_channels, num_input_vecs)
    if pattern == "silu":
        return make_silu_pattern_modelwrapper(num_channels, num_input_vecs)
    if pattern == "erf_gelu":
        return make_erf_gelu_modelwrapper(num_channels, num_input_vecs)
    raise ValueError("Unknown activation pattern %s" % pattern)


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


def execute_pwpolyf_reference(func, input_tensor, k=3, degree=2):
    ref_mod = PWPolyFActivation(func, K=k, degree=degree)
    with torch.no_grad():
        return ref_mod(torch.from_numpy(input_tensor)).numpy()


def export_pwpolyf_model(func, k, degree, num_channels):
    mod = PWPolyFActivation(func, K=k, degree=degree).eval()
    dummy = torch.randn(1, num_channels)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name
    try:
        torch.onnx.export(
            mod,
            dummy,
            export_path,
            input_names=["inp"],
            output_names=["outp"],
            opset_version=13,
            dynamo=False,
        )
        return load(export_path)
    finally:
        os.unlink(export_path)


def make_pwpolyf_rtl_inst(k=3, degree=2):
    model = make_pwpolyf_modelwrapper("gelu", k, 4, [1])
    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("degree", degree)
    return inst


# activation function
@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
# channels
@pytest.mark.parametrize("num_channels", [4, 16])
# input vector shape
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
# folding
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.fpgadataflow
def test_fpgadataflow_pwpolyf_cppsim(func, num_channels, num_input_vecs, fold):
    k = 3
    if fold == -1:
        pe = 1
    else:
        pe = max(1, num_channels // fold)
    assert num_channels % pe == 0

    model = make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    input_shape = tuple(num_input_vecs + [num_channels])
    x = np.random.uniform(-10, 10, input_shape).astype(np.float32)
    input_dict = prepare_inputs(x)
    y_expected = execute_pwpolyf_reference(func, x, k=k)

    # golden reference before specializing
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert y_produced.shape == y_expected.shape
    assert np.allclose(y_produced, y_expected, atol=1e-6)

    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    # PWPolyF_rtl cppsim delegates to the Python base op, so no C++ compilation is needed.
    model = model.transform(SetExecMode("cppsim"))
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.allclose(y_produced, y_expected, atol=1e-6), "cppsim failed"


# activation function
@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.fpgadataflow
def test_pwpolyf_onnx_export(func):
    k = 3
    degree = 3
    num_channels = 32
    onnx_model = export_pwpolyf_model(func, k, degree, num_channels)

    pwp_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyFunction"]
    assert len(pwp_nodes) == 1
    node = pwp_nodes[0]
    assert len(node.input) == 1
    func_attr = {a.name: a for a in node.attribute}
    assert func_attr["func"].s.decode("utf-8") == func
    assert func_attr["K"].i == k
    assert func_attr["degree"].i == degree


# activation function
@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_transform(func):
    k = 3
    degree = 3
    num_channels = 16
    model = ModelWrapper(export_pwpolyf_model(func, k, degree, num_channels))

    node = model.graph.node[0]
    assert node.op_type == "PWPolyFunction"

    model = model.transform(InferPWPolyFLayer())

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == func
    assert inst.get_nodeattr("K") == k
    assert inst.get_nodeattr("degree") == degree
    assert inst.get_nodeattr("NumChannels") == num_channels
    assert inst.get_nodeattr("PE") == 1
    assert inst.get_nodeattr("inputDataType") == "FLOAT32"

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = prepare_inputs(x)
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    y_expected = execute_pwpolyf_reference(func, x, k=k, degree=degree)
    assert np.allclose(y_produced, y_expected, atol=1e-6)


# activation function
@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.fpgadataflow
def test_pwpolyf_specialize_rtl(func):
    k = 3
    num_channels = 8
    model = make_pwpolyf_modelwrapper(func, k, num_channels, [1])
    model = model.transform(SpecializeLayers(TEST_FPGA_PART))

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    assert node.domain == "finn.custom_op.fpgadataflow.rtl"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == func
    assert inst.get_nodeattr("K") == k


@pytest.mark.fpgadataflow
def test_pwpolyf_specialize_rejects_non_versal():
    model = make_pwpolyf_modelwrapper("gelu", 3, 8, [1])

    with pytest.raises(Exception, match="Versal"):
        model.transform(SpecializeLayers(NON_VERSAL_FPGA_PART))


# activation function
@pytest.mark.parametrize("func", ["gelu", "tanh"])
# processing elements
@pytest.mark.parametrize("pe", [1, 2, 4])
# polynomial degree
@pytest.mark.parametrize("degree", [1, 2, 3])
# mantissa subdivision bits
@pytest.mark.parametrize("k", [3, 6])
@pytest.mark.fpgadataflow
def test_pwpolyf_resource_estimates(func, pe, degree, k):
    num_channels = 8
    model = make_pwpolyf_modelwrapper(func, k, num_channels, [1])
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("degree", degree)
    bram18_per_coeff_rom = 1 if k == 3 else 2

    assert inst.dsp_estimation() == degree * pe
    assert inst.lut_estimation() == 100 * degree * pe
    assert inst.bram_estimation() == max(degree - 1, 0) * pe * bram18_per_coeff_rom
    assert inst.uram_estimation() == 0


# activation function
@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
# processing elements
@pytest.mark.parametrize("pe", [1, 4])
@pytest.mark.fpgadataflow
def test_pwpolyf_folded_shape(func, pe):
    k = 3
    num_channels = 12
    num_input_vecs = [1, 3, 3]
    model = make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    assert inst.get_normal_input_shape() == (1, 3, 3, 12)
    assert inst.get_normal_output_shape() == (1, 3, 3, 12)
    expected_folded_shape = (1, 3, 3, num_channels // pe, pe)
    assert inst.get_folded_input_shape() == expected_folded_shape
    assert inst.get_folded_output_shape() == expected_folded_shape
    assert inst.get_instream_width() == pe * 32
    assert inst.get_outstream_width() == pe * 32


# activation function
@pytest.mark.parametrize("func", ["gelu", "silu"])
@pytest.mark.fpgadataflow
def test_pwpolyf_exp_cycles(func):
    k = 3
    num_channels = 8
    pe = 2
    num_input_vecs = [1, 4, 4]
    model = make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)

    exp = inst.get_exp_cycles()
    assert exp == 1 * 4 * 4 * (num_channels // pe)

    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    exp_dict = model.analysis(exp_cycles_per_layer)
    assert node.name in exp_dict
    assert exp_dict[node.name] == exp


# activation pattern
@pytest.mark.parametrize("pattern", list(ACTIVATION_PATTERNS))
# channels
@pytest.mark.parametrize("num_channels", [4, 16])
# input vector shape
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.fpgadataflow
def test_pwpolyf_infer_activation_pattern(pattern, num_channels, num_input_vecs):
    expected_nodes, expected_func = ACTIVATION_PATTERNS[pattern]
    model = make_activation_modelwrapper(pattern, num_channels, num_input_vecs)

    assert [node.op_type for node in model.graph.node] == expected_nodes

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


@pytest.mark.fpgadataflow
def test_pwpolyf_infer_silu_reversed_mul_inputs():
    num_channels = 8
    model = make_silu_pattern_modelwrapper(num_channels, [1], reverse_mul_inputs=True)
    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("func") == "silu"


@pytest.mark.fpgadataflow
def test_pwpolyf_sigmoid_multi_consumer_no_silu():
    num_channels = 8
    model = make_silu_pattern_modelwrapper(num_channels, [1], extra_consumer=True)
    model = model.transform(InferPWPolyFLayer())

    pwp_nodes = [n for n in model.graph.node if n.op_type == "PWPolyF"]
    assert len(pwp_nodes) == 1
    inst = getCustomOp(pwp_nodes[0])
    assert inst.get_nodeattr("func") == "sigmoid"
    assert any(n.op_type == "Mul" for n in model.graph.node)
    assert any(n.op_type == "Identity" for n in model.graph.node)


# activation pattern
@pytest.mark.parametrize("pattern", list(ACTIVATION_PATTERNS))
@pytest.mark.fpgadataflow
def test_pwpolyf_activation_pattern_execution(pattern):
    num_channels = 16
    expected_func = ACTIVATION_PATTERNS[pattern][1]
    model = make_activation_modelwrapper(pattern, num_channels, [1])
    model = model.transform(InferPWPolyFLayer())

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    y_produced = oxe.execute_onnx(model, prepare_inputs(x))["outp"]
    y_expected = execute_pwpolyf_reference(expected_func, x)
    assert np.allclose(y_produced, y_expected, atol=1e-6)


# mantissa subdivision bits
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.fpgadataflow
def test_pwpolyf_generate_coeffs_pkg(k):
    pkg = make_pwpolyf_rtl_inst(k=k)._generate_coeffs_pkg()

    assert "package pwpolyf_pkg" in pkg
    assert "endpackage" in pkg
    assert "DEGREE      = 2;" in pkg
    assert "K           = %d;" % k in pkg

    num_segs = 1 + 2 * 5 * (1 << k)
    assert "NUM_SEGS    = %d;" % num_segs in pkg
    assert all(label + " = '{" in pkg for label in ["GELU", "SILU", "SIGMOID", "TANH"])

    seg_lines = [line for line in pkg.split("\n") if "// seg" in line]
    assert len(seg_lines) == 4 * num_segs


# polynomial degree
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.fpgadataflow
def test_pwpolyf_generate_coeffs_pkg_degree(degree):
    k = 3
    pkg = make_pwpolyf_rtl_inst(k=k, degree=degree)._generate_coeffs_pkg()

    assert "DEGREE      = %d;" % degree in pkg
    seg_lines = [line for line in pkg.split("\n") if "// seg 0" in line]
    coeff_counts = [
        len([value for value in line.split() if value.startswith("32'h")]) for line in seg_lines
    ]
    assert all(count == degree + 1 for count in coeff_counts)


# activation function
@pytest.mark.parametrize("func", ["gelu", "tanh"])
# folding
@pytest.mark.parametrize("fold", [-1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_pwpolyf_generate_hdl(func, fold):
    num_channels = 4
    pe = 1 if fold == -1 else max(1, num_channels // fold)
    assert num_channels % pe == 0

    model = make_pwpolyf_modelwrapper(func, 3, num_channels, [1])
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("PE", pe)
    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    model = model.transform(PrepareIP(TEST_FPGA_PART, TARGET_CLK_NS))

    node = model.graph.node[0]
    inst = getCustomOp(node)

    code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
    assert code_gen_dir, "code_gen_dir_ipgen not set after PrepareIP"
    assert os.path.isfile(os.path.join(code_gen_dir, "pwpolyf_pkg.sv"))
    assert os.path.isfile(os.path.join(code_gen_dir, "pwpolyf.sv"))

    topname = inst.get_nodeattr("gen_top_module")
    assert os.path.isfile(os.path.join(code_gen_dir, topname + ".v"))

    with open(os.path.join(code_gen_dir, "pwpolyf_pkg.sv"), "r") as f:
        pkg_content = f.read()
    assert "DEGREE      = 2;" in pkg_content
    assert "K           = 3;" in pkg_content
    assert func.upper() + " = '{" in pkg_content


# The RTL matrix is intentionally smaller than the cppsim matrix to limit Vivado builds.
# activation function
@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
# channels
@pytest.mark.parametrize("num_channels", [4, 8])
# folding
@pytest.mark.parametrize("fold", [-1, 1, 2, 4])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_pwpolyf_rtlsim(func, num_channels, fold):
    pe = 1 if fold == -1 else max(1, num_channels // fold)
    assert num_channels % pe == 0

    k = 3
    model = make_pwpolyf_modelwrapper(func, k, num_channels, [1])

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = prepare_inputs(x)
    y_ref = oxe.execute_onnx(model, input_dict)["outp"]

    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("PE", pe)
    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "PWPolyF_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(TEST_FPGA_PART, TARGET_CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_rtl = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.allclose(y_ref, y_rtl, atol=1e-4), "RTL output does not match cppsim reference"

    node = model.graph.node[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


# activation function
@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
# folding
@pytest.mark.parametrize("fold", [-1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pwpolyf_rtlsim_stitched_ip(func, fold):
    k = 3
    num_channels = 4
    pe = 1 if fold == -1 else max(1, num_channels // fold)
    assert num_channels % pe == 0

    model = make_pwpolyf_modelwrapper(func, k, num_channels, [1])

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = prepare_inputs(x)
    y_ref = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]

    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("PE", pe)
    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(InsertAndSetFIFODepths(TEST_FPGA_PART, TARGET_CLK_NS))
    model = model.transform(PrepareIP(TEST_FPGA_PART, TARGET_CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(TEST_FPGA_PART, TARGET_CLK_NS))
    model.set_metadata_prop("exec_mode", "rtlsim")

    input_dict_stitched = {model.get_first_global_in(): x}
    y_rtl = oxe.execute_onnx(model, input_dict_stitched)[model.graph.output[0].name]
    assert np.allclose(
        y_ref, y_rtl, atol=1e-4
    ), "Stitched IP output does not match cppsim reference"


@pytest.mark.fpgadataflow
def test_pwpolyf_rtl_constants_match():
    """Verify that constants hardcoded in pwpolyf.sv match the Python definitions."""
    rtl_path = os.path.join(get_finn_root(), "finn-rtllib", "pwpolyf", "hdl", "pwpolyf.sv")

    with open(rtl_path, "r") as f:
        rtl_content = f.read()

    # Check EXP_CLAMP matches
    match = re.search(r"localparam\s+int\s+unsigned\s+EXP_CLAMP\s*=\s*(\d+)", rtl_content)
    assert match is not None, "EXP_CLAMP not found in pwpolyf.sv"
    rtl_exp_clamp = int(match.group(1))
    assert (
        rtl_exp_clamp == EXP_CLAMP
    ), f"EXP_CLAMP mismatch: RTL has {rtl_exp_clamp}, Python has {EXP_CLAMP}"

    # Check NUM_OCTAVES is used consistently (it comes from the generated package,
    # but we verify the Python constant matches what we'd generate)
    assert NUM_OCTAVES == 5, f"NUM_OCTAVES should be 5, got {NUM_OCTAVES}"

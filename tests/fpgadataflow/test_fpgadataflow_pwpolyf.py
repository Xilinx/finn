# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import os
import re
import tempfile
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
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


def make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs, pe=1, degree=2):
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
        degree=degree,
        NumChannels=num_channels,
        PE=pe,
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


def make_silu_pattern_modelwrapper(num_channels, num_input_vecs, reverse_mul_inputs=False):
    shape = num_input_vecs + [num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    sig_out = helper.make_tensor_value_info("sig_out", TensorProto.FLOAT, shape)

    sigmoid_node = helper.make_node("Sigmoid", ["inp"], ["sig_out"], name="Sigmoid_0")
    mul_inputs = ["sig_out", "inp"] if reverse_mul_inputs else ["inp", "sig_out"]
    mul_node = helper.make_node("Mul", mul_inputs, ["outp"], name="Mul_0")
    nodes = [sigmoid_node, mul_node]

    graph = helper.make_graph(nodes, "silu_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pwpolyf-test"))
    model.graph.value_info.append(sig_out)
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    model.set_tensor_datatype("sig_out", DataType["FLOAT32"])
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


def execute_pwpolyf_reference(func, input_tensor, k=3, degree=2):
    ref_mod = PWPolyFActivation(func, K=k, degree=degree)
    with torch.no_grad():
        return ref_mod(torch.from_numpy(input_tensor)).numpy()


# Main test covering cppsim and rtlsim execution
@pytest.mark.parametrize("func", ["gelu", "silu", "sigmoid", "tanh"])
@pytest.mark.parametrize("num_channels", [4, 8])
@pytest.mark.parametrize("num_input_vecs", [[1], [1, 2, 2]])
@pytest.mark.parametrize("fold", [-1, 2])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_pwpolyf(func, num_channels, num_input_vecs, fold, exec_mode):
    k = 3
    pe = 1 if fold == -1 else max(1, num_channels // fold)
    if num_channels % pe != 0:
        pytest.skip("num_channels % pe != 0, skipping")

    model = make_pwpolyf_modelwrapper(func, k, num_channels, num_input_vecs, pe=pe)

    input_shape = tuple(num_input_vecs + [num_channels])
    x = np.random.uniform(-10, 10, input_shape).astype(np.float32)
    input_dict = {model.get_first_global_in(): x}
    y_expected = execute_pwpolyf_reference(func, x, k=k)

    # Golden reference before specializing
    y_produced = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    assert y_produced.shape == y_expected.shape
    assert np.allclose(y_produced, y_expected, atol=1e-6), "HW layer execution failed"

    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF_rtl"
    assert node.domain == "finn.custom_op.fpgadataflow.rtl"

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "cppsim":
        pass  # PWPolyF_rtl cppsim delegates to the Python base op
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(TEST_FPGA_PART, TARGET_CLK_NS))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_produced = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    assert np.allclose(y_produced, y_expected, atol=1e-4), f"{exec_mode} failed"

    if exec_mode == "rtlsim":
        node = model.graph.node[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


@pytest.mark.parametrize("func", ["gelu", "sigmoid"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_pwpolyf_stitched_ip(func):
    k = 3
    num_channels = 4
    pe = 2

    model = make_pwpolyf_modelwrapper(func, k, num_channels, [1], pe=pe)

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {model.get_first_global_in(): x}
    y_ref = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]

    model = model.transform(SpecializeLayers(TEST_FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InsertAndSetFIFODepths(TEST_FPGA_PART, TARGET_CLK_NS))
    model = model.transform(PrepareIP(TEST_FPGA_PART, TARGET_CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(TEST_FPGA_PART, TARGET_CLK_NS))
    model.set_metadata_prop("exec_mode", "rtlsim")

    input_dict = {model.get_first_global_in(): x}
    y_rtl = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    assert np.allclose(y_ref, y_rtl, atol=1e-4), "Stitched IP output mismatch"


# InferPWPolyFLayer transformation for all input patterns
# For export patterns, degree is configurable; for ONNX patterns, InferPWPolyFLayer uses degree=2
@pytest.mark.parametrize(
    "pattern,expected_func,degree",
    [
        # Export patterns: test multiple degrees since they're preserved from the export
        pytest.param("export", "gelu", 2, id="export-gelu-deg2"),
        pytest.param("export", "gelu", 3, id="export-gelu-deg3"),
        pytest.param("export", "silu", 2, id="export-silu-deg2"),
        pytest.param("export", "silu", 3, id="export-silu-deg3"),
        pytest.param("export", "sigmoid", 2, id="export-sigmoid-deg2"),
        pytest.param("export", "sigmoid", 3, id="export-sigmoid-deg3"),
        pytest.param("export", "tanh", 2, id="export-tanh-deg2"),
        pytest.param("export", "tanh", 3, id="export-tanh-deg3"),
        # ONNX patterns: InferPWPolyFLayer hardcodes degree=2
        pytest.param("Gelu", "gelu", 2, id="onnx-gelu"),
        pytest.param("Sigmoid", "sigmoid", 2, id="onnx-sigmoid"),
        pytest.param("Tanh", "tanh", 2, id="onnx-tanh"),
        pytest.param("silu", "silu", 2, id="onnx-silu"),
        pytest.param("silu_reversed", "silu", 2, id="onnx-silu-reversed"),
        pytest.param("erf_gelu", "gelu", 2, id="onnx-erf-gelu"),
    ],
)
@pytest.mark.fpgadataflow
def test_fpgadataflow_pwpolyf_infer(pattern, expected_func, degree):
    num_channels = 16

    if pattern == "export":
        mod = PWPolyFActivation(expected_func, K=3, degree=degree).eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            export_qonnx(mod, torch.randn(1, num_channels), f.name)
            model = ModelWrapper(f.name)
        assert model.graph.node[0].op_type == "PWPolyFunction"
    elif pattern in ["Gelu", "Sigmoid", "Tanh"]:
        model = make_standard_activation_modelwrapper(pattern, num_channels, [1])
    elif pattern == "silu":
        model = make_silu_pattern_modelwrapper(num_channels, [1])
    elif pattern == "silu_reversed":
        model = make_silu_pattern_modelwrapper(num_channels, [1], reverse_mul_inputs=True)
    elif pattern == "erf_gelu":
        model = make_erf_gelu_modelwrapper(num_channels, [1])
    else:
        raise ValueError(f"Unknown pattern {pattern}")

    model = model.transform(InferPWPolyFLayer())

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("func") == expected_func
    assert inst.get_nodeattr("NumChannels") == num_channels

    x = np.random.uniform(-5, 5, (1, num_channels)).astype(np.float32)
    input_dict = {model.get_first_global_in(): x}
    y_produced = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    y_expected = execute_pwpolyf_reference(expected_func, x, degree=degree)
    assert np.allclose(y_produced, y_expected, atol=1e-6)


@pytest.mark.fpgadataflow
def test_fpgadataflow_pwpolyf_specialize_rejects_non_versal():
    """PWPolyF requires Versal DSPFP32 primitive."""
    model = make_pwpolyf_modelwrapper("gelu", 3, 8, [1])
    with pytest.raises(Exception, match="Versal"):
        model.transform(SpecializeLayers(NON_VERSAL_FPGA_PART))


@pytest.mark.fpgadataflow
def test_fpgadataflow_pwpolyf_rtl_constants_match():
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

    # Check NUM_OCTAVES is used consistently
    assert NUM_OCTAVES == 5, f"NUM_OCTAVES should be 5, got {NUM_OCTAVES}"

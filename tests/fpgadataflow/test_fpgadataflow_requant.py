# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test cases for InferRequantLayer transformation which converts MultiThreshold
or Quant nodes to Requant nodes.

The requant operation computes output as:
    clip(round(x * scale + bias), min, max)
instead of comparing against multiple thresholds.
"""

import pytest

import numpy as np
import onnx.parser as oprs
import os
import torch
from brevitas.core.scaling import ScalingImplType
from brevitas.export import export_qonnx
from brevitas.nn import QuantReLU
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferRequantLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir, pynq_part_map

test_pynq_board = "ZCU104"
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("max_val", [1.0, 6.0])
@pytest.mark.parametrize("ishape", [(1, 16), (1, 32, 1, 1)])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("input_dtype", ["INT8", "FLOAT32"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("pe", [1, 16, 32])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_infer_requant_layer(abits, max_val, ishape, per_channel, input_dtype, exec_mode, pe):
    """Test InferRequantLayer converts MultiThreshold to Requant."""

    num_channels = ishape[1]

    # Skip if PE doesn't divide evenly into num_channels
    if num_channels % pe != 0:
        pytest.skip(f"PE={pe} does not divide num_channels={num_channels}")

    if per_channel:
        b_act = QuantReLU(
            bit_width=abits,
            max_val=max_val,
            scaling_impl_type=ScalingImplType.CONST,
            scaling_per_output_channel=True,
            per_channel_broadcastable_shape=(1, num_channels) + (1,) * (len(ishape) - 2),
        )
    else:
        b_act = QuantReLU(
            bit_width=abits,
            max_val=max_val,
            scaling_impl_type=ScalingImplType.CONST,
        )

    # Export to QONNX
    build_dir = make_build_dir(prefix="test_infer_requant_layer_")
    m_path = os.path.join(build_dir, "model.onnx")
    export_qonnx(b_act, torch.randn(ishape), m_path)
    qonnx_cleanup(m_path, out_file=m_path)

    # Convert to FINN format (creates MultiThreshold)
    model = ModelWrapper(m_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())

    # Set input datatype
    # Must re-run InferDataTypes to propagate to intermediate tensors
    model.set_tensor_datatype(model.graph.input[0].name, DataType[input_dtype])
    model = model.transform(InferDataTypes())

    # Get golden output before conversion
    inp = gen_finn_dt_tensor(DataType[input_dtype], ishape)
    input_dict = {model.graph.input[0].name: inp}
    y_golden = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]

    # Apply InferRequantLayer and convert any remaining Mul nodes to HW
    model = model.transform(InferRequantLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Verify MultiThreshold was converted to Requant
    mt_nodes = model.get_nodes_by_op_type("MultiThreshold")
    requant_nodes = model.get_nodes_by_op_type("Requant")
    assert len(mt_nodes) == 0, "MultiThreshold should be converted"
    assert len(requant_nodes) == 1, "Expected one Requant node"

    # Verify Requant attributes
    requant_node = requant_nodes[0]
    requant_inst = getCustomOp(requant_node)
    assert requant_inst.get_nodeattr("NumChannels") == num_channels
    # Output datatype should be unsigned (ReLU output)
    odt = requant_inst.get_output_datatype()
    assert odt.bitwidth() == abits
    assert not odt.signed(), "ReLU output should be unsigned"

    # Verify scale and bias are set as initializers
    scale = model.get_initializer(requant_node.input[1])
    bias = model.get_initializer(requant_node.input[2])
    assert scale is not None, "Scale should be set as initializer"
    assert bias is not None, "Bias should be set as initializer"
    # Scale/bias shape depends on threshold shape from Brevitas, not per_channel flag
    # With CONST scaling and scalar max_val, thresholds are always (1, N) and broadcast
    assert scale.size >= 1, f"Scale should have at least 1 element: {scale.shape}"
    assert bias.size >= 1, f"Bias should have at least 1 element: {bias.shape}"

    # Verify functional correctness
    # Allow tolerance of 1 quantization step due to floating point precision at boundaries
    # The quantization step is max_val / (2^abits - 1)
    quant_step = max_val / (2**abits - 1)
    y_requant = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert np.allclose(y_golden, y_requant, atol=quant_step), (
        f"Output mismatch: max diff = {np.max(np.abs(y_golden - y_requant))}, "
        f"quant_step = {quant_step}"
    )

    # Specialize layers based on input type
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    # Verify correct backend was selected and set PE
    if input_dtype == "INT8":
        requant_rtl_nodes = model.get_nodes_by_op_type("Requant_rtl")
        assert len(requant_rtl_nodes) == 1, "Expected one Requant_rtl node"
        getCustomOp(requant_rtl_nodes[0]).set_nodeattr("PE", pe)
    else:
        requant_hls_nodes = model.get_nodes_by_op_type("Requant_hls")
        assert len(requant_hls_nodes) == 1, "Expected one Requant_hls node"
        getCustomOp(requant_hls_nodes[0]).set_nodeattr("PE", pe)

    # Prepare and run simulation
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        y_sim = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
        assert np.allclose(
            y_golden, y_sim, atol=quant_step
        ), f"cppsim mismatch: max diff = {np.max(np.abs(y_golden - y_sim))}"
    else:
        # Node-by-node rtlsim
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

        y_sim = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
        assert np.allclose(
            y_golden, y_sim, atol=quant_step
        ), f"rtlsim node-by-node mismatch: max diff = {np.max(np.abs(y_golden - y_sim))}"

        # Verify cycle estimation
        op_type = "Requant_rtl" if input_dtype == "INT8" else "Requant_hls"
        node = model.get_nodes_by_op_type(op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0

        # Stitched IP rtlsim - only run for one config per input type to save time
        # INT8: abits=4, max_val=1.0, ishape=(1,16), per_channel=False, pe=16
        # FLOAT32: abits=4, max_val=1.0, ishape=(1,16), per_channel=False, pe=16
        run_stitched = (
            abits == 4 and max_val == 1.0 and ishape == (1, 16) and not per_channel and pe == 16
        )
        if run_stitched:
            model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

            model.set_metadata_prop("exec_mode", "rtlsim")
            y_stitched = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
            assert np.allclose(
                y_golden, y_stitched, atol=quant_step
            ), f"rtlsim stitched mismatch: max diff = {np.max(np.abs(y_golden - y_stitched))}"


def make_quant_test_model(
    ishp, channelwise, bitwidth, need_extraction_scale, need_extraction_zeropt
):
    """Create a test model with a Quant node."""
    ishp_str = str(list(ishp))
    if channelwise:
        q_attr_shp = ishp
    else:
        q_attr_shp = (1,)
    attrshp_str = str(list(q_attr_shp))
    np.random.seed(0)
    if need_extraction_scale:
        scale = np.random.rand(*q_attr_shp).astype(np.float32)
    else:
        scale = np.ones(q_attr_shp, dtype=np.float32)
    if need_extraction_zeropt:
        zeropt = np.random.rand(*q_attr_shp).astype(np.float32)
    else:
        zeropt = np.zeros(q_attr_shp, dtype=np.float32)
    signed = 0  # unsigned output for RTL backend compatibility
    narrow = 0  # full range for RTL backend compatibility
    rounding_mode = "ROUND"

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (float{ishp_str} out0)
    <
        float{attrshp_str} scale_param,
        float{attrshp_str} zeropt_param,
        float bitwidth_param
    >
    {{
        out0 = qonnx.custom_op.general.Quant<
            signed={str(signed)},
            narrow={str(narrow)},
            rounding_mode="{rounding_mode}"
        >(in0, scale_param, zeropt_param, bitwidth_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_initializer("scale_param", scale)
    model.set_initializer("zeropt_param", zeropt)
    model.set_initializer("bitwidth_param", bitwidth)
    return model


@pytest.mark.parametrize("channelwise", [True, False])
@pytest.mark.parametrize("pe", [1, 5, 10])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_infer_requant_from_quant(channelwise, pe):
    """Test InferRequantLayer converts Quant node to Requant.

    This test is adapted from test_fpgadataflow_float2int but uses
    InferRequantLayer instead of InferQuantAsFloat2Int.
    """
    ishp = (1, 10)
    bitwidth = np.asarray(4.0, dtype=np.float32)
    model = make_quant_test_model(
        ishp, channelwise, bitwidth, need_extraction_scale=True, need_extraction_zeropt=True
    )
    ishp = model.get_tensor_shape("in0")
    # Use rand (uniform [0,1)) like original Float2Int test, not randn (normal distribution)
    inp = np.random.rand(*ishp).astype(np.float32)
    y_golden = oxe.execute_onnx(model, {"in0": inp})["out0"]

    # Extract scale and zeropt to separate Mul/Add nodes
    model = model.transform(ExtractQuantScaleZeroPt())
    y_ret = oxe.execute_onnx(model, {"in0": inp})["out0"]
    assert np.allclose(y_golden, y_ret)

    # Verify extraction worked
    qnt_node = model.get_nodes_by_op_type("Quant")[0]
    new_scale = model.get_initializer(qnt_node.input[1])
    assert (new_scale == 1).all()
    new_zeropt = model.get_initializer(qnt_node.input[2])
    assert (new_zeropt == 0).all()
    assert len(model.get_nodes_by_op_type("Mul")) == 1
    assert len(model.get_nodes_by_op_type("Div")) == 1
    assert len(model.get_nodes_by_op_type("Add")) == 1
    assert len(model.get_nodes_by_op_type("Sub")) == 1

    model = model.transform(ConvertSubToAdd())
    model = model.transform(ConvertDivToMul())

    # Convert Quant to Requant HW node
    model = model.transform(InferRequantLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())

    # Verify conversion
    assert len(model.get_nodes_by_op_type("Quant")) == 0, "Quant should be converted"
    assert len(model.get_nodes_by_op_type("Requant")) == 1, "Expected one Requant node"
    assert len(model.get_nodes_by_op_type("ElementwiseMul")) == 2
    assert len(model.get_nodes_by_op_type("ElementwiseAdd")) == 2

    y_hw = oxe.execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_hw)

    # Specialize Layers
    model = model.transform(SpecializeLayers(test_fpga_part))

    # Set PE for Requant node
    requant_nodes = model.get_nodes_by_op_type("Requant_hls") + model.get_nodes_by_op_type(
        "Requant_rtl"
    )
    assert len(requant_nodes) == 1, "Expected one specialized Requant node"
    getCustomOp(requant_nodes[0]).set_nodeattr("PE", pe)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    # cppsim
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    y_cppsim = oxe.execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_cppsim)

    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_rtlsim = oxe.execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_rtlsim)

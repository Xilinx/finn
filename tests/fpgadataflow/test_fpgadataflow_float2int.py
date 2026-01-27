###################################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
###################################################################################

# This test was taken from the test_extract_quant_scale_zeropoint.py of qonnx
# and was extended to test the fpgadataflow node
import pytest

import numpy as np
import onnx.parser as oprs
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_shapes import InferShapes

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map

test_pynq_board = "ZCU104"
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_test_model(ishp, channelwise, bitwidth, need_extraction_scale, need_extraction_zeropt):
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
    signed = 1
    narrow = 1
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
def test_fpgadataflow_float2int(channelwise, pe):
    ishp = (1, 10)
    bitwidth = np.asarray(4.0, dtype=np.float32)
    model = make_test_model(
        ishp, channelwise, bitwidth, need_extraction_scale=True, need_extraction_zeropt=True
    )
    ishp = model.get_tensor_shape("in0")
    inp = np.random.rand(*ishp).astype(np.float32)
    y_golden = execute_onnx(model, {"in0": inp})["out0"]
    model = model.transform(ExtractQuantScaleZeroPt())
    y_ret = execute_onnx(model, {"in0": inp})["out0"]
    assert np.allclose(y_golden, y_ret)
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
    # Convert to HW
    model = model.transform(to_hw.InferQuantAsFloat2Int())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    assert len(model.get_nodes_by_op_type("ElementwiseFloat2Int")) == 1
    assert len(model.get_nodes_by_op_type("ElementwiseMul")) == 2
    assert len(model.get_nodes_by_op_type("ElementwiseAdd")) == 2
    y_hw = execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_hw)

    # Specialize Layers
    model = model.transform(SpecializeLayers(test_fpga_part))
    float2int_node = model.get_nodes_by_op_type("ElementwiseFloat2Int_hls")[0]
    getCustomOp(float2int_node).set_nodeattr("PE", pe)

    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())

    # cppsim
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    y_cppsim = execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_cppsim)

    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_rtlsim = execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_rtlsim)

    # stitched ip
    model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_trace", "fifosim_trace.wdb")
    y_rtl_stitch = execute_onnx(model, {model.graph.input[0].name: inp})[model.graph.output[0].name]
    assert np.allclose(y_golden, y_rtl_stitch)

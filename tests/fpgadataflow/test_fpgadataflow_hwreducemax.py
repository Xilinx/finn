############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Based on HWSoftmax test implementation
############################################################################

import pytest

import numpy as np
import torch
import torch.nn as nn
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferHWReduceMax
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from qonnx.custom_op.registry import getCustomOp

test_fpga_part: str = "xczu7ev-ffvc1156-2-e"
target_clk_ns = 5
export_onnx_path = "pytest_reducemax_dut.onnx"


class ReduceMaxSimple(nn.Module):
    def __init__(self, dim=-1):
        super(ReduceMaxSimple, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use torch.max which returns (values, indices), we want just values
        max_vals, _ = torch.max(x, dim=self.dim, keepdim=True)
        return max_vals


def create_reducemax_model(io_shape, idt, axis=-1):
    dut = ReduceMaxSimple(dim=axis)
    input_tensor = torch.rand(io_shape)
    export_qonnx(dut, input_tensor, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    return model


@pytest.mark.parametrize("simd", [1, 2, 4])
@pytest.mark.parametrize("idt", ["FLOAT32"])  # Start with float32 for floating-point operations
@pytest.mark.parametrize("exec_mode", ["rtlsim"])  # Only rtlsim for now, cppsim has simulation issues
@pytest.mark.parametrize("ifm_dim,axis", [
    ((1, 32), -1),      # Reduce last dimension: [1, 32] -> [1, 1]
    ((1, 64), -1),      # Reduce last dimension: [1, 64] -> [1, 1] 
    ((1, 3840), -1),    # YOLOv4 case: [1, 3840] -> [1, 1]
    ((2, 16), -1),      # Batch size 2: [2, 16] -> [2, 1]
])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_hwreducemax(simd, idt, exec_mode, ifm_dim, axis):
    if ifm_dim[-1] % simd != 0:
        pytest.skip(f"SIMD {simd} does not divide into last dimension {ifm_dim[-1]}")
        
    idt = DataType[idt]
    io_shape = ifm_dim
    tolerance = 1e-5

    model = create_reducemax_model(io_shape, idt, axis)

    input_data = gen_finn_dt_tensor(idt, io_shape)
    # Ensure we have some variation in the data for meaningful max operations
    input_data = np.random.uniform(-10.0, 10.0, io_shape).astype(np.float32)
    
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input_data}

    # Create reference values using the original ONNX model
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Infer HWReduceMax
    model = model.transform(InferHWReduceMax())

    # Run the model with HWReduceMax
    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=tolerance), f"HWReduceMax output does not match reference: {y_ref} vs {y_out}"

    # Set SIMD attribute on the HWReduceMax node
    for node in model.graph.node:
        if node.op_type == "HWReduceMax":
            # Get the custom operation instance and set SIMD
            getCustomOp(node).set_nodeattr("SIMD", simd)

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    # Run the hardware model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    assert np.allclose(y_ref, y_hw, atol=tolerance), f"Hardware output does not match reference: {y_ref} vs {y_hw}"


@pytest.mark.parametrize("simd", [1])
@pytest.mark.parametrize("idt", ["FLOAT32"])
@pytest.mark.parametrize("exec_mode", ["rtlsim"])  # Only rtlsim for now
def test_fpgadataflow_hwreducemax_simple(simd, idt, exec_mode):
    """Simple test case for basic functionality."""
    idt = DataType[idt]
    io_shape = (1, 8)  # Simple 8-element vector
    tolerance = 1e-5

    # Create a simple test case with known max values
    input_data = np.array([[1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0]], dtype=np.float32)
    expected_max = 9.0

    model = create_reducemax_model(io_shape, idt, -1)
    
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input_data}

    # Verify reference computation
    y_ref = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, [[expected_max]], atol=tolerance), f"Reference max computation failed: {y_ref}"

    # Transform to HWReduceMax
    model = model.transform(InferHWReduceMax())
    
    # Set SIMD attribute
    for node in model.graph.node:
        if node.op_type == "HWReduceMax":
            getCustomOp(node).set_nodeattr("SIMD", simd)

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Test hardware implementation
    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_hw, [[expected_max]], atol=tolerance), f"Hardware max computation failed: {y_hw}"

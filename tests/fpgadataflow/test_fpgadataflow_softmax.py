############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier:  BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest

import numpy as np
import torch
import torch.nn as nn
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferHWSoftmax
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, robust_rmtree

test_fpga_part: str = "xcvc1902-vsva2197-2MP-e-S"
target_clk_ns = 5


class SoftMaxSimple(nn.Module):
    def __init__(self):
        super(SoftMaxSimple, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # softmax along the last dimension

    def forward(self, x):
        x = self.softmax(x)
        return x


def create_softmax_model(io_shape, idt, build_dir):
    export_onnx_path = f"{build_dir}/pytest_softmax_dut.onnx"
    dut = SoftMaxSimple()
    input = torch.rand(io_shape)
    export_qonnx(dut, input, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    return model


@pytest.mark.parametrize("simd", [1, 2, 4])
@pytest.mark.parametrize("idt", ["INT8", "INT9"])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.parametrize("sim_style", ["cppsim", "node_by_node", "stitched_ip"])
@pytest.mark.parametrize("ifm_dim", [(1, 32, 96), (1, 3, 32, 32), (1, 3, 16, 32)])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_hwsoftmax(simd, idt, impl_style, sim_style, ifm_dim):
    build_dir = make_build_dir(prefix="test_fpgadataflow_hwsoftmax_")
    try:
        _test_fpgadataflow_hwsoftmax(
            simd, idt, impl_style, sim_style, ifm_dim, build_dir
        )
    finally:
        robust_rmtree(build_dir)


def _test_fpgadataflow_hwsoftmax(simd, idt, impl_style, sim_style, ifm_dim, build_dir):
    # RTL backend's cppsim path falls through to scipy.special.softmax,
    # which adds no value over the HLS cppsim coverage; skip it.
    if impl_style == "rtl" and sim_style == "cppsim":
        pytest.skip("RTL cppsim duplicates scipy reference, no added coverage")
    idt = DataType[idt]
    io_shape = ifm_dim
    # tighter tolerance for HLS/cppsim, looser for RTL FP32 numerical drift
    if sim_style == "cppsim":
        tolerance = 1e-5
    elif impl_style == "rtl":
        tolerance = 2**-4
        rtol = 1e-3
    else:
        tolerance = 1e-5

    model = create_softmax_model(io_shape, idt, build_dir)

    input = gen_finn_dt_tensor(idt, io_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Infer HWSoftmax
    model = model.transform(InferHWSoftmax())

    # request the desired implementation style
    getCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", impl_style)

    # run the model (HWSoftmax base falls through to scipy reference)
    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=1e-5), "Model output does not match expected output"

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    expected_op_type = f"HWSoftmax_{impl_style}"
    assert (
        model.graph.node[0].op_type == expected_op_type
    ), f"HWSoftmax wasn't converted to {expected_op_type}"

    # set SIMD post-specialize (matches layernorm test pattern)
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", int(simd))

    if sim_style == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif sim_style == "node_by_node":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif sim_style == "stitched_ip":
        model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")

    # run the model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    if impl_style == "rtl" and sim_style != "cppsim":
        assert np.allclose(
            y_ref, y_hw, rtol=rtol, atol=tolerance
        ), "Model output does not match expected output"
    else:
        assert np.allclose(
            y_ref, y_hw, atol=tolerance
        ), "Model output does not match expected output"

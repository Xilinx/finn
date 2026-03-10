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
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferHWSoftmax
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part: str = "xczu7ev-ffvc1156-2-e"
target_clk_ns = 5
export_onnx_path = "pytest_softmax_dut.onnx"


class SoftMaxSimple(nn.Module):
    def __init__(self):
        super(SoftMaxSimple, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # softmax along the last dimension

    def forward(self, x):
        x = self.softmax(x)
        return x


def create_softmax_model(io_shape, idt):
    dut = SoftMaxSimple()
    input = torch.rand(io_shape)
    export_qonnx(dut, input, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    return model


@pytest.mark.parametrize("simd", ["1", "2", "4"])
@pytest.mark.parametrize("idt", ["INT8", "INT9"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("ifm_dim", [(1, 32, 96), (1, 3, 32, 32), (1, 3, 16, 32)])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_hwsoftmax(simd, idt, exec_mode, ifm_dim):
    idt = DataType[idt]
    io_shape = ifm_dim
    tollerance = 1e-5

    model = create_softmax_model(io_shape, idt)

    input = gen_finn_dt_tensor(idt, io_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Infer HWSoftmax
    model = model.transform(InferHWSoftmax())

    # run the model
    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=tollerance), "Model output does not match expected output"

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    # run the model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    assert np.allclose(y_ref, y_hw, atol=tollerance), "Model output does not match expected output"

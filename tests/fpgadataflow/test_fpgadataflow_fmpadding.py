# Copyright (c) 2020, Xilinx
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
import os
import numpy as np

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

from finn.util.basic import pynq_part_map

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_fmpadding_modelwrapper(idim, padding, num_ch, simd, idt, pad_style):
    assert pad_style == 2, "only pad_style == 2 supported in hlslib"
    assert padding > 0, "Output dim should be greater than input dim"
    odim = idim + padding

    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, idim, idim, num_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, odim, odim, num_ch]
    )

    FMPadding = helper.make_node(
        "FMPadding_Batch",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Padding=padding,
        NumChannels=num_ch,
        inputDataType=str(idt.name),
        PaddingStyle=pad_style,
        numInputVectors=1,
        SIMD=simd,
    )

    graph = helper.make_graph(
        nodes=[FMPadding], name="fmpadding_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fmpadding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    return model


# input image dimension
@pytest.mark.parametrize("idim", [8])
# number of rows and number of cols to add
@pytest.mark.parametrize("pad", [2, 3])
# number of channels
@pytest.mark.parametrize("num_ch", [2, 4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# PaddingStyle: selects behavior when (odim-idim)%2 != 0
@pytest.mark.parametrize("pad_style", [2])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType.INT2, DataType.INT4])
# execution mode
@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fmpadding(idim, pad, num_ch, simd, pad_style, idt, mode):
    if num_ch % simd != 0:
        pytest.skip(" num_ch % simd != 0, skipping")
    # generate input data
    x = gen_finn_dt_tensor(idt, [1, idim, idim, num_ch])
    input_dict = {"inp": x}
    odim = idim + pad

    model = make_single_fmpadding_modelwrapper(idim, pad, num_ch, simd, idt, pad_style)
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode(mode))
    model = model.transform(GiveUniqueNodeNames())
    if mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    expected_oshape = (1, odim, odim, num_ch)
    assert y_produced.shape == expected_oshape

    # calculate reference
    # calculate correct pad according to parameters
    if pad_style == 2:
        if pad % 2 == 0:
            pad_up = pad // 2
            pad_left = pad // 2
        else:
            pad_up = pad // 2 + 1
            pad_left = pad // 2 + 1
    else:
        pad_up = pad // 2
        pad_left = pad // 2

    pad_down = pad - pad_up
    pad_right = pad - pad_left

    y_expected = np.pad(
        x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), "constant"
    )

    assert (y_produced == y_expected).all()

    if mode == "rtlsim":
        node = model.get_nodes_by_op_type("FMPadding_Batch")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

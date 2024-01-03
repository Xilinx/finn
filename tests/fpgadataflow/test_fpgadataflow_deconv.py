# Copyright (c) 2023, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import (
    InferConvInpGen,
    InferQuantizedMatrixVectorActivation,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import (
    InferPixelPaddingDeconv,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import pynq_part_map

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def set_up_reference_model(idt, wdt, k, idim, ifm_ch, ofm_ch, stride, padding):
    idim_h, idim_w = idim
    stride_h, stride_w = stride
    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1
    odt = DataType["INT32"]

    inp = helper.make_tensor_value_info(
        "inp",
        TensorProto.FLOAT,
        [
            1,
            ifm_ch,
            idim_h,
            idim_w,
        ],
    )
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, odim_h, odim_w])

    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [ifm_ch, ofm_ch, k, k])

    ConvTranspose = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        dilations=(1, 1),
        group=1,
        kernel_shape=(k, k),
        pads=(padding, padding, padding, padding),
        strides=(stride_h, stride_w),
    )

    node_list = [ConvTranspose]
    value_info = [W]

    graph = helper.make_graph(
        nodes=node_list,
        name="convtranspose_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )

    model = qonnx_make_model(graph, producer_name="convtranspose-model")
    model = ModelWrapper(model)

    # initialize model
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(model.graph.output[0].name, odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, [ifm_ch, ofm_ch, k, k])
    model.set_initializer("W", w_tensor)

    model = model.transform(InferShapes())

    return model


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("stride", [[2, 2], [2, 3]])
# number of channels
@pytest.mark.parametrize("ifm_ch", [2])
# number of channels
@pytest.mark.parametrize("ofm_ch", [4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# PE
@pytest.mark.parametrize("pe", [1, 2])
# kernel size
@pytest.mark.parametrize("k", [2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
# exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv(idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, exec_mode):
    idt = wdt = DataType["INT4"]
    wdt = idt
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    if idim_h == idim_w and stride_h == stride_w:
        convinpgen_rtl = False
    else:
        convinpgen_rtl = True

    if exec_mode == "cppsim" and convinpgen_rtl:
        pytest.skip("ConvolutionInputGenerator_rtl has no cppsim, skipping cppsim")

    ref_model = set_up_reference_model(idt, wdt, k, idim, ifm_ch, ofm_ch, stride, padding)

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, idim_h, idim_w])
    input_dict = {"inp": input_tensor}

    model = ref_model.transform(InferPixelPaddingDeconv())
    model = model.transform(InferConvInpGen(use_rtl_variant=convinpgen_rtl))
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())

    for n in model.graph.node:
        if n.op_type == "ConvolutionInputGenerator" and not convinpgen_rtl:
            convinputgen_node = getCustomOp(n)
            convinputgen_node.set_nodeattr("SIMD", simd)
        elif n.op_type == "MatrixVectorActivation":
            mvau_node = getCustomOp(n)
            mvau_node.set_nodeattr("PE", pe)
            mvau_node.set_nodeattr("SIMD", simd)

    expected_oshape = (1, ofm_ch, odim_h, odim_w)
    y_expected = oxe.execute_onnx(ref_model, input_dict)["outp"]
    # cppsim
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        y_produced = oxe.execute_onnx(model, input_dict)["outp"]
        assert y_produced.shape == expected_oshape
        assert (y_produced == y_expected).all()

    # rtlsim
    else:
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))
        y_produced = oxe.execute_onnx(model, input_dict)["outp"]
        assert y_produced.shape == expected_oshape
        assert (y_produced == y_expected).all()

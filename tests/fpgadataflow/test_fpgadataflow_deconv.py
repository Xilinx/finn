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

import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import (
    InferConvInpGen,
    InferQuantizedMatrixVectorActivation,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.util.basic import pynq_part_map

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def convolution_2d(
    x: np.ndarray,
    weight: np.ndarray,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    Ic, Ih, Iw = x[0, :].shape
    assert Ic == in_channels
    Oh = 1 + (Ih - kernel_size + 2 * padding) // stride
    Ow = 1 + (Iw - kernel_size + 2 * padding) // stride
    output = np.zeros((1, out_channels, Oh, Ow))
    for oh in range(Oh):
        for ow in range(Ow):
            for oc in range(out_channels):
                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = stride * oh + kh - padding
                            iw = stride * ow + kw - padding
                            if ih >= 0 and ih < Ih and iw >= 0 and iw < Iw:
                                output[0, oc, oh, ow] += (
                                    weight[oc, ic, kh, kw] * x[0, ic, ih, iw]
                                )
    return output


def fractionally_strided_convolution(
    x: np.ndarray,
    weight: np.ndarray,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 0,
    stride: np.ndarray = np.array([1, 1]),
) -> np.ndarray:
    x_ = np.zeros(
        (
            1,
            x.shape[1],
            x.shape[2] + (x.shape[2] - 1) * (stride[0] - 1),
            x.shape[3] + (x.shape[3] - 1) * (stride[1] - 1),
        )
    )
    # adding the zeros into the input space for the fractional strides
    for i in range(x.shape[2]):
        for j in range(x.shape[3]):
            ih = i * stride[0]
            iw = j * stride[1]
            x_[0, :, ih, iw] = x[0, :, i, j]
    padding = kernel_size - padding - 1
    stride = 1
    # weight = np.rot90(weight, 2, [2,3])
    # weight = np.moveaxis(weight, 0, 1)
    output = convolution_2d(
        x_,
        weight=weight,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return output


def set_up_reference_model(idt, wdt, k, idim, ifm_ch, ofm_ch, stride, padding, simd):

    idim_h, idim_w = idim
    stride_h, stride_w = stride
    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    odt = DataType["INT32"]

    padded_odim_h = idim_h + (idim_h - 1) * (stride_h - 1)
    padded_odim_w = idim_w + (idim_w - 1) * (stride_w - 1)
    conv_padding = k - padding - 1

    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, idim_h, idim_w, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_ch, odim_h, odim_w]
    )
    out_pad = helper.make_tensor_value_info(
        "out_pad", TensorProto.FLOAT, [1, padded_odim_h, padded_odim_w, ifm_ch]
    )
    out_pad_trans = helper.make_tensor_value_info(
        "out_pad_trans", TensorProto.FLOAT, [1, ifm_ch, padded_odim_h, padded_odim_w]
    )
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, ifm_ch, k, k])

    FMPadding_Pixel = helper.make_node(
        "FMPadding_Pixel",
        [inp],
        [out_pad],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Stride=stride,
        NumChannels=ifm_ch,
        inputDataType=str(idt.name),
        numInputVectors=1,
        SIMD=simd,
    )
    Transpose = helper.make_node(
        "Transpose", ["out_pad"], ["out_pad_trans"], perm=[0, 3, 1, 2]
    )

    Conv = helper.make_node(
        "Conv",
        [out_pad_trans, W],
        [outp],
        dilations=(1, 1),
        group=1,
        kernel_shape=(k, k),
        pads=(conv_padding, conv_padding, conv_padding, conv_padding),
        strides=(1, 1),
    )

    node_list = [FMPadding_Pixel, Transpose, Conv]
    value_info = [W]

    graph = helper.make_graph(
        nodes=node_list,
        name="deconv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )
    model = qonnx_make_model(graph, producer_name="deconv-model")
    model = ModelWrapper(model)

    # initialize model
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(model.graph.output[0].name, odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, ifm_ch, k, k])
    model.set_initializer("W", w_tensor)

    model = model.transform(InferShapes())

    return model


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("stride", [[2, 2], [2, 3]])
# number of channels
@pytest.mark.parametrize("ifm_ch", [2, 4])
# number of channels
@pytest.mark.parametrize("ofm_ch", [2, 4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# PE
@pytest.mark.parametrize("pe", [1, 2])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# padding
@pytest.mark.parametrize("padding", [0, 1])
# execution mode
@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# @pytest.mark.parametrize("mode", ["stitched_ip_rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv(idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, mode):
    idt = wdt = DataType["INT4"]
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    if idim_h == idim_w and stride_h == stride_w:
        convinpgen_rtl = False
    else:
        convinpgen_rtl = True

    if convinpgen_rtl and mode == "cppsim":
        pytest.skip("ConvolutionInputGenerator_rtl has no cppsim, skipping")

    model = set_up_reference_model(
        idt, wdt, k, idim, ifm_ch, ofm_ch, stride, padding, simd
    )

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, idim_h, idim_w, ifm_ch])
    weight_tensor = model.get_initializer("W")
    input_dict = {"inp": input_tensor}

    model = model.transform(LowerConvsToMatMul())
    model = model.transform(InferDataTypes())
    model = model.transform(InferConvInpGen(use_rtl_variant=convinpgen_rtl))
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferShapes())
    if mode == "stitched_ip_rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
    else:
        model = model.transform(SetExecMode(mode))
    model = model.transform(GiveUniqueNodeNames())

    for n in model.graph.node:
        if n.op_type == "ConvolutionInputGenerator" and not convinpgen_rtl:
            convinputgen_node = getCustomOp(n)
            convinputgen_node.set_nodeattr("SIMD", simd)
        elif n.op_type == "MatrixVectorActivation":
            mvau_node = getCustomOp(n)
            mvau_node.set_nodeattr("PE", pe)
            mvau_node.set_nodeattr("SIMD", simd)

    if mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif mode == "stitched_ip_rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(
            CreateStitchedIP(test_fpga_part, target_clk_ns, vitis=False)
        )

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    expected_oshape = (1, ofm_ch, odim_h, odim_w)
    assert y_produced.shape == expected_oshape

    y_expected = fractionally_strided_convolution(
        input_tensor.transpose(0, 3, 1, 2),
        weight_tensor,
        ifm_ch,
        ofm_ch,
        k,
        padding,
        stride,
    )
    assert (y_produced == y_expected).all()

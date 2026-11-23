# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map
from finn.util.test import tree_model_test

test_pynq_board = "AUP-ZU3_8GB"
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_fmpadding_modelwrapper(idim, padding, num_ch, simd, idt, eager=1):
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    idim_h, idim_w = idim

    assert pad_h > 0 or pad_w > 0, "Output dim should be greater than input dim"
    odim_h = idim_h + pad_h
    odim_w = idim_w + pad_w

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim_h, idim_w, num_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, num_ch])

    FMPadding = helper.make_node(
        "FMPadding",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Padding=padding,
        NumChannels=num_ch,
        inputDataType=str(idt.name),
        numInputVectors=1,
        SIMD=simd,
        Eager=eager,
    )

    graph = helper.make_graph(
        nodes=[FMPadding], name="fmpadding_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fmpadding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    return model


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("pad", [[1, 1, 1, 1], [1, 1, 2, 2], [1, 3, 2, 3], [7, 0, 8, 0]])
# number of channels
@pytest.mark.parametrize("num_ch", [2, 4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["INT4"]])
# execution mode
@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# emit the leading padding ahead of the first input (1) or only once it is available (0)
@pytest.mark.parametrize("eager", [1, 0], ids=["eager", "lazy"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fmpadding(idim, pad, num_ch, simd, idt, mode, eager):
    if num_ch % simd != 0:
        pytest.skip(" num_ch % simd != 0, skipping")
    if mode == "cppsim" and not eager:
        pytest.skip("cppsim does not model the handshake of the leading padding")

    idim_h, idim_w = idim
    pad_h = pad[0] + pad[2]
    pad_w = pad[1] + pad[3]

    # generate input data
    x = gen_finn_dt_tensor(idt, [1, idim_h, idim_w, num_ch])
    input_dict = {"inp": x}
    odim_h = idim_h + pad_h
    odim_w = idim_w + pad_w

    y_expected = np.pad(x, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), "constant")
    expected_oshape = (1, odim_h, odim_w, num_ch)

    model = make_single_fmpadding_modelwrapper(idim, pad, num_ch, simd, idt, eager)

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all(), "HW layer execution failed"

    model = model.transform(SpecializeLayers(test_fpga_part))

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

    assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all()

    if mode == "rtlsim":
        op_type = "FMPadding_rtl"
        node = model.get_nodes_by_op_type(op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


# input image dimension
# Both backends, across the pad shapes that change the schedule (symmetric,
# asymmetric, and one-sided) and folded/unfolded channels -- not their product.
@pytest.mark.parametrize(
    "impl_style,pad,num_ch,simd",
    [
        ("rtl", [1, 1, 1, 1], 4, 1),
        ("rtl", [1, 1, 1, 1], 4, 2),
        ("rtl", [1, 1, 2, 2], 4, 2),
        ("rtl", [7, 0, 8, 0], 4, 2),
        ("rtl", [1, 1, 1, 1], 2, 2),
        ("rtl", [0, 0, 2, 1], 4, 2),
        ("hls", [1, 1, 1, 1], 4, 1),
        ("hls", [1, 1, 1, 1], 4, 2),
        ("hls", [1, 1, 2, 2], 4, 2),
        ("hls", [7, 0, 8, 0], 4, 2),
        ("hls", [1, 1, 1, 1], 2, 2),
    ],
)
@pytest.mark.parametrize("idim", [[10, 8]])
@pytest.mark.parametrize("idt", [DataType["INT2"]])
@pytest.mark.parametrize("mode", ["rtlsim"])
# emit the leading padding ahead of the first input (1) or only once it is available (0)
@pytest.mark.parametrize("eager", [1, 0], ids=["eager", "lazy"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_fmpadding(
    idim, pad, num_ch, simd, idt, mode, impl_style, eager
):
    if num_ch % simd != 0:
        pytest.skip(" num_ch % simd != 0, skipping")
    if impl_style == "hls" and not eager:
        pytest.skip("lazy leading padding is covered by the rtl cases")

    model = make_single_fmpadding_modelwrapper(idim, pad, num_ch, simd, idt, eager)
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode(mode))

    node_details = ("FMPadding", idim, pad, num_ch, simd, idt, mode, impl_style)
    part = "xc7z020clg400-1"
    target_clk_ns = 4

    # The schedule is exact for both Eager settings, including the read-ahead of
    # the sidestep register and the phase of the stored window, so no tolerance.
    max_allowed_volume_frac = 0
    volume_const = 0
    max_allowed_length_frac = 0
    length_const = 0

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"


def make_single_fmpadding_pixel_modelwrapper(idim, stride, num_ch, simd, idt):
    idim_h, idim_w = idim
    stride_h, stride_w = stride
    odim_h = idim_h + (idim_h - 1) * (stride_h - 1)
    odim_w = idim_w + (idim_w - 1) * (stride_w - 1)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim_h, idim_w, num_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, num_ch])

    node = helper.make_node(
        "FMPadding_Pixel",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Stride=stride,
        NumChannels=num_ch,
        SIMD=simd,
        inputDataType=str(idt.name),
        numInputVectors=1,
    )
    graph = helper.make_graph(nodes=[node], name="pad_pixel_graph", inputs=[inp], outputs=[outp])
    model = qonnx_make_model(graph, producer_name="pad_pixel-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


# Square and non-square strides, folded and unfolded channels, on two image
# shapes -- the four things the raster mask depends on, one at a time.
@pytest.mark.parametrize(
    "idim,stride,num_ch,simd",
    [
        ([4, 4], [2, 2], 4, 2),
        ([4, 4], [2, 2], 8, 4),
        ([4, 4], [2, 2], 8, 8),
        ([4, 4], [3, 2], 4, 2),
        ([4, 4], [3, 3], 8, 4),
        ([6, 4], [2, 2], 4, 2),
        ([6, 4], [3, 2], 8, 4),
        ([6, 4], [2, 3], 4, 4),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_fmpadding_pixel(idim, stride, num_ch, simd):
    if num_ch % simd != 0:
        pytest.skip("num_ch % simd != 0, skipping")
    idt = DataType["INT8"]
    model = make_single_fmpadding_pixel_modelwrapper(idim, stride, num_ch, simd, idt)
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode("rtlsim"))

    node_details = ("FMPadding_Pixel", tuple(idim), tuple(stride), num_ch, simd)
    # The schedule, like get_exp_cycles, assumes the HLS pipeline reaches II=1.
    # Whether it does is a synthesis outcome, not a property of the node: this
    # same node schedules at II=1, 2 or 4 depending on the part and the clock
    # target. Characterize on a target that meets II=1, otherwise the comparison
    # measures HLS scheduling rather than the model.
    part = "xczu7ev-ffvc1156-2-e"
    target_clk_ns = 4

    # The padded raster is emitted one word per cycle with no wind-up to model,
    # so both the period and the token counts should land within a couple of
    # cycles of rtlsim.
    max_allowed_volume_frac = 0.01
    volume_const = 4
    max_allowed_length_frac = 0.01
    length_const = 4

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"

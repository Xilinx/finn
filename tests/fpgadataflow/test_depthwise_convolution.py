# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_depthwise_conv_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding):
    """Build an ONNX Conv model (depthwise, group=ifm_ch) as reference + HW source.

    The model uses NCHW layout (standard ONNX Conv convention).
    If act is not None, a MultiThreshold is appended after the Conv.
    """
    ofm_ch = ifm_ch
    total_pad = 2 * padding
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, total_pad=total_pad)

    if act is None:
        odt = DataType["INT32"]
    else:
        odt = act

    input_shape = [1, ifm_ch, ifm_dim, ifm_dim]
    output_shape = [1, ofm_ch, ofm_dim, ofm_dim]
    conv_weight_shape = [ofm_ch, 1, k, k]

    conv_config = {
        "dilations": [1, 1],
        "group": ifm_ch,
        "kernel_shape": [k, k],
        "pads": [padding, padding, padding, padding],
        "strides": [stride, stride],
    }

    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    value_info = [
        oh.make_tensor_value_info("conv_w", TensorProto.FLOAT, conv_weight_shape),
    ]

    nodes = []

    if act is None:
        top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
        conv_out_name = "top_out"
    else:
        conv_out_vi = oh.make_tensor_value_info(
            "conv_out", TensorProto.FLOAT, output_shape
        )
        value_info.append(conv_out_vi)
        conv_out_name = "conv_out"
        # Threshold in NCHW layout
        T_vi = oh.make_tensor_value_info("T", TensorProto.FLOAT, [ofm_ch, 15])
        value_info.append(T_vi)
        top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    nodes.append(
        oh.make_node("Conv", ["top_in", "conv_w"], [conv_out_name], **conv_config)
    )

    if act is not None:
        nodes.append(
            oh.make_node(
                "MultiThreshold",
                domain="qonnx.custom_op.general",
                inputs=[conv_out_name, "T"],
                outputs=["top_out"],
                out_dtype=odt.name,
                out_scale=1.0,
                out_bias=0.0,
            )
        )

    graph = oh.make_graph(
        nodes=nodes,
        name="depthwise_conv_test",
        inputs=[top_in],
        outputs=[top_out],
        value_info=value_info,
    )
    model = ModelWrapper(qonnx_make_model(graph))

    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_datatype("top_out", odt)
    model.set_tensor_datatype("conv_w", wdt)
    model.set_initializer("conv_w", gen_finn_dt_tensor(wdt, conv_weight_shape))

    if act is not None:
        (mn, mx) = calculate_signed_dot_prod_range(idt, wdt, k * k)
        n_steps = odt.get_num_possible_values() - 1
        T_values = np.random.randint(mn, mx - 1, (ofm_ch, n_steps)).astype(np.float32)
        T_values = np.sort(T_values, axis=1)
        model.set_initializer("T", T_values)
        model.set_tensor_datatype("T", DataType["INT32"])

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


# PE
@pytest.mark.parametrize("pe", [1, 2, 4])
# Output activation
@pytest.mark.parametrize("act", [None, DataType["UINT4"]])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_depthwise_conv_hw_cppsim(act, pe, k, stride, padding):
    idt = wdt = DataType["INT4"]
    ifm_dim = 6
    ifm_ch = 4

    model = make_depthwise_conv_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding)

    input_shape = [1, ifm_ch, ifm_dim, ifm_dim]
    input_tensor = gen_finn_dt_tensor(idt, input_shape)
    input_dict = {model.get_first_global_in(): input_tensor}

    # InferDotProd handles Conv directly (no LowerConvsToMatMul needed)
    new_model = model.transform(to_hw.InferDotProd())
    new_model = new_model.transform(to_hw.InferThresholdingLayer())
    new_model = new_model.transform(SpecializeLayers("xc7z020clg400-1"))

    # set PE in DotProd node
    for n in new_model.graph.node:
        if n.op_type.startswith("DotProd"):
            dp_inst = getCustomOp(n)
            dp_inst.set_nodeattr("PE", pe)

    new_model = new_model.transform(SetExecMode("cppsim"))
    new_model = new_model.transform(PrepareCppSim())
    new_model = new_model.transform(CompileCppSim())

    assert oxe.compare_execution(model, new_model, input_dict)


# PE
@pytest.mark.parametrize("pe", [1, 2, 4])
# Output activation
@pytest.mark.parametrize("act", [None, DataType["UINT4"]])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_depthwise_conv_hw_rtlsim(act, pe, k, stride, padding):
    idt = wdt = DataType["INT4"]
    ifm_dim = 6
    ifm_ch = 4

    model = make_depthwise_conv_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding)

    input_shape = [1, ifm_ch, ifm_dim, ifm_dim]
    input_tensor = gen_finn_dt_tensor(idt, input_shape)
    input_dict = {model.get_first_global_in(): input_tensor}

    # InferDotProd handles Conv directly (no LowerConvsToMatMul needed)
    new_model = model.transform(to_hw.InferDotProd())
    new_model = new_model.transform(to_hw.InferThresholdingLayer())
    new_model = new_model.transform(SpecializeLayers("xc7z020clg400-1"))

    # set PE in DotProd node
    for n in new_model.graph.node:
        if n.op_type.startswith("DotProd"):
            dp_inst = getCustomOp(n)
            dp_inst.set_nodeattr("PE", pe)

    new_model = new_model.transform(SetExecMode("rtlsim"))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
    new_model = new_model.transform(HLSSynthIP())
    new_model = new_model.transform(PrepareRTLSim())

    assert oxe.compare_execution(model, new_model, input_dict)

# Copyright (c) 2022, Advanced Micro Devices, Inc.
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


import numpy as np
import onnx.parser as oprs
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


def create_conv_model(
    idim_h, idim_w, ifm, k_h, k_w, stride_h, stride_w, dil_h, dil_w, ofm, idt, wdt, pads, depthwise
):
    np.random.seed(0)
    group = ifm if depthwise else 1
    group_str = str(group)
    ishp = (1, ifm, idim_h, idim_w)
    int_dim_h = compute_conv_output_dim(
        idim_h, k_h, stride_h, total_pad=pads[0] + pads[2], dilation=dil_h
    )
    int_dim_w = compute_conv_output_dim(
        idim_w, k_w, stride_w, total_pad=pads[1] + pads[3], dilation=dil_w
    )

    oshp = (1, ifm, int_dim_h, int_dim_w) if depthwise else (1, ofm, int_dim_h, int_dim_w)
    wshp = (ifm, 1, k_h, k_w) if depthwise else (ofm, ifm, k_h, k_w)
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    kshp_str = str([k_h, k_w])
    pad_0_str = str(list(pads))
    stride_str = str([stride_h, stride_w])
    dil_str = str([dil_h, dil_w])

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 11]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float{wshp_str} param_c0_weight
    >
    {{
        out0 = Conv<
                dilations={dil_str},group={group_str},kernel_shape={kshp_str},pads={pad_0_str},
                strides={stride_str}
            >(in0, param_c0_weight)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model.set_tensor_datatype("in0", idt)
    model.set_tensor_datatype("param_c0_weight", wdt)
    model.set_initializer("param_c0_weight", gen_finn_dt_tensor(wdt, wshp))
    return model


def test_convert_to_hls_rdwdl_conv():
    idim_h = 192
    idim_w = 14
    ifm = 64
    k_h = 3
    k_w = 3
    stride_h = 1
    stride_w = 1
    dil_h = 2
    dil_w = 3
    ofm = 64
    pads = [2, 3, 2, 3]
    idt = DataType["UINT8"]
    wdt = DataType["INT8"]
    depthwise = True
    model = create_conv_model(
        idim_h,
        idim_w,
        ifm,
        k_h,
        k_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        ofm,
        idt,
        wdt,
        pads,
        depthwise,
    )
    ishp = (1, ifm, idim_h, idim_w)
    inp = gen_finn_dt_tensor(idt, ishp)
    exp_out = execute_onnx(model, {"in0": inp})["out0"]
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
    model = model.transform(to_hls.InferVectorVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_inst = getCustomOp(parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0])
    model = ModelWrapper(sdp_inst.get_nodeattr("model"))
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5, vitis=False))
    model.set_metadata_prop("exec_mode", "rtlsim")
    inp_nhwc = inp.transpose(0, 2, 3, 1)
    gen_out_nhwc = execute_onnx(model, {"global_in": inp_nhwc})["global_out"]
    gen_out = gen_out_nhwc.transpose(0, 3, 1, 2)
    assert (gen_out == exp_out).all()

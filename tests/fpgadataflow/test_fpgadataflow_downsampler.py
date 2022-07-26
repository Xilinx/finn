# Copyright (c) 2022, Xilinx, Inc.
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
import onnx.parser as oprs
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def build_model(is_1d, in_dim, k, stride, dt_in, dt_w, pad_half=0, flip_1d=False):
    np.random.seed(0)
    out_dim = compute_conv_output_dim(in_dim, k, stride, 2 * pad_half)
    ifm = 8
    ofm = 16
    if is_1d:
        if flip_1d:
            shape_in = [1, ifm, 1, in_dim]
            shape_out = [1, ofm, 1, out_dim]
            shape_k = [1, k]
            shape_s = [1, stride]
            shape_p = [0, pad_half, 0, pad_half]
        else:
            shape_in = [1, ifm, in_dim, 1]
            shape_out = [1, ofm, out_dim, 1]
            shape_k = [k, 1]
            shape_s = [stride, 1]
            shape_p = [pad_half, 0, pad_half, 0]
    else:
        shape_in = [1, ifm, in_dim, in_dim]
        shape_out = [1, ofm, out_dim, out_dim]
        shape_k = [k, k]
        shape_s = [stride, stride]
        shape_p = [pad_half, pad_half, pad_half, pad_half]
    shape_w = [ofm, ifm] + shape_k

    sstr_in = str(shape_in)
    sstr_out = str(shape_out)
    sstr_k = str(shape_k)
    sstr_s = str(shape_s)
    sstr_p = str(shape_p)
    sstr_w = str(shape_w)

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{sstr_in} in0) => (float{sstr_out} out0)
    <
        float{sstr_w} param_w_conv0
    >
    {{
        out0 = Conv<kernel_shape={sstr_k}, group=1, pads={sstr_p},
                    strides={sstr_s}>(in0, param_w_conv0)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt_in)
    model.set_tensor_datatype("param_w_conv0", dt_w)
    model.set_initializer("param_w_conv0", gen_finn_dt_tensor(dt_w, shape_w))
    model = model.transform(InferShapes())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(InferShapes())
    return model


@pytest.mark.parametrize("is_1d", [True, False])
@pytest.mark.parametrize("flip_1d", [True, False])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_downsampler(is_1d, flip_1d, exec_mode):
    if flip_1d and not is_1d:
        pytest.skip("flip_1d only applicable for is_1d")
    in_dim = 32
    k = 1
    stride = 2
    dt_in = DataType["UINT8"]
    dt_w = DataType["INT2"]
    model = build_model(
        is_1d, in_dim, k, stride, dt_in, dt_w, pad_half=0, flip_1d=flip_1d
    )
    inp = gen_finn_dt_tensor(dt_in, model.get_tensor_shape("in0"))
    idict = {"in0": inp}
    y_expected = execute_onnx(model, idict)["out0"]
    model = model.transform(to_hls.InferConvInpGen())
    assert len(model.get_nodes_by_op_type("DownSampler")) == 1
    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")
    y_produced = execute_onnx(model, idict)["out0"]
    assert (y_produced == y_expected).all()
    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DownSampler")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        # small adjustment for 2D testcase due to how rtlsim works:
        # output is finished before all pixels are read, since last
        # row is dropped (rtlsim finishes based on # of expected
        # pixels)
        if not is_1d:
            exp_cycles = exp_cycles - in_dim
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

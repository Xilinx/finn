# Copyright (c) 2022, Xilinx
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
from qonnx.transformation.infer_shapes import InferShapes

# from qonnx.util.basic import gen_finn_dt_tensor

# import finn.core.onnx_exec as oxe


@pytest.mark.fpgadataflow
@pytest.mark.vivado
# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT8"]])
# output datatype
@pytest.mark.parametrize("odt", [DataType["UINT8"]])
# input and output shape
@pytest.mark.parametrize("shp", [[1, 1024, 1, 1]])
def test_fpgadataflow_quantmaxnorm(idt, odt, shp):
    np.random.seed(0)
    shp_str = str(shp)
    n_steps = idt.get_num_possible_values() - 1
    T = np.random.uniform(0, 1.0, (shp[3], n_steps)).astype(np.float32)
    T = np.sort(T, axis=1)
    odt_str = str(odt)
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0) => (float{shp_str} out0)
    <
        float[1, 255] thresh_param
    >
    {{
        redmax_out = ReduceMax(in0)
        div_out = Div(in0, redmax_out)
        out0 = qonnx.custom_op.general.MultiThreshold<out_dtype=odt_str,
            data_layout="NHWC">(div_out, thresh_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", idt)
    model.set_initializer("thresh_param", T)
    model = model.transform(InferShapes())
    model.save("maxnorm_before.onnx")

    idt_str = str(idt)
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0) => (float{shp_str} out0)
    {{
        out0 = finn.custom_op.fpgadataflow.QuantMaxNorm<IFMDim=[1024,1],
            NorMax=0,inputDataType={idt_str},outputDataType={odt_str}>(in0)
    }}
    """

    new_model = oprs.parse_model(input)
    new_model = ModelWrapper(new_model)
    new_model = new_model.transform(InferShapes())
    new_model.save("maxnorm_after.onnx")

    # generate input data
    # x = gen_finn_dt_tensor(idt, shp)
    # package input data as dictionary
    # input_dict = {"inp": x}
    # y_expected = oxe.execute_onnx(model, input_dict)
    # y_produced = oxe.execute_onnx(new_model, input_dict)

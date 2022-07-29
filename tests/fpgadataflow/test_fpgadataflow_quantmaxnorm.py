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
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import InferQuantizedMaxNorm
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


@pytest.mark.fpgadataflow
@pytest.mark.vivado
# python and hls code behaviour is not the same, so temporarily marked as xfail
@pytest.mark.xfail
# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT8"]])
# output datatype
@pytest.mark.parametrize("odt", [DataType["UINT8"]])
# input and output shape
@pytest.mark.parametrize("shp", [[1, 1024, 1, 1]])
# max thresholds
@pytest.mark.parametrize("thresh_max", [0.8, 1.0, 1.8])
def test_fpgadataflow_quantmaxnorm(idt, odt, shp, thresh_max):
    np.random.seed(0)
    shp_str = str(shp)
    n_steps = idt.get_num_possible_values() - 1
    T = np.random.uniform(0, thresh_max, (shp[3], n_steps)).astype(np.float64)
    T = np.sort(T, axis=1)
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
        out0 = qonnx.custom_op.general.MultiThreshold<out_dtype="{odt.name}",
            data_layout="NHWC">(div_out, thresh_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", idt)
    model.set_initializer("thresh_param", T)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # generate input data
    x = gen_finn_dt_tensor(idt, shp)
    # package input data as dictionary
    input_dict = {"global_in": x}
    y_python = oxe.execute_onnx(model, input_dict)

    # convert to hls
    model = model.transform(InferQuantizedMaxNorm())
    # cppsim
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    y_cppsim = oxe.execute_onnx(model, input_dict)

    # rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_rtlsim = oxe.execute_onnx(model, input_dict)

    assert (y_cppsim["global_out"] == y_rtlsim["global_out"]).all()
    assert (y_cppsim["global_out"] == y_python["global_out"]).all()

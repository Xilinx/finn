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

import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


def make_single_maxpool_modelwrapper(onnx_op_name, ishape, idt, pdt, pshape):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    p0 = helper.make_tensor_value_info("p0", TensorProto.FLOAT, pshape)

    model = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[inp],
            outputs=[outp],
            value_info=[p0],
            nodes=[helper.make_node(onnx_op_name, ["inp", "p0"], ["outp"])],
        )
    )

    model = ModelWrapper(model)
    model.set_initializer("p0", gen_finn_dt_tensor(pdt, pshape))
    model.set_tensor_datatype("inp", idt)
    model.transform(InferDataLayouts(), make_deepcopy=False)
    model.transform(InferShapes(), make_deepcopy=False)
    return model


# parameter datatype
@pytest.mark.parametrize(
    "pdt", [DataType["BIPOLAR"], DataType["UINT4"], DataType["INT2"]]
)
# input datatype
@pytest.mark.parametrize(
    "idt", [DataType["INT32"], DataType["UINT4"], DataType["INT4"]]
)
# function
@pytest.mark.parametrize("onnx_op_name", ["Add", "Mul"])
# vector parameter or scalar parameter (broadcast)
@pytest.mark.parametrize("scalar_param", [True, False])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
@pytest.mark.slow
def test_convert_to_hls_channelwise_layer(
    pdt, idt, onnx_op_name, scalar_param, exec_mode
):
    ifm_ch = 16
    ifm_dim = 5
    ishape = (1, ifm_ch, ifm_dim, ifm_dim)
    if scalar_param:
        pshape = (1,)
    else:
        pshape = (1, ifm_ch, 1, 1)

    np.random.seed(0)
    model = make_single_maxpool_modelwrapper(onnx_op_name, ishape, idt, pdt, pshape)

    # Since the aren't Data types with a bit width of a non power of 2,
    # there are cases where the input won't use it full range.
    if idt == DataType["INT32"]:
        x = gen_finn_dt_tensor(DataType["INT16"], (1, ifm_ch, ifm_dim, ifm_dim))
    elif idt == DataType["UINT32"]:
        x = gen_finn_dt_tensor(DataType["UINT16"], (1, ifm_ch, ifm_dim, ifm_dim))
    else:
        x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))

    input_dict = prepare_inputs(x)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hls.InferChannelwiseLinearLayer())
    new_model = new_model.transform(GiveUniqueNodeNames())

    if exec_mode == "cppsim":
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
        new_model = new_model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    ctx_produced = oxe.execute_onnx(
        new_model, input_dict, return_full_exec_context=True
    )
    y_produced = ctx_produced["outp"]

    assert (y_produced == y_expected).all()
    assert new_model.graph.node[1].op_type == "ChannelwiseOp_Batch"

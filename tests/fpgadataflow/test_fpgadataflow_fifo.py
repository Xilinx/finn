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
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor

build_dir = os.environ["FINN_BUILD_DIR"]
test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 10


def make_single_fifo_modelwrapper(Shape, Depth, fld_shape, finn_dtype):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, Shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, Shape)

    FIFO_node = helper.make_node(
        "StreamingFIFO",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        depth=Depth,
        folded_shape=fld_shape,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(
        nodes=[FIFO_node], name="fifo_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fifo-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


# shape
@pytest.mark.parametrize("Shape", [[1, 128]])
# inWidth
@pytest.mark.parametrize("folded_shape", [[1, 1, 128]])
# outWidth
@pytest.mark.parametrize("depth", [16])
# finn_dtype
@pytest.mark.parametrize("finn_dtype", [DataType.BIPOLAR])  # , DataType.INT2])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fifo_rtlsim(Shape, folded_shape, depth, finn_dtype):

    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, Shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_fifo_modelwrapper(Shape, depth, folded_shape, finn_dtype)

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        y == x
    ).all(), """The output values are not the same as the
       input values anymore."""
    assert y.shape == tuple(Shape), """The output shape is incorrect."""

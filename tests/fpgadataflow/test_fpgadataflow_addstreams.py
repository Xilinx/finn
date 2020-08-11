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
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer


def make_addstreams_modelwrapper(ch, pe, idt):
    inp1 = helper.make_tensor_value_info("inp1", TensorProto.FLOAT, [1, ch])
    inp2 = helper.make_tensor_value_info("inp2", TensorProto.FLOAT, [1, ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])

    addstreams_node = helper.make_node(
        "AddStreams_Batch",
        ["inp1", "inp2"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        NumChannels=ch,
        PE=pe,
        inputDataType=idt.name,
    )
    graph = helper.make_graph(
        nodes=[addstreams_node], name="graph", inputs=[inp1, inp2], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="addstreams-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp1", idt)
    model.set_tensor_datatype("inp2", idt)

    return model


def prepare_inputs(input1, input2):
    return {"inp1": input1, "inp2": input2}


# data types
@pytest.mark.parametrize("idt", [DataType.UINT4, DataType.UINT8])
# channels
@pytest.mark.parametrize("ch", [1, 64])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
def test_fpgadataflow_addstreams(idt, ch, fold, exec_mode):
    if fold == -1:
        pe = 1
    else:
        pe = max(1, ch // fold)
    assert ch % pe == 0

    # generate input data
    x1 = gen_finn_dt_tensor(idt, (1, ch))
    x2 = gen_finn_dt_tensor(idt, (1, ch))

    model = make_addstreams_modelwrapper(ch, pe, idt)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # prepare input data
    input_dict = prepare_inputs(x1, x2)

    oshape = model.get_tensor_shape("outp")
    y = x1 + x2
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("AddStreams_Batch")[0]
        inst = getCustomOp(node)
        sim_cycles = inst.get_nodeattr("sim_cycles")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, sim_cycles, atol=10)
        assert exp_cycles != 0

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
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
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


def make_dupstreams_modelwrapper(ch, pe, idim, idt, n_dupl):
    shape = [1, idim, idim, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    out_names = []
    out_vi = []
    for i in range(n_dupl):
        outp_name = "outp%d" % i
        out_names.append(outp_name)
        out_vi.append(
            helper.make_tensor_value_info(outp_name, TensorProto.FLOAT, shape)
        )

    dupstrm_node = helper.make_node(
        "DuplicateStreams_Batch",
        ["inp"],
        out_names,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        NumOutputStreams=n_dupl,
        PE=pe,
        inputDataType=idt.name,
        numInputVectors=[1, idim, idim],
    )
    graph = helper.make_graph(
        nodes=[dupstrm_node], name="graph", inputs=[inp], outputs=out_vi
    )

    model = qonnx_make_model(graph, producer_name="addstreams-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def prepare_inputs(input_tensor, idt):
    return {"inp": input_tensor}


# data type
@pytest.mark.parametrize("idt", [DataType["INT4"], DataType["UINT16"]])
# channels
@pytest.mark.parametrize("ch", [64])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# image dimension
@pytest.mark.parametrize("imdim", [7])
# amount of duplication
@pytest.mark.parametrize("n_dupl", [2, 3])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_duplicatestreams(idt, ch, fold, imdim, n_dupl, exec_mode):
    if fold == -1:
        pe = 1
    else:
        pe = ch // fold
    assert ch % pe == 0

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, imdim, imdim, ch))

    model = make_dupstreams_modelwrapper(ch, pe, imdim, idt, n_dupl)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # prepare input data and execute
    input_dict = prepare_inputs(x, idt)
    output_dict = oxe.execute_onnx(model, input_dict)

    expected_y = x
    for i in range(n_dupl):
        y = output_dict["outp%d" % i]
        assert (y == expected_y).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DuplicateStreams_Batch")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

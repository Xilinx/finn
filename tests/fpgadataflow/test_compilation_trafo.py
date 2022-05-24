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

import finn.util.basic as util
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_compilation_trafo():
    idt = wdt = odt = DataType["BIPOLAR"]
    mw = 8
    mh = 8
    pe = 4
    simd = 4

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    node_inp_list = ["inp", "weights", "thresh"]
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=1,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    W = util.gen_finn_dt_tensor(wdt, (mw, mh))
    model.set_initializer("weights", W)

    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    for node in model.graph.node:
        compilation_attribute = util.get_by_name(node.attribute, "executable_path")
        executable = compilation_attribute.s.decode("UTF-8")
        print(executable)
        assert os.path.isfile(
            executable
        ), """Executable of node with
            op type {} does not exist!""".format(
            node.op_type
        )

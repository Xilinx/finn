# Copyright (c) 2021, Xilinx
# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import onnx
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferSplitLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_split_model(IN_SHAPE, IN_DTYPE, SPLIT, AXIS):
    out_shapes = [IN_SHAPE[:-1] + [s] for s in SPLIT]
    outputs = []
    for i in range(len(SPLIT)):
        name = "global_out_" + str(i)
        out = oh.make_tensor_value_info(name, onnx.TensorProto.FLOAT, out_shapes[i])
        outputs.append(out)

    inp = oh.make_tensor_value_info("global_in", onnx.TensorProto.FLOAT, IN_SHAPE)
    split_init = onnx.numpy_helper.from_array(
        np.array(SPLIT, dtype=np.int64), name="Split_0_param0"
    )
    split_node = oh.make_node(
        "Split", [inp.name, split_init.name], [out.name for out in outputs], axis=AXIS
    )
    graph = oh.make_graph(nodes=[split_node], name="split_test", inputs=[inp], outputs=outputs)

    # set opset version to 13 for specific Split configuration
    opset_imports = [oh.make_opsetid("", 13)]
    model = qonnx_make_model(graph, opset_imports=opset_imports)
    model = ModelWrapper(model)
    for out in outputs:
        model.set_tensor_datatype(out.name, IN_DTYPE)
        model.set_tensor_layout(out.name, ["N", "H", "W", "C"])
    model.set_tensor_datatype(inp.name, IN_DTYPE)
    model.set_tensor_layout(inp.name, ["N", "H", "W", "C"])
    model.set_initializer(split_init.name, np.array(SPLIT, dtype=np.int64))
    model = model.transform(GiveUniqueNodeNames())

    return model


@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim", "stitched_rtlsim"])
@pytest.mark.parametrize("idt", [DataType["INT3"]])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_split(exec_mode, idt):
    fpga_part = "xc7z020clg400-1"
    clk_ns = 10
    i_shape = [1, 5, 5, 10]
    split = [2, 2, 6]
    split_axis = 3
    model = make_split_model(i_shape, idt, split, split_axis)
    assert len(model.graph.output) == len(split)
    exp_oshapes = []
    for s in split:
        oshape = i_shape.copy()
        oshape[split_axis] = s
        exp_oshapes.append(oshape)
    onames = [o.name for o in model.graph.output]
    assert all(model.get_tensor_shape(oname) == exp_oshapes[i] for i, oname in enumerate(onames))

    inputs = []
    for out_shape in exp_oshapes:
        inputs.append(np.random.randint(idt.min(), idt.max() + 1, out_shape).astype(np.float32))
    test_input = np.concatenate(inputs, axis=split_axis)
    input_dict = {model.graph.input[0].name: test_input}
    ret = execute_onnx(model, input_dict)
    for i, (k, v) in enumerate(ret.items()):
        assert (v == inputs[i]).all()

    # call transformation to convert to HW and verify conversion
    model = model.transform(InferSplitLayer())
    assert model.graph.node[0].op_type == "StreamingSplit"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow"
    ret = execute_onnx(model, input_dict)
    for i, (k, v) in enumerate(ret.items()):
        assert (v == inputs[i]).all()

    model = model.transform(SpecializeLayers(fpga_part))
    assert model.graph.node[0].op_type == "StreamingSplit_hls"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.hls"
    if exec_mode == "cppsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(fpga_part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(SpecializeLayers(fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(fpga_part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(
            CreateStitchedIP(
                fpga_part,
                clk_ns,
                vitis=False,
            )
        )
        model.set_metadata_prop("exec_mode", "rtlsim")
        model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    ret_sim = execute_onnx(model, input_dict)
    for i, (k, v) in enumerate(ret_sim.items()):
        assert (v == inputs[i]).all()

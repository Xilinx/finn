# Copyright (c) 2020-2022, Xilinx
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
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferDuplicateStreamsLayer,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_dupstreams_modelwrapper(ch, pe, idim, idt, n_dupl, impl_style):
    shape = [1, idim, idim, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    out_names = []
    out_vi = []
    for i in range(n_dupl):
        outp_name = "outp%d" % i
        out_names.append(outp_name)
        out_vi.append(helper.make_tensor_value_info(outp_name, TensorProto.FLOAT, shape))

    dupstrm_node = helper.make_node(
        "DuplicateStreams",
        ["inp"],
        out_names,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        NumOutputStreams=n_dupl,
        PE=pe,
        inputDataType=idt.name,
        numInputVectors=[1, idim, idim],
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(nodes=[dupstrm_node], name="graph", inputs=[inp], outputs=out_vi)

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
# impl_style
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_duplicatestreams(idt, ch, fold, imdim, n_dupl, exec_mode, impl_style):
    if fold == -1:
        pe = 1
    else:
        pe = ch // fold
    assert ch % pe == 0

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, imdim, imdim, ch))

    model = make_dupstreams_modelwrapper(ch, pe, imdim, idt, n_dupl, impl_style)

    # prepare input data and execute
    input_dict = prepare_inputs(x, idt)

    # check behavior of hw abstraction layer
    output_dict = oxe.execute_onnx(model, input_dict)
    expected_y = x
    for i in range(n_dupl):
        y = output_dict["outp%d" % i]
        assert (y == expected_y).all(), "HW layer execution failed"

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

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

    output_dict = oxe.execute_onnx(model, input_dict)

    for i in range(n_dupl):
        y = output_dict["outp%d" % i]
        assert (y == expected_y).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DuplicateStreams_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


@pytest.mark.fpgadataflow
def test_infer_duplicatestreams_with_global_output():
    """Test that InferDuplicateStreamsLayer handles the case where a node output
    is both connected to another node AND is a global output."""

    # Create a model with three Add nodes in a row
    # where the first and third Add nodes' outputs are also global outputs
    ch = 64
    idim = 7
    idt = DataType["INT4"]
    shape = [1, idim, idim, ch]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)

    # Create constant tensors to add
    const1_values = gen_finn_dt_tensor(idt, shape)
    const2_values = gen_finn_dt_tensor(idt, shape)
    const3_values = gen_finn_dt_tensor(idt, shape)

    const1 = helper.make_tensor("const1", TensorProto.FLOAT, shape, const1_values.flatten())
    const2 = helper.make_tensor("const2", TensorProto.FLOAT, shape, const2_values.flatten())
    const3 = helper.make_tensor("const3", TensorProto.FLOAT, shape, const3_values.flatten())

    # First Add node - output is a global output AND connects to second Add
    add1 = helper.make_node(
        "Add",
        ["inp", "const1"],
        ["add1_out"],
    )

    # Second Add node - intermediate node
    add2 = helper.make_node(
        "Add",
        ["add1_out", "const2"],
        ["add2_out"],
    )

    # Third Add node - output is a global output
    add3 = helper.make_node(
        "Add",
        ["add2_out", "const3"],
        ["add3_out"],
    )

    # Make add1_out and add3_out global outputs
    # add1_out is both a global output AND an intermediate connection
    add1_out = helper.make_tensor_value_info("add1_out", TensorProto.FLOAT, shape)
    add2_out = helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, shape)
    add3_out = helper.make_tensor_value_info("add3_out", TensorProto.FLOAT, shape)

    graph = helper.make_graph(
        nodes=[add1, add2, add3],
        name="test_graph",
        inputs=[inp],
        outputs=[add1_out, add3_out],
        initializer=[const1, const2, const3],
        value_info=[add2_out],
    )

    model = qonnx_make_model(graph, producer_name="test-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("const1", idt)
    model.set_tensor_datatype("const2", idt)
    model.set_tensor_datatype("const3", idt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Apply the InferDuplicateStreamsLayer transformation
    model = model.transform(InferDuplicateStreamsLayer())

    # Verify that a DuplicateStreams node was inserted
    assert (
        len(model.get_nodes_by_op_type("DuplicateStreams")) > 0
    ), "DuplicateStreams node was not inserted."

    # Verify execution: test that the model still produces correct outputs
    x = gen_finn_dt_tensor(idt, shape)
    input_dict = {"inp": x}

    output_dict = oxe.execute_onnx(model, input_dict)

    # Compute expected outputs
    expected_add1 = x + const1_values
    expected_add2 = expected_add1 + const2_values
    expected_add3 = expected_add2 + const3_values

    assert (
        output_dict[model.graph.output[0].name] == expected_add1
    ).all(), "First output incorrect."
    assert (
        output_dict[model.graph.output[1].name] == expected_add3
    ).all(), "Second output incorrect."

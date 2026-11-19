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
@pytest.mark.parametrize("idt", [DataType["INT4"], DataType["FLOAT32"]])
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


# Test scenarios with different numbers of successors and global outputs
# Tuples: (split_source, num_successors, num_global_outputs)
# split_source: "global_input" or "first_node"
@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "split_source,num_successors,num_global_outputs",
    [
        ("global_input", 2, 0),  # Global input splits into 2 successors
        ("global_input", 5, 0),  # Global input splits into 5 successors
        ("first_node", 1, 1),  # First node splits into 1 successor + 1 global output
        ("first_node", 3, 1),  # First node splits into 3 successors + 1 global output
        ("first_node", 4, 0),  # First node splits into 4 successors
    ],
)
def test_infer_duplicatestreams(split_source, num_successors, num_global_outputs):
    """Test that InferDuplicateStreamsLayer handles various fanout scenarios."""

    ch = 64
    idim = 7
    idt = DataType["INT4"]
    shape = [1, idim, idim, ch]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)

    nodes = []
    initializers = []
    outputs = []
    value_infos = []
    const_values_list = []

    if split_source == "global_input":
        # Create multiple successor nodes that consume the global input
        for i in range(num_successors):
            const_val = gen_finn_dt_tensor(idt, shape)
            const_values_list.append(const_val)
            const = helper.make_tensor(f"const{i}", TensorProto.FLOAT, shape, const_val.flatten())
            initializers.append(const)

            add_node = helper.make_node("Add", ["inp", f"const{i}"], [f"out{i}"])
            nodes.append(add_node)

            out_info = helper.make_tensor_value_info(f"out{i}", TensorProto.FLOAT, shape)
            outputs.append(out_info)

    elif split_source == "first_node":
        # Create first node
        const0_val = gen_finn_dt_tensor(idt, shape)
        const_values_list.append(const0_val)
        const0 = helper.make_tensor("const0", TensorProto.FLOAT, shape, const0_val.flatten())
        initializers.append(const0)

        add0 = helper.make_node("Add", ["inp", "const0"], ["add0_out"])
        nodes.append(add0)

        # Create successor nodes that consume add0_out
        for i in range(num_successors):
            const_val = gen_finn_dt_tensor(idt, shape)
            const_values_list.append(const_val)
            const = helper.make_tensor(f"const{i+1}", TensorProto.FLOAT, shape, const_val.flatten())
            initializers.append(const)

            add_node = helper.make_node("Add", ["add0_out", f"const{i+1}"], [f"out{i}"])
            nodes.append(add_node)

            out_info = helper.make_tensor_value_info(f"out{i}", TensorProto.FLOAT, shape)
            outputs.append(out_info)

        # Add add0_out as global output if specified
        if num_global_outputs > 0:
            add0_out_info = helper.make_tensor_value_info("add0_out", TensorProto.FLOAT, shape)
            outputs.insert(0, add0_out_info)  # Insert at beginning

    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[inp],
        outputs=outputs,
        initializer=initializers,
        value_info=value_infos,
    )

    model = qonnx_make_model(graph, producer_name="test-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    for i, const in enumerate(initializers):
        model.set_tensor_datatype(const.name, idt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Apply the InferDuplicateStreamsLayer transformation
    model = model.transform(InferDuplicateStreamsLayer())

    # Verify that a DuplicateStreams node was inserted
    dup_nodes = model.get_nodes_by_op_type("DuplicateStreams")
    assert len(dup_nodes) == 1, f"Expected 1 DuplicateStreams node, got {len(dup_nodes)}"

    # Verify execution
    x = gen_finn_dt_tensor(idt, shape)
    input_dict = {"inp": x}

    output_dict = oxe.execute_onnx(model, input_dict)

    # Verify outputs
    if split_source == "global_input":
        for i in range(num_successors):
            expected = x + const_values_list[i]
            assert (output_dict[f"out{i}"] == expected).all(), f"Output out{i} incorrect"

    else:  # first_node
        add0_result = x + const_values_list[0]

        if num_global_outputs > 0:
            # First output should be add0_out
            assert (
                output_dict[model.graph.output[0].name] == add0_result
            ).all(), "add0_out global output incorrect"

        # Remaining outputs are the successor results
        for i in range(num_successors):
            expected = add0_result + const_values_list[i + 1]
            assert (output_dict[f"out{i}"] == expected).all(), f"Output out{i} incorrect"

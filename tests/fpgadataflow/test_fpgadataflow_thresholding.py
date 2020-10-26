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
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.multithreshold import multithreshold
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import gen_finn_dt_tensor
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
import os
from finn.util.pyverilator import axilite_read
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.core.rtlsim_exec import rtlsim_exec

test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 5


def make_single_thresholding_modelwrapper(T, pe, idt, odt, actval, mem_mode):
    NumChannels = T.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, NumChannels])

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        PE=pe,
        numSteps=T.shape[1],
        inputDataType=idt.name,
        weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
        outputDataType=odt.name,
        ActVal=actval,
        mem_mode=mem_mode,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model


# activation: None or DataType
@pytest.mark.parametrize("act", [DataType.INT4, DataType.BIPOLAR])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT16, DataType.UINT16])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# number of input features
@pytest.mark.parametrize("ich", [16])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# memory mode
@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_thresholding(idt, act, nf, ich, exec_mode, mem_mode):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, ich))

    odt = act
    n_steps = act.get_num_possible_values() - 1
    T = np.random.randint(idt.min(), idt.max() + 1, (ich, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T = np.sort(T, axis=1)

    if odt == DataType.BIPOLAR:
        actval = 0
    else:
        actval = odt.min()

    model = make_single_thresholding_modelwrapper(T, pe, idt, odt, actval, mem_mode)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # package input data as dictionary
    input_dict = {"inp": x}

    y = multithreshold(x, T)
    if act == DataType.BIPOLAR:
        # binary to bipolar
        y = 2 * y - 1
    else:
        # signed offset
        y += act.min()

    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "cppsim failed"

    if exec_mode == "rtlsim":
        hls_synt_res_est = model.analysis(hls_synth_res_estimation)
        assert "Thresholding_Batch_0" in hls_synt_res_est

        node = model.get_nodes_by_op_type("Thresholding_Batch")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


@pytest.mark.vivado
def test_runtime_thresholds_single_layer():
    mem_mode = "decoupled"
    act = DataType.UINT2
    idt = DataType.UINT8
    nf = 2
    ich = 16
    pe = ich // nf
    assert ich % pe == 0

    # generate input data
    in_tensor = gen_finn_dt_tensor(idt, (1, ich))

    odt = act
    n_steps = act.get_num_possible_values() - 1
    T = np.random.randint(idt.min(), idt.max() + 1, (ich, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T = np.sort(T, axis=1)

    if odt == DataType.BIPOLAR:
        actval = 0
    else:
        actval = odt.min()

    model = make_single_thresholding_modelwrapper(T, pe, idt, odt, actval, mem_mode)
    op_inst = getCustomOp(model.graph.node[0])
    op_inst.set_nodeattr("runtime_writeable_weights", 1)
    op_inst.make_weight_file(T, "decoupled_runtime", "old_weights.dat")
    with open("old_weights.dat", "r") as f:
        old_weight_stream = f.read().strip()
    os.remove("old_weights.dat")
    old_weight_stream = map(lambda x: int(x, 16), old_weight_stream.split("\n"))
    old_weight_stream = list(old_weight_stream)
    # need to create stitched IP for runtime weight testing
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model.set_metadata_prop("exec_mode", "rtlsim")
    # add two copies of the input tensor as the first one is just used to
    # "flush out" the pipeline (as mvau already starts receiving old weights while
    # we read/write new ones and reads seem to cause a disturbance too)
    in_tensor = np.tile(in_tensor, (2, 1))
    exec_ctx = {"inp": in_tensor}
    extracted_weight_stream = []

    def read_weights(sim):
        addr = 0
        for i in range(len(old_weight_stream)):
            extracted_weight_stream.append(
                axilite_read(sim, addr, basename="s_axilite_0_")
            )
            addr += 4

    rtlsim_exec(model, exec_ctx, pre_hook=read_weights)
    assert extracted_weight_stream == old_weight_stream
    y = exec_ctx["outp"]
    # only use second batch element in output; first will be invalid due to
    # old weights (see above)
    expected = multithreshold(in_tensor, T)[1]
    if act == DataType.BIPOLAR:
        # binary to bipolar
        expected = 2 * expected - 1
    else:
        # signed offset
        expected += act.min()
    assert (y[1] == expected).all()

# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import os
from onnx import TensorProto, helper
from pyverilator.util.axi_utils import axilite_read, axilite_write
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


def generate_random_threshold_values(input_data_type, num_input_channels, num_steps):
    return np.random.randint(
        input_data_type.min(),
        input_data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


# n = batch, c = channel, h = height, w = width of feature map
# Standard = NCHW; FINN = NHWC
# Convert from NHWC(FINN) to NCHW(Standard)
def layout_FINN2NCHW(data):
    return np.transpose(data, (0, 3, 1, 2))


# Convert from NCHW(Standard) to NHWC(FINN)
def layout_NCHW2FINN(data):
    return np.transpose(data, (0, 2, 3, 1))


def make_single_thresholding_modelwrapper(impl_style, T, idt, odt, actval, n_inp_vecs):
    NumChannels = T.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, n_inp_vecs + [NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, n_inp_vecs + [NumChannels])

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        numSteps=T.shape[1],
        inputDataType=idt.name,
        weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
        outputDataType=odt.name,
        ActVal=actval,
        numInputVectors=n_inp_vecs,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model


# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT4"], DataType["BIPOLAR"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT16"], DataType["UINT16"]])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# number of input features
@pytest.mark.parametrize("ich", [16])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# memory mode
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.parametrize("impl_style", ["rtl", "hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_thresholding(impl_style, idt, act, nf, ich, exec_mode, mem_mode):
    # the mem_mode parameter can only be used for the hls thresholding
    # so the test will only be executed once for impl_style=rtl and once skipped
    # when the mem_mode is varied. Otherwise, the same test configuration would always
    # run twice.
    if impl_style == "rtl" and mem_mode == "internal_decoupled":
        pytest.skip(
            "Skip, because test is identical to impl_style=rtl and mem_mode=internal_embedded"
        )
    if nf == -1:
        nf = ich
    pe = ich // nf
    n_inp_vecs = [1, 2, 2]
    assert ich % pe == 0

    # generate input data, data layout is NHWC for FINN
    x = gen_finn_dt_tensor(idt, tuple(n_inp_vecs + [ich]))

    odt = act
    n_steps = act.get_num_possible_values() - 1

    # Generate random, non-decreasing thresholds
    thresholds = generate_random_threshold_values(idt, ich, n_steps)

    thresholds = sort_thresholds_increasing(thresholds)

    if odt == DataType["BIPOLAR"]:
        actval = 0
    else:
        actval = odt.min()

    # Build DUT
    model = make_single_thresholding_modelwrapper(
        impl_style, thresholds, idt, odt, actval, n_inp_vecs
    )

    # Expected Reference output
    # multithreshold util fxn wants NCHW input, not NHWC
    x_nchw = layout_FINN2NCHW(x)
    y = multithreshold(x_nchw, thresholds)

    # convert back to NHWC for comparison to hw outputs
    y = layout_NCHW2FINN(y)
    if act == DataType["BIPOLAR"]:
        # binary to bipolar
        y = 2 * y - 1
    else:
        # signed offset
        y += act.min()

    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)

    # package input data as dictionary
    input_dict = {"inp": x}

    # execute DUT
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all()

    model = model.transform(SpecializeLayers())
    # Make sure that SpecializeLayers did not default to HLS implementation unexpectedly
    assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    if impl_style == "hls":
        inst.set_nodeattr("mem_mode", mem_mode)

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

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        if impl_style == "hls":
            hls_synt_res_est = model.analysis(hls_synth_res_estimation)
            assert model.graph.node[0].name in hls_synt_res_est
        node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


@pytest.mark.parametrize("impl_style", ["rtl", "hls"])
# configuration (ch, pe)
@pytest.mark.parametrize("cfg", [(1, 1), (6, 2), (6, 3), (8, 4)])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_runtime_thresholds_read(impl_style, cfg):
    """Read back threshold weights during runtime

    1. Create random initial weights T
    2. Execute model
    3. Read back weights via AXI
    4. Compare with initial weights T
    """
    ch = cfg[0]
    pe = cfg[1]
    n_inp_vecs = [1, 2, 2]
    hls_mem_mode = "internal_decoupled"
    act = DataType["INT4"]
    idt = DataType["INT16"]
    odt = act
    n_steps = act.get_num_possible_values() - 1
    np.random.seed(2)
    T = np.random.randint(idt.min(), idt.max() + 1, (ch, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T = np.sort(T, axis=1)

    if odt == DataType["BIPOLAR"]:
        actval = 0
    else:
        actval = odt.min()

    model = make_single_thresholding_modelwrapper(impl_style, T, idt, odt, actval, n_inp_vecs)
    model = model.transform(SpecializeLayers())

    # Make sure that specialize layer did not default to HLS implementation
    assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)

    node = model.get_nodes_by_op_type(f"Thresholding_{impl_style}")[0]
    op_inst = getCustomOp(node)
    op_inst.set_nodeattr("PE", pe)
    if impl_style == "hls":
        op_inst.set_nodeattr("mem_mode", hls_mem_mode)
    op_inst.set_nodeattr("runtime_writeable_weights", 1)

    dat_fname = f"old_weights_{cfg}.dat"
    op_inst.make_weight_file(T, "decoupled_runtime", dat_fname)
    with open(dat_fname, "r") as f:
        old_weight_stream = f.read().strip()
    os.remove(dat_fname)
    old_weight_stream = map(lambda x: int(x, 16), old_weight_stream.split("\n"))
    old_weight_stream = list(old_weight_stream)
    # need to create stitched IP for runtime weight testing
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model.set_metadata_prop("exec_mode", "rtlsim")
    # add two copies of the input tensor as the first one is just used to
    # "flush out" the pipeline (as mvau already starts receiving old weights while
    # we read/write new ones and reads seem to cause a disturbance too)
    # generate input data
    in_tensor = gen_finn_dt_tensor(idt, tuple(n_inp_vecs + [ch]))
    in_tensor = np.tile(in_tensor, (2, 1, 1, 1))

    exec_ctx = {"inp": in_tensor}
    extracted_weight_stream = []

    def read_weights(sim):
        addr = 0
        for i in range(len(old_weight_stream)):
            extracted_weight_stream.append(axilite_read(sim, addr, basename="s_axilite_0_"))
            addr += 4

    rtlsim_exec(model, exec_ctx, pre_hook=read_weights)

    # Validate the AXI Read weights
    assert extracted_weight_stream == old_weight_stream

    y = exec_ctx["outp"][0]

    # multithreshold util fxn wants NCHW input, not NHWC
    expected = multithreshold(np.transpose(in_tensor, (0, 3, 1, 2)), T)
    # convert back to NHWC for comparison to hw outputs
    expected = np.transpose(expected, (0, 2, 3, 1))[1]

    if act == DataType["BIPOLAR"]:
        # binary to bipolarW
        expected = 2 * expected - 1
    else:
        # signed offset
        expected += act.min()

    # Validate the output is as expected
    assert (y == expected).all()


@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
# configuration (ch, pe)
@pytest.mark.parametrize("cfg", [(1, 1), (6, 2), (6, 3), (8, 4)])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_runtime_thresholds_write(impl_style, cfg):
    """Write threshold weights during runtime

    1. Create random initial weights T_init
    2. Create model with initial weights
    3. Create new set of weights T_write
    4. Write T_write using AXI bus
    5. Read back using AXI bus to T_read
    6. Compare T_write and T_read
    7. Validate outputs with expected vectors
    """
    ch = cfg[0]
    pe = cfg[1]

    n_inp_vecs = [1, 2, 2]
    hls_mem_mode = "internal_decoupled"
    act = DataType["INT4"]
    idt = DataType["INT16"]

    odt = act
    n_steps = act.get_num_possible_values() - 1
    np.random.seed(2)
    T_init = np.random.randint(idt.min(), idt.max() + 1, (ch, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T_init = np.sort(T_init, axis=1)

    if odt == DataType["BIPOLAR"]:
        actval = 0
    else:
        actval = odt.min()

    model = make_single_thresholding_modelwrapper(impl_style, T_init, idt, odt, actval, n_inp_vecs)
    model = model.transform(SpecializeLayers())

    # Validate that specialize layer did not default to HLS implementation
    assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)

    op_inst = getCustomOp(model.graph.node[0])
    op_inst.set_nodeattr("PE", pe)
    if impl_style == "hls":
        op_inst.set_nodeattr("mem_mode", hls_mem_mode)
    op_inst.set_nodeattr("runtime_writeable_weights", 1)

    # Make new weights for runtime write
    np.random.seed(4)
    T_write = np.random.randint(idt.min(), idt.max() + 1, (ch, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T_write = np.sort(T_write, axis=1)

    dat_fname = f"T_write_{cfg}.dat"  # distinguish fname per paramter for distributed testing
    op_inst.make_weight_file(T_write, "decoupled_runtime", dat_fname)
    with open(dat_fname, "r") as f:
        T_write_stream = f.read().strip()
    os.remove(dat_fname)

    T_write_stream = map(lambda x: int(x, 16), T_write_stream.split("\n"))
    T_write_stream = list(T_write_stream)

    # need to create stitched IP for runtime weight testing
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model.set_metadata_prop("exec_mode", "rtlsim")
    # add two copies of the input tensor as the first one is just used to
    # "flush out" the pipeline (as mvau already starts receiving old weights while
    # we read/write new ones and reads seem to cause a disturbance too)
    # generate input data
    in_tensor = gen_finn_dt_tensor(idt, tuple(n_inp_vecs + [ch]))
    in_tensor = np.tile(in_tensor, (2, 1, 1, 1))

    exec_ctx_write = {"inp": in_tensor}

    def write_weights(sim):
        addr = 0
        for nw in T_write_stream:
            axilite_write(sim, addr, nw, basename="s_axilite_0_")
            addr += 4

    T_read_stream = []

    def read_weights(sim):
        addr = 0
        for i in range(len(T_write_stream)):
            T_read_stream.append(axilite_read(sim, addr, basename="s_axilite_0_"))
            addr += 4

    rtlsim_exec(model, exec_ctx_write, pre_hook=write_weights, post_hook=read_weights)

    y = exec_ctx_write["outp"][1]

    assert T_read_stream == T_write_stream

    # multithreshold util fxn wants NCHW input, not NHWC
    expected = multithreshold(np.transpose(in_tensor, (0, 3, 1, 2)), T_write)
    # convert back to NHWC for comparison to hw outputs
    expected = np.transpose(expected, (0, 2, 3, 1))[1]

    if act == DataType["BIPOLAR"]:
        # binary to bipolarW
        expected = 2 * expected - 1
    else:
        # signed offset
        expected += act.min()

    # Validate the output is as expected
    assert (y == expected).all()

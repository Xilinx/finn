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
from pyverilator.util.axi_utils import axilite_write, reset_rtlsim
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


# Helper functions
def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


def generate_random_threshold_values(input_data_type, num_input_channels, num_steps):
    return np.random.randint(
        input_data_type.min(),
        input_data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def generate_pe_value(fold, num_input_channels):
    if fold == -1:
        fold = num_input_channels
    pe = num_input_channels // fold
    assert num_input_channels % pe == 0
    return pe


# n = batch, c = channel, h = height, w = width of feature map
# Standard = NCHW; FINN = NHWC
# Convert from NCHW to NHWC
def convert_np_array_to_finn_data_layout(data):
    return np.transpose(data, (0, 2, 3, 1))


# n = batch, c = channel, h = height, w = width of feature map
# Standard = NCHW; FINN = NHWC
# Convert from NHWC to NCHW
def convert_np_array_to_standard_data_layout(data):
    return np.transpose(data, (0, 3, 1, 2))


def make_single_thresholding_binary_search_modelwrapper(
    thresholds,
    pe,
    input_data_type,
    output_data_type,
    activation_bias,
    mem_mode,
    num_input_vecs,
):
    NumChannels = thresholds.shape[0]

    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, num_input_vecs + [NumChannels]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, num_input_vecs + [NumChannels]
    )

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding_Binary_Search",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        PE=pe,
        numSteps=thresholds.shape[1],
        inputDataType=input_data_type.name,
        weightDataType=input_data_type.name,
        outputDataType=output_data_type.name,
        activation_bias=activation_bias,
        mem_mode=mem_mode,
        numInputVectors=num_input_vecs,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", input_data_type)
    model.set_tensor_datatype("outp", output_data_type)

    model.set_tensor_datatype("thresh", input_data_type)
    model.set_initializer("thresh", thresholds)
    return model


# Test brief: a particular method for this class was causing a bug:
# find_next_power_of_2()
# Weights in the thresholding core are programmed on a per-channel basis and are
# byte-addressable. When a channel is programmed, the next channel can start
# programming at the next power-of-2 byte boundary. This test is to show that the
# function that calculates that boundary is working correctly.
#
# A Thresholding_Binary_Search layer was created and a SW generated dataset with a
# threshold channel depth of 1 weight (1 layer of N channels in the thresholding core).
# However, find_next_power_of_2() was returning a next-power-of-2 address boundary at
# address '0', instead of '2'. This unit test is to prove that this bug no longer
# occurs. It was originally seen when the input datatype was 'DataType["BIPOLAR"]'.
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_thresholding_binary_search_unit():
    activation = DataType["BIPOLAR"]
    input_data_type = DataType["INT16"]
    fold = -1
    num_input_channels = 16
    mem_mode = "decoupled"

    # Handle inputs to the test
    pe = generate_pe_value(fold, num_input_channels)
    num_steps = activation.get_num_possible_values() - 1

    # Other non-input parameters
    num_input_vecs = [1, 2, 2]
    output_data_type = activation
    if output_data_type == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = output_data_type.min()

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(
        input_data_type, num_input_channels, num_steps
    )

    # Generate model from input parameters to the test
    model = make_single_thresholding_binary_search_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        mem_mode,
        num_input_vecs,
    )

    # Retrieve the class to get the method-under-test
    tbs_node = model.get_nodes_by_op_type("Thresholding_Binary_Search")[0]
    tbs_inst = getCustomOp(tbs_node)

    test_vector = [
        {"input": -2, "expected_result": 0},
        {"input": -1, "expected_result": 0},
        {"input": 0, "expected_result": 0},
        {"input": 1, "expected_result": 2},
        {"input": 2, "expected_result": 2},
        {"input": 3, "expected_result": 4},
        {"input": 4, "expected_result": 4},
        {"input": 7, "expected_result": 8},
        {"input": 8, "expected_result": 8},
        {"input": 11, "expected_result": 16},
        {"input": 15, "expected_result": 16},
        {"input": 16, "expected_result": 16},
        {"input": 18, "expected_result": 32},
        {"input": 27, "expected_result": 32},
        {"input": 31, "expected_result": 32},
        {"input": 32, "expected_result": 32},
        {"input": 42, "expected_result": 64},
        {"input": 65, "expected_result": 128},
    ]

    for test_dict in test_vector:
        output = tbs_inst.find_next_power_of_2(test_dict["input"])
        assert output >= test_dict["input"]
        assert output == test_dict["expected_result"]

    return


# Test brief: Prove that cppsim is not supported for this class
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_thresholding_binary_search_cppsim():
    input_data_type = DataType["UINT16"]
    act = DataType["BIPOLAR"]
    fold = -1
    num_input_channels = 16
    # 'const' is unsupported see test:
    # test_fpgadataflow_thresholding_binary_search_const_mem_mode()
    mem_mode = "decoupled"

    pe = generate_pe_value(fold, num_input_channels)
    num_steps = act.get_num_possible_values() - 1

    # Generate random, non-decreasing thresholds
    thresholds = generate_random_threshold_values(
        input_data_type, num_input_channels, num_steps
    )

    # make the vivado_hls threshold bug appear (incorrect rtlsim result when first
    # threshold of first channel is zero, while using BIPOLAR output)
    if act == DataType["BIPOLAR"]:
        thresholds[0][0] = 0
    thresholds = sort_thresholds_increasing(thresholds)

    # Other non-input parameters
    num_input_vecs = [1, 2, 2]
    output_data_type = act
    if output_data_type == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = output_data_type.min()

    # Generate model from input parameters to the test
    model = make_single_thresholding_binary_search_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        mem_mode,
        num_input_vecs,
    )

    # Cppsim is not supported for this class, catch the specific exception thrown by
    # cppsim. Exception raised in cppsim: Custom op_type Thresholding_Binary_Search is
    # currently not supported.
    try:
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    except Exception as e:
        if (
            str(e)
            != "Custom op_type Thresholding_Binary_Search is currently not supported."
        ):
            raise


# Test brief: Prove that memory mode 'const' is not supported for this layer type
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_thresholding_binary_search_const_mem_mode():
    input_data_type = DataType["INT16"]
    activation = DataType["INT4"]
    fold = -1
    num_input_channels = 16
    mem_mode = "const"

    pe = generate_pe_value(fold, num_input_channels)
    num_input_vecs = [1, 2, 2]
    output_data_type = activation
    activation_bias = output_data_type.min()

    # Generate random thresholds and sort in ascending order
    num_steps = activation.get_num_possible_values() - 1
    thresholds = generate_random_threshold_values(
        input_data_type, num_input_channels, num_steps
    )

    # Generate model from input parameters to the test
    model = make_single_thresholding_binary_search_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        mem_mode,
        num_input_vecs,
    )

    # Prove that 'const' memory mode is not supported for this class
    # 'const' memory mode is not supported for this class, catch the specific exception
    # thrown by FINN. Exception: ('Unrecognized memory mode for this node:', 'const')
    try:
        model = model.transform(InsertFIFO(True))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    except Exception as e:
        if str(e) != "Unrecognized memory mode for this node: {}".format(mem_mode):
            raise
        # Caught the expected exception, leave the test early
        return


# Test brief: Test that PrepareRTLSim() runs successfully. This function is not
# tested in test_fpgadataflow_thresholding_binary_search()
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_thresholding_binary_search_prepare_rtlsim():
    input_data_type = DataType["INT16"]
    act = DataType["INT4"]
    fold = -1
    num_input_channels = 16
    mem_mode = "decoupled"

    # Handle inputs to the test
    pe = generate_pe_value(fold, num_input_channels)
    num_steps = act.get_num_possible_values() - 1

    # Generate random, non-decreasing thresholds
    thresholds = generate_random_threshold_values(
        input_data_type, num_input_channels, num_steps
    )
    # make the vivado_hls threshold bug appear (incorrect rtlsim result when first
    # threshold of first channel is zero, while using BIPOLAR output)
    if act == DataType["BIPOLAR"]:
        thresholds[0][0] = 0
    thresholds = sort_thresholds_increasing(thresholds)

    # Other non-input parameters
    num_input_vecs = [1, 2, 2]
    output_data_type = act
    if output_data_type == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = output_data_type.min()

    # Generate model from input parameters to the test
    model = make_single_thresholding_binary_search_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        mem_mode,
        num_input_vecs,
    )

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    return


# Test brief: Create a Thresholding binary search layer using various parameters
# and test against a SW generated & simulated dataset
# N.B. - fold factor of '-1' is supported only (no PE/SIMD support)
@pytest.mark.parametrize("activation", [DataType["INT4"], DataType["BIPOLAR"]])
@pytest.mark.parametrize("input_data_type", [DataType["INT16"], DataType["UINT16"]])
@pytest.mark.parametrize("fold", [-1])  # 1, 2, etc. will fail
@pytest.mark.parametrize("num_input_channels", [16])
# no need to test 'const' mode, it's already done in:
# test_fpgadataflow_thresholding_binary_search_const_mem_mode()
@pytest.mark.parametrize("mem_mode", ["decoupled"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_thresholding_binary_search(
    activation, input_data_type, fold, num_input_channels, mem_mode
):
    # Handle inputs to the test
    pe = generate_pe_value(fold, num_input_channels)
    num_steps = activation.get_num_possible_values() - 1

    # Other non-input parameters
    num_input_vecs = [1, 2, 2]
    output_data_type = activation
    if output_data_type == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = output_data_type.min()

    # generate random input data
    tensor_shape = tuple(num_input_vecs + [num_input_channels])
    x = gen_finn_dt_tensor(input_data_type, tensor_shape)

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(
        input_data_type, num_input_channels, num_steps
    )

    # make the vivado_hls threshold bug appear (incorrect rtlsim result when first
    # threshold of first channel is zero, while using BIPOLAR output)
    if activation == DataType["BIPOLAR"]:
        thresholds[0][0] = 0

    # provide non-decreasing/ascending thresholds
    thresholds = sort_thresholds_increasing(thresholds)

    x_nhwc = convert_np_array_to_standard_data_layout(x)
    y = multithreshold(x_nhwc, thresholds)

    # convert back to NHWC for comparison to hw outputs
    y = convert_np_array_to_finn_data_layout(y)
    if activation == DataType["BIPOLAR"]:
        # binary to bipolar
        y = 2 * y - 1
    else:
        # signed offset
        y += activation.min()

    # Generate model from input parameters to the test
    model = make_single_thresholding_binary_search_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        mem_mode,
        num_input_vecs,
    )

    model = model.transform(InsertFIFO(True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    # Retrieve the axilite programming sequence for weights - for decoupled mode only
    tbs_node = model.get_nodes_by_op_type("Thresholding_Binary_Search")[0]
    tbs_inst = getCustomOp(tbs_node)
    config = tbs_inst.get_dynamic_config(model)

    # Reshape generated data (not from model)
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)

    # Helper function that delivers the hook to program the thresholds via AXI-Lite
    def config_hook(config):
        if config is None:
            return None

        def write_thresh_config(sim):
            # axi_name = "s_axilite_0_" # works
            axi_name = getCustomOp(
                model.get_nodes_by_op_type("Thresholding_Binary_Search")[0]
            ).get_verilog_top_module_intf_names()["axilite"][0]
            axi_name += "_0_"

            # Write config registers to the Threshold memory.
            # The dictionary defines (addr, value) tuples.
            for config_entry in config.values():
                addr = config_entry[0]
                val = config_entry[1]
                axilite_write(sim, addr, val, basename=axi_name)

            reset_rtlsim(sim)

        return write_thresh_config

    input_dict = {"inp": x}
    rtlsim_exec(model, input_dict, pre_hook=config_hook(config))
    y_produced = input_dict["outp"]
    assert (y_produced == y_expected).all()

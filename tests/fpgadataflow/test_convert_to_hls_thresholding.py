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
from pyverilator.util.axi_utils import axilite_write, reset_rtlsim
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

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
        numInputVectors=num_input_vecs,
        activation_bias=activation_bias,
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


def make_single_multithresholding_modelwrapper(
    thresholds,
    pe,
    input_data_type,
    output_data_type,
    activation_bias,
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

    Multithresholding_node = helper.make_node(
        "MultiThreshold",
        node_inp_list,
        ["outp"],
        domain="qonnx.custom_op.general",
        out_dtype=output_data_type.name,
        out_bias=float(activation_bias),
        out_scale=1.0,
    )

    graph = helper.make_graph(
        nodes=[Multithresholding_node],
        name="multithresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="multithresholding-model")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model.set_tensor_datatype("inp", input_data_type)
    model.set_tensor_datatype("outp", output_data_type)

    model.set_tensor_datatype("thresh", input_data_type)
    model.set_initializer("thresh", thresholds)
    return model


@pytest.mark.parametrize("activation", [DataType["INT4"], DataType["BIPOLAR"]])
@pytest.mark.parametrize("input_data_type", [DataType["INT16"], DataType["UINT16"]])
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("num_input_channels", [16])
@pytest.mark.parametrize("mem_mode", ["decoupled", "const"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_convert_to_hls_tbs_rtl_variant(
    activation, input_data_type, fold, num_input_channels, mem_mode
):
    # Handle inputs to the test
    pe = generate_pe_value(fold, num_input_channels)
    num_steps = activation.get_num_possible_values() - 1

    # Cppsim is not supported for this node (as it is an RTL node)
    if mem_mode == "const":
        pytest.skip(
            "const memory mode not supported for "
            "RTL Thresholding Binary Search node"
        )
    elif mem_mode != "decoupled":
        raise Exception("Unknown mem_mode: {}".format(mem_mode))

    if activation == DataType["BIPOLAR"]:
        pytest.skip(
            "Only negative activations are supported for "
            "RTL Thresholding Binary Search node"
        )

    # Paralellisation not supported for thresholding binary search rtl node
    if pe != 1:
        pytest.skip(
            "Paralellisation of IP not supported for RTL Thresholding Binary Search node"
        )

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

    # Make a Multithreshold graph and convert to thresholding binary search node
    new_model = make_single_multithresholding_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
    )

    # Recreate the model using the ConvertToHLS transform
    new_model = new_model.transform(
        to_hls.InferThresholdingLayer(mem_mode=mem_mode, use_rtl_variant=True)
    )
    new_model = new_model.transform(InsertFIFO(True))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    new_model = new_model.transform(HLSSynthIP())
    new_model = new_model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    input_dict = {"inp": x}
    rtlsim_exec(new_model, input_dict, pre_hook=config_hook(config))
    y_produced_new = input_dict["outp"]
    assert (y_produced_new == y_expected).all()

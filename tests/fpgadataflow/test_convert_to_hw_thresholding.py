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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


# Helper functions
def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# n = batch, c = channel, h = height, w = width of feature map
# Standard = NCHW; FINN = NHWC
# Convert from NHWC(FINN) to NCHW(Standard)
def layout_FINN2NCHW(data):
    return np.transpose(data, (0, 3, 1, 2))


# Convert from NCHW(Standard) to NHWC(FINN)
def layout_NCHW2FINN(data):
    return np.transpose(data, (0, 2, 3, 1))


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


def make_single_multithresholding_modelwrapper(
    thresholds,
    pe,
    input_data_type,
    output_data_type,
    activation_bias,
    num_input_vecs,
):
    NumChannels = thresholds.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, num_input_vecs + [NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [NumChannels])

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


# N.B. Fold values where C % PE != 0 fail
@pytest.mark.parametrize("activation", [DataType["INT4"], DataType["BIPOLAR"]])
@pytest.mark.parametrize("input_data_type", [DataType["INT16"], DataType["UINT16"]])
@pytest.mark.parametrize("fold", [-1, 1, 2, 4, 6])
@pytest.mark.parametrize("num_input_channels", [16])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_convert_multithreshold_to_hardware(
    impl_style,
    activation,
    input_data_type,
    fold,
    num_input_channels,
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

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(input_data_type, num_input_channels, num_steps)

    # provide non-decreasing/ascending thresholds
    thresholds = sort_thresholds_increasing(thresholds)

    # Make a Multithreshold graph and convert to thresholding binary search node
    model = make_single_multithresholding_modelwrapper(
        thresholds,
        pe,
        input_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
    )

    model = model.transform(InferThresholdingLayer())

    # Perform functional validation of the InferThresholdingLayer transform
    x = gen_finn_dt_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))

    x_nchw = layout_FINN2NCHW(x)
    y_expected = multithreshold(x_nchw, thresholds)

    # convert back to NHWC for comparison to hw outputs
    y_expected = layout_NCHW2FINN(y_expected)
    if activation == DataType["BIPOLAR"]:
        # binary to bipolar
        y_expected = 2 * y_expected - 1
    else:
        # signed offset
        y_expected += activation.min()

    input_dict = prepare_inputs(x)
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    assert (y_produced == y_expected).all()

    # Transform to the specified implementation style, either the
    # RTL or HLS according to test parameters
    node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("preferred_impl_style", impl_style)
    model = model.transform(SpecializeLayers())
    model = model.transform(InferShapes())
    assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)

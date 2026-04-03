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
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


def generate_edge_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    """Generate threshold values that include edge cases (min/max of datatype range)."""
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1

    # Use gen_finn_dt_tensor to generate valid values for the datatype
    thresholds = gen_finn_dt_tensor(data_type, (num_input_channels, num_steps))

    # Get min and max for this datatype
    dt_min = data_type.min()
    dt_max = data_type.max()

    # Replace first and last threshold per channel with min and max
    # if num_steps >=2
    if num_steps >= 2:
        for ch in range(num_input_channels):
            thresholds[ch, 0] = dt_min
            thresholds[ch, -1] = dt_max

    return thresholds.astype(np.float32)


def generate_edge_input_tensor(data_type, shape):
    """Generate input tensor that includes edge cases (min/max of datatype range)."""
    # Use gen_finn_dt_tensor to generate valid values for the datatype
    values = gen_finn_dt_tensor(data_type, shape)

    # Flatten to easily replace some values
    flat_values = values.flatten()
    total_elements = len(flat_values)

    # Get min and max for this datatype
    dt_min = data_type.min()
    dt_max = data_type.max()

    # Replace some values with min and max
    num_edge_values = max(1, min(total_elements // 4, 10))

    # Set first few elements to min
    flat_values[:num_edge_values] = dt_min
    # Set next few elements to max
    flat_values[num_edge_values : 2 * num_edge_values] = dt_max

    # Shuffle to distribute edge cases throughout
    np.random.shuffle(flat_values)

    return flat_values.reshape(shape)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


def make_single_multithresholding_modelwrapper(
    thresholds,
    input_data_type,
    threshold_data_type,
    output_data_type,
    activation_bias,
    num_input_vecs,
    num_channels,
):
    if input_data_type == DataType["FLOAT16"]:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT16, num_input_vecs + [num_channels]
        )
    else:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, num_input_vecs + [num_channels]
        )
    if threshold_data_type == DataType["FLOAT16"]:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT16, thresholds.shape)
    else:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, thresholds.shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [num_channels])

    node_inp_list = ["inp", "thresh"]

    Multithresholding_node = helper.make_node(
        "MultiThreshold",
        node_inp_list,
        ["outp"],
        domain="qonnx.custom_op.general",
        out_dtype=output_data_type.name,
        out_bias=float(activation_bias),
        out_scale=1.0,
        data_layout="NHWC",
    )

    graph = helper.make_graph(
        nodes=[Multithresholding_node],
        name="multithresholding_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[thresh],
    )

    model = helper.make_model(graph, producer_name="multithresholding-model")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model.set_tensor_datatype("inp", input_data_type)
    model.set_tensor_datatype("outp", output_data_type)

    model.set_tensor_datatype("thresh", threshold_data_type)
    model.set_initializer("thresh", thresholds)
    return model


@pytest.mark.parametrize("num_input_channels", [6, 16])
@pytest.mark.parametrize(
    "num_input_vecs",
    [
        [1],
        [1, 2, 2],
    ],
)
@pytest.mark.parametrize("activation", [DataType["UINT4"], DataType["INT4"], DataType["BIPOLAR"]])
@pytest.mark.parametrize(
    "idt_tdt_cfg",
    [
        (DataType["INT8"], DataType["INT25"]),
        (DataType["UINT5"], DataType["UINT8"]),
        (DataType["INT8"], DataType["INT7"]),
        (DataType["FLOAT32"], DataType["FLOAT32"]),
        (DataType["FLOAT16"], DataType["FLOAT16"]),
        (DataType["FIXED<6,2>"], DataType["FIXED<8,4>"]),
    ],
)
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("per_tensor", [True, False])
@pytest.mark.parametrize("impl_style", ["rtl"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.parametrize("round_thresh", [True, False])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_thresholding(
    num_input_channels,
    num_input_vecs,
    activation,
    idt_tdt_cfg,
    fold,
    narrow,
    per_tensor,
    impl_style,
    exec_mode,
    mem_mode,
    round_thresh,
):
    # the mem_mode parameter can only be used for the hls thresholding
    # so the test will only be executed once for impl_style=rtl and once skipped
    # when the mem_mode is varied. Otherwise, the same test configuration would always
    # run twice.
    if impl_style == "rtl" and mem_mode == "internal_decoupled":
        pytest.skip(
            "Skip, because test is identical to impl_style=rtl and mem_mode=internal_embedded"
        )
    if narrow and activation == DataType["BIPOLAR"]:
        pytest.skip("Narrow needs to be false with biploar activation.")
    input_data_type, threshold_data_type = idt_tdt_cfg
    num_steps = activation.get_num_possible_values() - 1
    if input_data_type in ["FLOAT32", "FLOAT16"] and round_thresh:
        pytest.skip(
            "Thresholds will not be rounded when inputs are floating-point. "
            "Test case is identical with floating-point input and round_thresh=False."
        )
    if (
        impl_style == "rtl"
        and input_data_type.is_fixed_point()
        and not threshold_data_type.is_fixed_point()
    ):
        pytest.skip("Fixed-point inputs and non-fixed-point thresholds are not supported in RTL.")

    if fold == -1:
        fold = num_input_channels
    pe = num_input_channels // fold
    if num_input_channels % pe != 0:
        pytest.skip("Invalid folding configuration. Skipping test.")

    output_data_type = activation
    if activation == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = activation.min()
        if narrow and activation.signed():
            activation_bias += 1

    # Generate thresholds with edge cases (min/max) and sort in ascending order
    thresholds = generate_edge_threshold_values(
        threshold_data_type, num_input_channels, num_steps, narrow, per_tensor
    )

    # provide non-decreasing/ascending thresholds
    thresholds = sort_thresholds_increasing(thresholds)

    # Make a Multithreshold graph and convert to thresholding binary search node
    model = make_single_multithresholding_modelwrapper(
        thresholds,
        input_data_type,
        threshold_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
    )

    # calculate reference output with edge case inputs (min/max values)
    x = generate_edge_input_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))

    input_dict = {model.get_first_global_in(): x}
    y_expected = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]

    if output_data_type == DataType["BIPOLAR"]:
        # binary to bipolar
        y_expected = 2 * y_expected - 1

    model = model.transform(InferThresholdingLayer())

    # Perform functional validation of the InferThresholdingLayer transform
    y_produced = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    assert (y_produced.astype(np.float32) == y_expected.astype(np.float32)).all()

    # Transform to the specified implementation style, either the
    # RTL or HLS according to test parameters
    node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("preferred_impl_style", impl_style)
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(InferShapes())
    assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)

    node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    if round_thresh is True:
        model = model.transform(RoundAndClipThresholds())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(GiveUniqueNodeNames())

    if impl_style == "hls":
        inst.set_nodeattr("mem_mode", mem_mode)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_produced = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]
    assert (y_produced.astype(np.float32) == y_expected.astype(np.float32)).all()

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


@pytest.mark.parametrize("num_input_channels", [6])
@pytest.mark.parametrize(
    "num_input_vecs",
    [
        [1, 2, 2],
    ],
)
@pytest.mark.parametrize("activation", [DataType["INT4"]])
@pytest.mark.parametrize(
    "idt_tdt_cfg",
    [
        (DataType["INT8"], DataType["INT25"]),
    ],
)
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("ram_style", ["distributed", "block"])
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S", "xczu7ev-ffvc1156-2-e"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_thresholding_stitched_ip(
    num_input_channels, num_input_vecs, activation, idt_tdt_cfg, fold, ram_style, part
):
    input_data_type, threshold_data_type = idt_tdt_cfg
    num_steps = activation.get_num_possible_values() - 1

    if fold == -1:
        fold = num_input_channels
    pe = num_input_channels // fold

    output_data_type = activation
    activation_bias = activation.min()

    # Generate thresholds with edge cases (min/max) and sort in ascending order
    thresholds = generate_edge_threshold_values(
        threshold_data_type, num_input_channels, num_steps, False, False
    )

    # provide non-decreasing/ascending thresholds
    thresholds = sort_thresholds_increasing(thresholds)

    # Make a Multithreshold graph and convert to thresholding binary search node
    model = make_single_multithresholding_modelwrapper(
        thresholds,
        input_data_type,
        threshold_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
    )

    # calculate reference output with edge case inputs (min/max values)
    x = generate_edge_input_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))

    input_dict = {model.get_first_global_in(): x}
    y_expected = oxe.execute_onnx(model, input_dict)[model.get_first_global_out()]

    model = model.transform(InferThresholdingLayer())

    # Transform to the specified implementation style, either the
    # RTL or HLS according to test parameters
    node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("preferred_impl_style", "hls")
    model = model.transform(SpecializeLayers(part))
    model = model.transform(InferShapes())
    assert model.graph.node[0].op_type == "Thresholding_hls"

    node = model.get_nodes_by_op_type(model.graph.node[0].op_type)[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("ram_style", ram_style)

    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(GiveUniqueNodeNames())
    # Run stitched-ip RTLsim to have memstream in the test loop
    model = model.transform(InsertAndSetFIFODepths(part, target_clk_ns))
    model = model.transform(PrepareIP(part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")

    exec_ctx_dict = {"global_in": x}

    y_produced = oxe.execute_onnx(model, exec_ctx_dict)["global_out"]

    assert (
        y_expected == y_produced
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"

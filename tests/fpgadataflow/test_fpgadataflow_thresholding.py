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
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


def generate_random_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1

    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


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
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, num_input_vecs + [num_channels])
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
    ],
)
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("per_tensor", [True, False])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
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

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(
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

    # calculate reference output
    x = gen_finn_dt_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))

    input_dict = {model.graph.input[0].name: x}
    y_expected = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]

    if output_data_type == DataType["BIPOLAR"]:
        # binary to bipolar
        y_expected = 2 * y_expected - 1

    model = model.transform(InferThresholdingLayer())

    # Perform functional validation of the InferThresholdingLayer transform
    y_produced = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert (y_produced == y_expected).all()

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

    y_produced = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
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

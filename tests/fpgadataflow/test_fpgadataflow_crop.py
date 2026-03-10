############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
# Note: This test was originally written by Josh Monson and was adjusted.
#
############################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part: str = "xczu7ev-ffvc1156-2-e"
target_clk_ns = 5


def make_gather_model(indices, ishape, axis):
    # Define the input tensor
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, ishape)

    # Define the output tensor and leave shape undefined to be inferred later
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

    indices = helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)

    gather_node = helper.make_node(
        "Gather", inputs=["data", "indices"], outputs=["output"], axis=axis
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=[gather_node],
        name="GatherGraph",
        inputs=[data],
        outputs=[output],
        initializer=[
            indices,
        ],
    )

    # Create the QONNX model
    model = qonnx_make_model(graph, producer_name="gather-model")
    model = ModelWrapper(model, fix_missing_initializer_valueinfo=True)
    model = model.transform(InferShapes())

    return model


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.parametrize(
    "ishape_axis_indices",
    [
        ([1, 16, 48], 1, [0]),
        ([1, 16, 48], 1, [4, 5, 6]),
        ([32, 48], 0, [15]),
        ([1, 16, 48, 16], 2, [1]),
    ],
)
@pytest.mark.parametrize("simd", [1, 8, 16])
@pytest.mark.parametrize("idt", [DataType["INT8"], DataType["FLOAT32"]])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
def test_fpgadataflow_gather_crop(ishape_axis_indices, simd, idt, exec_mode):
    ishape, axis, indices = ishape_axis_indices
    indices = np.array(indices)
    model = make_gather_model(indices, ishape, axis=axis)
    model.set_tensor_datatype(model.graph.input[0].name, idt)

    # reference calculation
    input = gen_finn_dt_tensor(idt, ishape)
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    model = model.transform(to_hw.InferCrop())

    input_t = {model.graph.input[0].name: input}
    y_hw = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert (y_ref == y_hw).all()

    model = model.transform(SpecializeLayers(test_fpga_part))
    assert model.graph.node[0].op_type == "Crop_hls"
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", simd)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    input_t = {model.graph.input[0].name: input}

    y_sim = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert (y_ref == y_sim).all()

    if exec_mode == "rtlsim":
        cycles_rtlsim = getCustomOp(model.graph.node[0]).get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[model.graph.node[0].name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

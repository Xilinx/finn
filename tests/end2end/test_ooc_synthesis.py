############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.vivado import parse_ooc_synth_results

fpga_part = "xczu7ev-ffvc1156-2-e"
clk_ns = 10


def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    if data_type.is_integer():
        return np.random.randint(
            data_type.min(),
            data_type.max() + 1,
            (num_input_channels, num_steps),
        ).astype(np.float32)
    else:
        return (np.random.randn(num_input_channels, num_steps) * 1000).astype(
            data_type.to_numpy_dt()
        )


def create_test_model():
    W = gen_finn_dt_tensor(DataType["INT4"], (16, 32))
    T = np.sort(
        generate_random_threshold_values(
            DataType["FLOAT32"],
            1,
            DataType["INT8"].get_num_possible_values() - 1,
        ),
        axis=1,
    )
    MulParam = gen_finn_dt_tensor(DataType["FLOAT32"], [1])
    AddParam = gen_finn_dt_tensor(DataType["FLOAT32"], [1, 4, 32])

    # Initialize a new graph
    nodes = []

    # Add nodes
    mt_op = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["mt_output"],
        domain="qonnx.custom_op.general",
        out_dtype="INT8",
        out_bias=float(DataType["INT8"].min()),
    )
    nodes.append(mt_op)

    matmul_op = helper.make_node(
        "MatMul",
        inputs=["mt_output", "matmul_weight"],
        outputs=["matmul_output"],
    )
    nodes.append(matmul_op)

    scalar_mul_op = helper.make_node(
        "Mul",
        inputs=["matmul_output", "scalar_input"],
        outputs=["scalar_output"],
    )
    nodes.append(scalar_mul_op)

    channel_add_op = helper.make_node(
        "Add",
        inputs=["scalar_output", "channelwise_bias"],
        outputs=["final_output"],
    )
    nodes.append(channel_add_op)

    # Define inputs
    inputs = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 16]),
    ]

    # Define outputs
    outputs = [helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [1, 4, 32])]

    value_info = [
        helper.make_tensor_value_info("mt_output", TensorProto.FLOAT, [1, 4, 16]),
        helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [1, 255]),
        helper.make_tensor_value_info("matmul_output", TensorProto.FLOAT, [1, 4, 32]),
        helper.make_tensor_value_info("matmul_weight", TensorProto.FLOAT, [16, 32]),
        helper.make_tensor_value_info("scalar_input", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("scalar_output", TensorProto.FLOAT, [1, 4, 32]),
        helper.make_tensor_value_info("channelwise_bias", TensorProto.FLOAT, [1, 4, 32]),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes, name="TestModelGraph", inputs=inputs, outputs=outputs, value_info=value_info
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)

    # Set initializers and datatypes
    model.set_initializer("matmul_weight", W)
    model.set_initializer("thresh", T)
    model.set_initializer("scalar_input", MulParam)
    model.set_initializer("channelwise_bias", AddParam)

    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("matmul_weight", DataType["INT4"])
    model.set_tensor_datatype("thresh", DataType["FLOAT32"])
    model.set_tensor_datatype("scalar_input", DataType["FLOAT32"])
    model.set_tensor_datatype("channelwise_bias", DataType["FLOAT32"])

    return model


@pytest.mark.end2end
@pytest.mark.vivado
@pytest.mark.slow
def test_ooc_synthesis():
    model = create_test_model()
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # generate reference output
    x = gen_finn_dt_tensor(DataType["FLOAT32"], (1, 4, 16))
    y_dict = oxe.execute_onnx(model, {model.get_first_global_in(): x})
    y_ref = y_dict[model.get_first_global_out()]

    # infer and specialize layers
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(SpecializeLayers(fpga_part))

    # node-by-node rtlsim
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_dict = oxe.execute_onnx(model, {model.get_first_global_in(): x})
    y_prod = y_dict[model.get_first_global_out()]
    assert (y_prod == y_ref).all()

    # FIFO sizing
    model = model.transform(InsertAndSetFIFODepths(fpga_part, clk_ns))

    # stitched IP rtlsim
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())

    # Test NEW flow: CreateStitchedIP with run_pnr=True
    model = model.transform(CreateStitchedIP(fpga_part, clk_ns, run_pnr=True))

    # Parse results from new flow (pass directory, not file)
    vivado_stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
    new_results = parse_ooc_synth_results(vivado_stitch_proj)
    assert new_results is not None, "New OOC flow did not produce results"

    # Verify new flow results
    assert new_results["LUT"] > 0
    assert new_results["FF"] > 0
    assert new_results["DSP"] > 0
    assert new_results["BRAM_18K"] > 0 or new_results["BRAM_36K"] > 0
    assert new_results["fmax_mhz"] > 100

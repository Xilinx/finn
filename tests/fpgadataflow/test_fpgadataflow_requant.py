# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test cases for InferRequantLayer transformation which converts MultiThreshold
nodes with uniform thresholds to Requant nodes.

The requant operation computes output as:
    clip(round(x * scale + bias), min, max)
instead of comparing against multiple thresholds.
"""

import pytest

import numpy as np
import os
import torch
from brevitas.core.scaling import ScalingImplType
from brevitas.export import export_qonnx
from brevitas.nn import QuantReLU
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferRequantLayer
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir, pynq_part_map


@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("max_val", [1.0, 6.0])
@pytest.mark.parametrize("ishape", [(1, 15), (1, 32, 1, 1)])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("input_dtype", ["INT8", "FLOAT32"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_infer_requant_layer(abits, max_val, ishape, per_channel, input_dtype, exec_mode):
    """Test InferRequantLayer converts MultiThreshold to Requant."""

    num_channels = ishape[1]

    if per_channel:
        b_act = QuantReLU(
            bit_width=abits,
            max_val=max_val,
            scaling_impl_type=ScalingImplType.CONST,
            scaling_per_output_channel=True,
            per_channel_broadcastable_shape=(1, num_channels) + (1,) * (len(ishape) - 2),
        )
    else:
        b_act = QuantReLU(
            bit_width=abits,
            max_val=max_val,
            scaling_impl_type=ScalingImplType.CONST,
        )

    # Export to QONNX
    build_dir = make_build_dir(prefix="test_infer_requant_layer_")
    m_path = os.path.join(build_dir, "model.onnx")
    export_qonnx(b_act, torch.randn(ishape), m_path)
    qonnx_cleanup(m_path, out_file=m_path)

    # Convert to FINN format (creates MultiThreshold)
    model = ModelWrapper(m_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())

    # Set input datatype
    # Must re-run InferDataTypes to propagate to intermediate tensors
    model.set_tensor_datatype(model.graph.input[0].name, DataType[input_dtype])
    model = model.transform(InferDataTypes())

    # Get golden output before conversion
    if input_dtype == "INT8":
        inp = np.random.randint(-128, 127, size=ishape).astype(np.float32)
    else:
        inp = np.random.uniform(-10.0, 10.0, size=ishape).astype(np.float32)
    input_dict = {model.graph.input[0].name: inp}
    y_golden = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]

    # Apply InferRequantLayer
    model = model.transform(InferRequantLayer())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Verify MultiThreshold was converted to Requant
    mt_nodes = model.get_nodes_by_op_type("MultiThreshold")
    requant_nodes = model.get_nodes_by_op_type("Requant")
    assert len(mt_nodes) == 0, "MultiThreshold should be converted"
    assert len(requant_nodes) == 1, "Expected one Requant node"

    # Verify Requant attributes
    requant_node = requant_nodes[0]
    requant_inst = getCustomOp(requant_node)
    assert requant_inst.get_nodeattr("NumChannels") == num_channels
    # Output datatype should be unsigned (ReLU output)
    odt = requant_inst.get_output_datatype()
    assert odt.bitwidth() == abits
    assert not odt.signed(), "ReLU output should be unsigned"

    # Verify scale and bias are set as initializers
    scale = model.get_initializer(requant_node.input[1])
    bias = model.get_initializer(requant_node.input[2])
    assert scale is not None, "Scale should be set as initializer"
    assert bias is not None, "Bias should be set as initializer"
    # Scale/bias shape depends on threshold shape from Brevitas, not per_channel flag
    # With CONST scaling and scalar max_val, thresholds are always (1, N) and broadcast
    assert scale.size >= 1, f"Scale should have at least 1 element: {scale.shape}"
    assert bias.size >= 1, f"Bias should have at least 1 element: {bias.shape}"

    # Verify functional correctness
    # Allow tolerance of 1 quantization step due to floating point precision at boundaries
    # The quantization step is max_val / (2^abits - 1)
    quant_step = max_val / (2**abits - 1)
    y_requant = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert np.allclose(y_golden, y_requant, atol=quant_step), (
        f"Output mismatch: max diff = {np.max(np.abs(y_golden - y_requant))}, "
        f"quant_step = {quant_step}"
    )

    # Specialize layers based on input type
    test_fpga_part = pynq_part_map["ZCU104"]
    target_clk_ns = 10

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    # Verify correct backend was selected
    if input_dtype == "INT8":
        requant_rtl_nodes = model.get_nodes_by_op_type("Requant_rtl")
        assert len(requant_rtl_nodes) == 1, "Expected one Requant_rtl node"
    else:
        requant_hls_nodes = model.get_nodes_by_op_type("Requant_hls")
        assert len(requant_hls_nodes) == 1, "Expected one Requant_hls node"

    # Prepare and run simulation
    model = model.transform(SetExecMode(exec_mode))

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    else:
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_sim = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert np.allclose(
        y_golden, y_sim, atol=quant_step
    ), f"{exec_mode} mismatch: max diff = {np.max(np.abs(y_golden - y_sim))}"

    # Verify cycle estimation for rtlsim
    if exec_mode == "rtlsim":
        op_type = "Requant_rtl" if input_dtype == "INT8" else "Requant_hls"
        node = model.get_nodes_by_op_type(op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0

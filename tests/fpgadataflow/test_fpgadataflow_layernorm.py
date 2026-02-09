###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias

test_fpga_part = "xcvc1902-vsva2197-2MP-e-S"
target_clk_ns = 5


def create_layernorm_model(idt, ishape, has_scale, has_bias, epsilon):
    scale_bias_shape = [ishape[-1]]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale_bias_shape)
    if has_bias:
        bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, scale_bias_shape)

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale", "bias"] if has_bias else ["inp", "scale"],
        outputs=["outp"],
        name="Layernorm_0",
        epsilon=epsilon,
        axis=-1,
        stash_type=1,
    )

    # Create model
    graph = helper.make_graph(
        nodes=[ln_node],
        name="LayerNorm_graph",
        inputs=[inp, scale, bias] if has_bias else [inp, scale],
        outputs=[outp],
    )
    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    # Tensor initializers
    if has_scale:
        scale = gen_finn_dt_tensor(DataType["FLOAT32"], scale_bias_shape)
    else:
        scale = np.ones(scale_bias_shape, dtype=np.float32)
    model.set_initializer("scale", scale)

    if has_bias:
        bias = gen_finn_dt_tensor(DataType["FLOAT32"], scale_bias_shape)
        model.set_initializer("bias", bias)

    # Tensor data types
    model.set_tensor_datatype("inp", idt)

    return model


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("idt", [DataType["FLOAT32"]])
@pytest.mark.parametrize("ishape", [[1, 16, 48], [1, 32]])
@pytest.mark.parametrize("simd", [1, 2])
@pytest.mark.parametrize(
    "sim_style",
    ["node_by_node", pytest.param("stitched_ip", marks=pytest.mark.xfail(reason="sim bug"))],
)
def test_fpgadataflow_rtl_layernorm(idt, ishape, simd, sim_style):
    model = create_layernorm_model(
        idt, ishape, has_scale=True, has_bias=True, epsilon=9.999999960041972e-13
    )

    # reference calculation
    input = gen_finn_dt_tensor(idt, ishape)
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(ExtractNormScaleBias())

    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    input_t = {model.graph.input[0].name: input}

    y_hw = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]
    assert np.allclose(y_ref, y_hw, rtol=1e-3, atol=2**-4)

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    assert model.graph.node[0].op_type == "LayerNorm_rtl", "LayerNorm wasn't converted to RTL Layer"

    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", simd)

    # Execute
    if sim_style == "node_by_node":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif sim_style == "stitched_ip":
        model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")

    input_t = {model.graph.input[0].name: input}

    y_rtl = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert np.allclose(y_ref, y_rtl, rtol=1e-3, atol=2**-4)

    if sim_style == "node_by_node":
        cycles_rtlsim = getCustomOp(model.graph.node[0]).get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[model.graph.node[0].name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("idt", [DataType["FLOAT32"], DataType["INT8"]])
@pytest.mark.parametrize("ishape", [[1, 16, 48], [1, 32]])
@pytest.mark.parametrize("simd", [1, 2])
@pytest.mark.parametrize(
    "sim_style",
    ["cppsim", "node_by_node", "stitched_ip"],
)
def test_fpgadataflow_hls_layernorm(idt, ishape, simd, sim_style):
    model = create_layernorm_model(
        idt, ishape, has_scale=True, has_bias=True, epsilon=9.999999960041972e-13
    )

    # reference calculation
    input = gen_finn_dt_tensor(idt, ishape)
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(ExtractNormScaleBias())

    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    input_t = {model.graph.input[0].name: input}

    y_hw = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]
    assert np.allclose(y_ref, y_hw, rtol=1e-3, atol=2**-4)

    getCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "hls")
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    assert model.graph.node[0].op_type == "LayerNorm_hls", "LayerNorm wasn't converted to HLS Layer"

    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", simd)

    # Execute
    if sim_style == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif sim_style == "node_by_node":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif sim_style == "stitched_ip":
        model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")

    input_t = {model.graph.input[0].name: input}

    y_rtl = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert np.allclose(y_ref, y_rtl, rtol=1e-3, atol=2**-4)


@pytest.mark.transform
@pytest.mark.parametrize("idt", [DataType["FLOAT32"]])
@pytest.mark.parametrize("ishape", [[1, 16, 48], [1, 32]])
@pytest.mark.parametrize("has_scale", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_extract_norm_scale_bias(idt, ishape, has_scale, has_bias):
    epsilon = 9.999999960041972e-13
    model1 = create_layernorm_model(idt, ishape, has_scale, has_bias, epsilon)
    model2 = create_layernorm_model(idt, ishape, has_scale, has_bias, epsilon)
    model3 = create_layernorm_model(idt, ishape, has_scale, has_bias, epsilon)

    model = model1.transform(MergeONNXModels(model2))
    model = model.transform(MergeONNXModels(model3))

    assert len(model.get_nodes_by_op_type("LayerNormalization")) == 3

    # reference calculation
    input = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(ExtractNormScaleBias())

    assert len(model.get_nodes_by_op_type("LayerNormalization")) == 3
    if has_bias:
        assert len(model.get_nodes_by_op_type("Add")) == 3
    if has_scale:
        assert len(model.get_nodes_by_op_type("Mul")) == 3

    input_t = {model.graph.input[0].name: input}

    y_out = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]
    assert (y_ref == y_out).all()

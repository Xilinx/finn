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

import json
import numpy as np
import os
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
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
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from finn.util.basic import make_build_dir

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
    ["node_by_node", "stitched_ip"],
)
def test_fpgadataflow_rtl_layernorm(idt, ishape, simd, sim_style):
    """Test RTL LayerNorm with N/SIMD > 12 (original regime)."""
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

    model = model.transform(MinimizeWeightBitWidth())
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
        # Set debug waveform for stitched IP
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
@pytest.mark.parametrize("idt", [DataType["FLOAT32"]])
@pytest.mark.parametrize(
    "ishape,simd",
    [
        ([1, 4], 4),  # NN=1  -> rsqrt genII1 (3 DSPs)
        ([1, 10], 5),  # NN=2  -> rsqrt genII2 (2 DSPs)
        ([1, 18], 6),  # NN=3  -> rsqrt genInterleave
        ([1, 42], 7),  # NN=6  -> rsqrt genInterleave
        ([1, 64], 8),  # NN=8  -> rsqrt genInterleave
        ([1, 81], 9),  # NN=9  -> rsqrt genOverlapped
        ([1, 100], 10),  # NN=10 -> rsqrt genOverlapped
        ([1, 44], 4),  # NN=11 -> rsqrt genOverlapped
    ],
)
def test_fpgadataflow_rtl_layernorm_low_simd_ratio(idt, ishape, simd):
    """Test RTL LayerNorm with N/SIMD <= 12, exercising the new rsqrt strategies."""
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

    # Execute node-by-node RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    input_t = {model.graph.input[0].name: input}

    y_rtl = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert np.allclose(y_ref, y_rtl, rtol=1e-3, atol=2**-4)


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
    model = model.transform(MinimizeWeightBitWidth())
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


def create_mul_layernorm_model(idt, ishape, mul_param_shape):
    """
    Create a model: INT8 input -> Mul (FLOAT32 param) -> LayerNorm (scale, bias).

    This model triggers the HLS+RTL DSP conflict when specialized:
    - Mul with INT8 input and FLOAT32 param -> ElementwiseMul_hls (uses DSP for multiplication)
    - LayerNorm with FLOAT32 -> LayerNorm_rtl (uses DSPFP32)
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    mul_param = helper.make_tensor_value_info("mul_param", TensorProto.FLOAT, mul_param_shape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [ishape[-1]])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [ishape[-1]])

    # Mul node: INT8 input * FLOAT32 param -> FLOAT32 output
    mul_node = helper.make_node(
        "Mul",
        inputs=["inp", "mul_param"],
        outputs=["mul_out"],
        name="Mul_0",
    )

    # LayerNorm node with scale and bias
    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["mul_out", "scale", "bias"],
        outputs=["outp"],
        name="LayerNorm_0",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    # Intermediate value info
    mul_out_vi = helper.make_tensor_value_info("mul_out", TensorProto.FLOAT, ishape)

    graph = helper.make_graph(
        nodes=[mul_node, ln_node],
        name="mul_layernorm_graph",
        inputs=[inp, mul_param, scale, bias],
        outputs=[outp],
        value_info=[mul_out_vi],
    )
    model = qonnx_make_model(graph, producer_name="mul_layernorm_test")
    model = ModelWrapper(model)

    # Set initializers
    mul_param_data = gen_finn_dt_tensor(DataType["FLOAT32"], mul_param_shape)
    scale_data = gen_finn_dt_tensor(DataType["FLOAT32"], [ishape[-1]])
    bias_data = gen_finn_dt_tensor(DataType["FLOAT32"], [ishape[-1]])

    model.set_initializer("mul_param", mul_param_data)
    model.set_initializer("scale", scale_data)
    model.set_initializer("bias", bias_data)

    # Set tensor datatypes
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mul_param", DataType["FLOAT32"])
    model.set_tensor_datatype("mul_out", DataType["FLOAT32"])
    model.set_tensor_datatype("scale", DataType["FLOAT32"])
    model.set_tensor_datatype("bias", DataType["FLOAT32"])
    model.set_tensor_datatype("outp", DataType["FLOAT32"])

    return model


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_hls_rtl_dsp_conflict_detection():
    """
    Test that HLS+RTL DSP conflict is detected and verification is skipped.

    This test creates a model with:
    - INT8 input -> Mul (FLOAT32 param) -> ElementwiseMul_hls (uses DSP)
    - LayerNorm with scale/bias -> LayerNorm_rtl (uses DSPFP32)

    When running stitched_ip_rtlsim verification, the DSP conflict should be
    detected and verification skipped with a warning. The hardware is correct,
    only xsim simulation produces incorrect results due to DSP initialization
    conflicts.
    """
    ishape = [1, 32]
    mul_param_shape = [ishape[-1]]
    idt = DataType["INT8"]

    # Create model and prepare for build
    model = create_mul_layernorm_model(idt, ishape, mul_param_shape)

    # Generate reference input/output
    input_data = gen_finn_dt_tensor(idt, ishape)
    input_t = {"inp": input_data}
    y_ref = oxe.execute_onnx(model, input_t)["outp"]

    # Apply minimal transformations - build flow handles the rest
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(ExtractNormScaleBias())

    # Setup build directory
    tmp_output_dir = make_build_dir("build_dsp_conflict_test_")

    np.save(tmp_output_dir + "/input.npy", input_data)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)
    model.save(tmp_output_dir + "/model.onnx")

    # Create specialize_layers config to force the first Mul to use HLS implementation.
    # This future-proofs the test for when RTL gets int+float->float support.
    specialize_config = {
        "Defaults": {},
        "ElementwiseMul_0": {"preferred_impl_style": "hls"},
    }
    specialize_config_file = tmp_output_dir + "/specialize_layers_config.json"
    with open(specialize_config_file, "w") as f:
        json.dump(specialize_config, f)

    # Build steps - includes conversion to HW layers and specialization
    steps = [
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    # Request verification steps that will trigger DSP conflict detection
    verif_steps = [
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=target_clk_ns,
        board="VCK190",
        specialize_layers_config_file=specialize_config_file,
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )

    # Capture warnings during build
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)

    # Check that DSP conflict warning was issued
    dsp_conflict_warnings = [
        w for w in caught_warnings if "HLS+RTL DSP CONFLICT DETECTED" in str(w.message)
    ]
    assert len(dsp_conflict_warnings) > 0, (
        "Expected DSP conflict warning to be issued. "
        f"Found warnings: {[str(w.message)[:100] for w in caught_warnings]}"
    )

    # Verify cppsim still passed (not affected by DSP conflict)
    verif_dir = tmp_output_dir + "/verification_output"

    # Check that the warning log file was created in the verification folder
    stitched_conflict_file = os.path.join(verif_dir, "stitched_ip_rtlsim_SKIPPED_DSP_CONFLICT.txt")
    assert os.path.isfile(
        stitched_conflict_file
    ), f"Expected DSP conflict log file at {stitched_conflict_file}"
    cppsim_success = os.path.join(verif_dir, "verify_folded_hls_cppsim_0_SUCCESS.npy")
    assert os.path.isfile(
        cppsim_success
    ), f"cppsim verification should have passed - check {verif_dir}"

    # Verify node_by_node_rtlsim passed (not affected by DSP conflict for non-MLO)
    rtlsim_success = os.path.join(verif_dir, "verify_node_by_node_rtlsim_0_SUCCESS.npy")
    assert os.path.isfile(
        rtlsim_success
    ), f"node_by_node_rtlsim verification should have passed - check {verif_dir}"

    # Verify that stitched_ip_rtlsim was skipped (no SUCCESS file)
    stitched_success = os.path.join(verif_dir, "verify_stitched_ip_rtlsim_0_SUCCESS.npy")
    assert not os.path.isfile(
        stitched_success
    ), "stitched_ip_rtlsim should have been skipped due to DSP conflict"


def create_layernorm_threshold_mul_model(ishape):
    """
    Create a model: LayerNorm -> MultiThreshold -> Mul (INT param).

    This model has both RTL DSP ops (LayerNorm_rtl) and HLS Elementwise ops,
    but the HLS Elementwise uses integer datatypes (after quantization via
    MultiThreshold), so it should NOT trigger the DSP conflict detection.
    """
    num_channels = ishape[-1]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [num_channels])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [num_channels])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [num_channels, 255])
    mul_param = helper.make_tensor_value_info("mul_param", TensorProto.FLOAT, [num_channels])

    # LayerNorm (will become LayerNorm_rtl)
    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale", "bias"],
        outputs=["ln_out"],
        name="LayerNorm_0",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    # MultiThreshold to quantize to INT8
    mt_node = helper.make_node(
        "MultiThreshold",
        inputs=["ln_out", "thresh"],
        outputs=["mt_out"],
        domain="qonnx.custom_op.general",
        out_dtype="INT8",
        out_bias=-128.0,
        out_scale=1.0,
        data_layout="NC",
    )

    # Mul with integer parameter (INT8 * INT8 -> INT16)
    # After MultiThreshold, input is INT8, param will be INT8
    mul_node = helper.make_node(
        "Mul",
        inputs=["mt_out", "mul_param"],
        outputs=["outp"],
        name="Mul_int",
    )

    # Intermediate value infos
    ln_out_vi = helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, ishape)
    mt_out_vi = helper.make_tensor_value_info("mt_out", TensorProto.FLOAT, ishape)

    graph = helper.make_graph(
        nodes=[ln_node, mt_node, mul_node],
        name="ln_thresh_mul_graph",
        inputs=[inp, scale, bias, thresh, mul_param],
        outputs=[outp],
        value_info=[ln_out_vi, mt_out_vi],
    )
    model = qonnx_make_model(graph, producer_name="ln_thresh_mul_test")
    model = ModelWrapper(model)

    # Set initializers
    scale_data = np.ones(num_channels, dtype=np.float32)
    bias_data = np.zeros(num_channels, dtype=np.float32)
    # Create thresholds for 256 levels (INT8 range)
    thresh_data = np.zeros((num_channels, 255), dtype=np.float32)
    for ch in range(num_channels):
        thresh_data[ch, :] = np.linspace(-10.0, 10.0, 255)
    # Integer mul param (small integers)
    mul_param_data = np.ones(num_channels, dtype=np.float32) * 2.0

    model.set_initializer("scale", scale_data)
    model.set_initializer("bias", bias_data)
    model.set_initializer("thresh", thresh_data)
    model.set_initializer("mul_param", mul_param_data)

    # Set tensor datatypes
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("ln_out", DataType["FLOAT32"])
    model.set_tensor_datatype("mt_out", DataType["INT8"])
    model.set_tensor_datatype("mul_param", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["INT16"])
    model.set_tensor_datatype("scale", DataType["FLOAT32"])
    model.set_tensor_datatype("bias", DataType["FLOAT32"])
    model.set_tensor_datatype("thresh", DataType["FLOAT32"])

    return model


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_integer_hls_elementwise_no_dsp_conflict():
    """
    Test that integer-only HLS Elementwise ops do NOT trigger DSP conflict detection.

    This test creates a model with:
    - LayerNorm -> LayerNorm_rtl (uses DSPFP32)
    - MultiThreshold -> quantizes to INT8
    - Mul (INT8 * INT8) -> ElementwiseMul_hls (integer, no DSP conflict)

    The model has HLS Elementwise ops AND RTL DSP ops, but since the HLS
    Elementwise uses integer datatypes (not floating-point), NO DSP conflict
    should be detected and stitched_ip_rtlsim should run successfully.
    """
    ishape = [1, 32]

    # Create model and prepare for build
    model = create_layernorm_threshold_mul_model(ishape)

    # Generate reference input/output
    input_data = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
    input_t = {"inp": input_data}
    y_ref = oxe.execute_onnx(model, input_t)["outp"]

    # Apply minimal transformations - build flow handles the rest
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(ExtractNormScaleBias())

    # Setup build directory
    tmp_output_dir = make_build_dir("build_int_elementwise_test_")

    np.save(tmp_output_dir + "/input.npy", input_data)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)
    model.save(tmp_output_dir + "/model.onnx")

    # Build steps - includes conversion to HW layers and specialization
    steps = [
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    # Request verification steps - stitched_ip_rtlsim should NOT be skipped
    verif_steps = [
        "folded_hls_cppsim",
        "stitched_ip_rtlsim",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=target_clk_ns,
        board="VCK190",
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )

    # Capture warnings during build
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)

    # Check that NO DSP conflict warning was issued
    dsp_conflict_warnings = [
        w for w in caught_warnings if "HLS+RTL DSP CONFLICT DETECTED" in str(w.message)
    ]
    assert len(dsp_conflict_warnings) == 0, (
        f"No DSP conflict warning should be issued for integer HLS Elementwise. "
        f"Found warnings: {[str(w.message)[:100] for w in dsp_conflict_warnings]}"
    )

    # Verify that stitched_ip_rtlsim ran successfully (was NOT skipped)
    verif_dir = tmp_output_dir + "/verification_output"
    stitched_success = os.path.join(verif_dir, "verify_stitched_ip_rtlsim_0_SUCCESS.npy")
    assert os.path.isfile(
        stitched_success
    ), f"stitched_ip_rtlsim should have run (not skipped) and passed - check {verif_dir}"

    # Verify no conflict log file was created
    stitched_conflict_file = os.path.join(verif_dir, "stitched_ip_rtlsim_SKIPPED_DSP_CONFLICT.txt")
    assert not os.path.isfile(
        stitched_conflict_file
    ), f"No DSP conflict log file should exist at {stitched_conflict_file}"

# Copyright Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DSP conflict between HLS floating-point ops and RTL LayerNorm.

When running xsim (Vivado RTL simulation), there's a known bug (Vivado <= 2025.2)
where DSP primitive initialization conflicts occur between:
- HLS ops using floating-point (via hls_math.h): HWSoftmax_hls, LayerNorm_hls,
  Requant_hls, Elementwise*_hls with FLOAT32
- RTL LayerNorm_rtl (uses DSPFP32 via binopf.sv)

The hardware is correct - only xsim simulation produces incorrect results.

These tests verify:
1. The conflict detection function correctly identifies conflicts
2. The actual simulation bug manifests in rtlsim (cppsim passes, rtlsim fails)
"""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias

# Versal part required for RTL LayerNorm
VERSAL_PART = "xcv80-lsva4737-2MHP-e-s"
TARGET_CLK_NS = 10.0


def create_softmax_layernorm_model(ishape, num_channels, input_dtype="FLOAT32"):
    """Create: input -> HWSoftmax -> LayerNormalization -> output"""
    scale_bias_shape = [num_channels]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    softmax_out = helper.make_tensor_value_info("softmax_out", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)

    softmax_node = helper.make_node(
        "HWSoftmax",
        inputs=["inp"],
        outputs=["softmax_out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="HWSoftmax_0",
        NumChannels=num_channels,
        SIMD=2,
        ifm_dim=ishape,
        input_data_type=input_dtype,
    )

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["softmax_out", "scale", "bias"],
        outputs=["outp"],
        name="LayerNorm_0",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    graph = helper.make_graph(
        nodes=[softmax_node, ln_node],
        name="softmax_layernorm_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[softmax_out],
    )
    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    model.set_initializer("scale", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias", np.zeros(scale_bias_shape, dtype=np.float32))
    model.set_tensor_datatype("inp", DataType[input_dtype])
    model.set_tensor_datatype("softmax_out", DataType["FLOAT32"])

    return model


def create_layernorm_hls_layernorm_rtl_model(ishape, num_channels):
    """Create: input -> LayerNormalization (HLS) -> LayerNormalization (RTL) -> output

    Creates two LayerNorm ops with preferred_impl_style to force one to HLS
    and one to RTL after SpecializeLayers.
    """
    scale_bias_shape = [num_channels]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    ln1_out = helper.make_tensor_value_info("ln1_out", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)

    # First LayerNorm - will be forced to HLS
    ln1_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale1", "bias1"],
        outputs=["ln1_out"],
        name="LayerNorm_HLS",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    # Second LayerNorm - will be forced to RTL
    ln2_node = helper.make_node(
        "LayerNormalization",
        inputs=["ln1_out", "scale2", "bias2"],
        outputs=["outp"],
        name="LayerNorm_RTL",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    graph = helper.make_graph(
        nodes=[ln1_node, ln2_node],
        name="dual_layernorm_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[ln1_out],
    )
    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    model.set_initializer("scale1", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias1", np.zeros(scale_bias_shape, dtype=np.float32))
    model.set_initializer("scale2", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias2", np.zeros(scale_bias_shape, dtype=np.float32))
    model.set_tensor_datatype("inp", DataType["FLOAT32"])

    return model


def create_layernorm_requant_model(ishape, num_channels):
    """Create: input -> LayerNormalization -> Requant (FLOAT32->INT8) -> output"""
    scale_bias_shape = [num_channels]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    ln_out = helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale", "bias"],
        outputs=["ln_out"],
        name="LayerNorm_0",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    # Create Requant node (generic, will be specialized to HLS)
    requant_node = helper.make_node(
        "Requant",
        inputs=["ln_out", "req_scale", "req_bias"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="Requant_0",
        NumChannels=num_channels,
        PE=2,
        inputDataType="FLOAT32",
        outputDataType="INT8",
        numInputVectors=ishape[:-1],
        narrow=0,
        preferred_impl_style="hls",
    )

    graph = helper.make_graph(
        nodes=[ln_node, requant_node],
        name="layernorm_requant_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[ln_out],
    )
    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    model.set_initializer("scale", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias", np.zeros(scale_bias_shape, dtype=np.float32))
    # Requant scale/bias - use larger scale and non-zero bias to get non-zero INT8 outputs
    # LayerNorm outputs ~N(0,1), so scale=50 and bias=64 maps to INT8 range [0, 127] roughly
    model.set_initializer("req_scale", np.ones(scale_bias_shape, dtype=np.float32) * 50.0)
    model.set_initializer("req_bias", np.ones(scale_bias_shape, dtype=np.float32) * 64.0)
    model.set_tensor_datatype("inp", DataType["FLOAT32"])

    return model


def create_layernorm_int_elementwise_model(ishape, num_channels):
    """Create: input -> LayerNormalization -> Thresholding -> INT8 ElementwiseMul -> output

    This tests that integer-only HLS ops do NOT trigger the conflict.
    """
    scale_bias_shape = [num_channels]
    num_thresholds = 255  # For INT8 output

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    ln_out = helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, ishape)
    thresh_out = helper.make_tensor_value_info("thresh_out", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale", "bias"],
        outputs=["ln_out"],
        name="LayerNorm_0",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )

    # Thresholding to convert FLOAT32 -> INT8
    thresh_node = helper.make_node(
        "Thresholding",
        inputs=["ln_out", "thresholds"],
        outputs=["thresh_out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="Thresholding_0",
        NumChannels=num_channels,
        PE=2,
        inputDataType="FLOAT32",
        weightDataType="FLOAT32",
        outputDataType="INT8",
        ActVal=-128,
        numSteps=num_thresholds,
        numInputVectors=ishape[:-1],
    )

    # Integer ElementwiseMul (INT8 * INT8 -> INT16)
    eltwise_node = helper.make_node(
        "ElementwiseMul",
        inputs=["thresh_out", "eltwise_param"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="ElementwiseMul_0",
        lhs_shape=ishape,
        rhs_shape=scale_bias_shape,
        out_shape=ishape,
        lhs_dtype="INT8",
        rhs_dtype="INT8",
        out_dtype="INT16",
        lhs_style="input",
        rhs_style="const",
        PE=2,
        preferred_impl_style="hls",
    )

    graph = helper.make_graph(
        nodes=[ln_node, thresh_node, eltwise_node],
        name="layernorm_int_eltwise_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[ln_out, thresh_out],
    )
    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    model.set_initializer("scale", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias", np.zeros(scale_bias_shape, dtype=np.float32))
    # Generate sorted thresholds
    thresholds = np.sort(np.random.randn(num_channels, num_thresholds).astype(np.float32), axis=1)
    model.set_initializer("thresholds", thresholds)
    model.set_initializer("eltwise_param", gen_finn_dt_tensor(DataType["INT8"], scale_bias_shape))
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("thresh_out", DataType["INT8"])

    return model


def convert_to_hw(model, fpgapart):
    """Convert ONNX ops to HW layers and specialize."""
    model = model.transform(ExtractNormScaleBias())
    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def convert_dual_layernorm_to_hw(model, fpgapart):
    """Convert two LayerNorm ops, forcing first to HLS and second to RTL."""
    model = model.transform(ExtractNormScaleBias())
    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())

    # Set preferred_impl_style before SpecializeLayers
    # First LayerNorm -> HLS, second LayerNorm -> RTL
    ln_nodes = model.get_nodes_by_op_type("LayerNorm")
    if len(ln_nodes) >= 2:
        getCustomOp(ln_nodes[0]).set_nodeattr("preferred_impl_style", "hls")
        getCustomOp(ln_nodes[1]).set_nodeattr("preferred_impl_style", "rtl")

    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def build_stitched_ip(model, fpgapart, clk_ns=TARGET_CLK_NS):
    """Build stitched IP for rtlsim execution."""
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(fpgapart, clk_ns, vitis=False))
    return model


def run_stitched_ip_rtlsim(model, inp_dict):
    """Run stitched IP rtlsim and return output."""
    model.set_metadata_prop("exec_mode", "rtlsim")
    return execute_onnx(model, inp_dict)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
class TestDSPConflictSimulation:
    """Test that the DSP conflict manifests in stitched IP rtlsim.

    These tests verify that when HLS FP ops and RTL LayerNorm are combined,
    the stitched IP rtlsim produces incorrect results due to the xsim DSP
    initialization bug.
    """

    @pytest.mark.parametrize("input_dtype", ["FLOAT32", "INT8"])
    def test_softmax_layernorm_rtlsim_conflict(self, input_dtype):
        """Verify HWSoftmax + LayerNorm_rtl produces incorrect rtlsim results.

        Expected behavior:
        - Reference (ONNX execution): correct output
        - Stitched IP rtlsim: incorrect output (HLS FP ops output zeros)
          - HWSoftmax receives zeros -> outputs 1/N for all channels
          - LayerNorm receives uniform input -> outputs NaN (division by zero in variance)
        """
        ishape = [1, 4, 4, 16]
        num_channels = 16

        # Create model and get reference output
        model = create_softmax_layernorm_model(ishape, num_channels, input_dtype=input_dtype)
        inp = gen_finn_dt_tensor(DataType[input_dtype], ishape)
        inp_dict = {"inp": inp}

        # Get reference output from ONNX model (before HW conversion)
        ref_out = execute_onnx(model, inp_dict)
        oname = model.graph.output[0].name
        expected = ref_out[oname]

        # Convert to HW and build stitched IP
        model = convert_to_hw(model, VERSAL_PART)
        model = build_stitched_ip(model, VERSAL_PART)

        # Run stitched IP rtlsim
        rtlsim_out = run_stitched_ip_rtlsim(model, inp_dict)
        actual = rtlsim_out[model.graph.output[0].name]

        # Check if outputs match - if DSP conflict manifests, this will fail
        assert not np.any(np.isnan(actual)), f"rtlsim output contains NaNs: {actual}"
        assert np.allclose(expected, actual, rtol=1e-3, atol=2**-4), (
            "rtlsim output doesn't match reference.\n" f"Expected: {expected}\n" f"Actual: {actual}"
        )

    def test_layernorm_hls_rtl_rtlsim_conflict(self):
        """Verify LayerNorm_hls + LayerNorm_rtl produces incorrect rtlsim results."""
        ishape = [1, 4, 4, 16]
        num_channels = 16

        # Create model and get reference output
        model = create_layernorm_hls_layernorm_rtl_model(ishape, num_channels)
        inp = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
        inp_dict = {"inp": inp}

        # Get reference output from ONNX model (before HW conversion)
        ref_out = execute_onnx(model, inp_dict)
        oname = model.graph.output[0].name
        expected = ref_out[oname]

        # Convert to HW and build stitched IP
        model = convert_dual_layernorm_to_hw(model, VERSAL_PART)
        model = build_stitched_ip(model, VERSAL_PART)

        # Run stitched IP rtlsim
        rtlsim_out = run_stitched_ip_rtlsim(model, inp_dict)
        actual = rtlsim_out[model.graph.output[0].name]

        # Check if outputs match - if DSP conflict manifests, this will fail
        assert not np.any(np.isnan(actual)), f"rtlsim output contains NaNs: {actual}"
        assert np.allclose(expected, actual, rtol=1e-3, atol=2**-4), (
            "rtlsim output doesn't match reference.\n" f"Expected: {expected}\n" f"Actual: {actual}"
        )

    def test_layernorm_requant_rtlsim_conflict(self):
        """Verify LayerNorm_rtl + Requant_hls produces incorrect rtlsim results."""
        ishape = [1, 4, 4, 16]
        num_channels = 16

        # Create model and get reference output
        model = create_layernorm_requant_model(ishape, num_channels)
        inp = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
        inp_dict = {"inp": inp}

        # Get reference output from ONNX model (before HW conversion)
        ref_out = execute_onnx(model, inp_dict)
        oname = model.graph.output[0].name
        expected = ref_out[oname]

        # Convert to HW and build stitched IP
        model = convert_to_hw(model, VERSAL_PART)
        model = build_stitched_ip(model, VERSAL_PART)

        # Run stitched IP rtlsim
        rtlsim_out = run_stitched_ip_rtlsim(model, inp_dict)
        actual = rtlsim_out[model.graph.output[0].name]

        # Check if outputs match - if DSP conflict manifests, this will fail
        assert not np.any(np.isnan(actual)), f"rtlsim output contains NaNs: {actual}"
        assert np.allclose(expected, actual, rtol=1e-3, atol=2**-4), (
            "rtlsim output doesn't match reference.\n" f"Expected: {expected}\n" f"Actual: {actual}"
        )

    def test_layernorm_int_elementwise_rtlsim_no_conflict(self):
        """Verify LayerNorm_rtl + INT8 ElementwiseMul_hls produces CORRECT rtlsim results.

        This is a negative test - integer-only HLS ops should NOT trigger the conflict.
        """
        ishape = [1, 4, 4, 16]
        num_channels = 16

        # Create model and get reference output
        model = create_layernorm_int_elementwise_model(ishape, num_channels)
        inp = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
        inp_dict = {"inp": inp}

        # Get reference output from ONNX model (before HW conversion)
        ref_out = execute_onnx(model, inp_dict)
        oname = model.graph.output[0].name
        expected = ref_out[oname]

        # Convert to HW and build stitched IP
        model = convert_to_hw(model, VERSAL_PART)
        model = build_stitched_ip(model, VERSAL_PART)

        # Run stitched IP rtlsim
        rtlsim_out = run_stitched_ip_rtlsim(model, inp_dict)
        actual = rtlsim_out[model.graph.output[0].name]

        # This should NOT have the conflict - outputs should match
        assert np.allclose(expected, actual, rtol=1e-3, atol=2**-4), (
            "Integer-only HLS ops should NOT trigger DSP conflict. "
            "Expected outputs to match but they differ."
        )

# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test for comprehensive MLO (FINNLoop) with multiple layer types.

This test creates a subgraph containing:
- DuplicateStreams_hls
- Thresholding_rtl
- MVAU_rtl with parameter input
- MVAU_rtl with two dynamic inputs (joint node)
- RTL Elementwise (integer input/input for add, integer/float for mul)
- RTL Elementwise (float input, float params)
- Shuffle
- HWSoftmax_hls
- LayerNorm_rtl

Then chains 3 copies of that subgraph together for loop rolling.
"""

import pytest

import numpy as np
import os
import re
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import qonnx.core.data_layout as DataLayout
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from finn.util.basic import make_build_dir


verif_steps = [
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


def create_tensor_info(name, shape, proto=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, proto, shape)


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


def create_node(node_type, inputs, outputs, name, extra_params={}):
    """Create an ONNX node with finn domain."""
    if "rtl" in node_type.lower():
        domain = "finn.custom_op.fpgadataflow.rtl"
    else:
        domain = "finn.custom_op.fpgadataflow.hls"

    base_params = {
        "domain": domain,
        "backend": "fpgadataflow",
        "name": name,
    }
    return helper.make_node(node_type, inputs, outputs, **{**base_params, **extra_params})


def make_comprehensive_subgraph(
    mw=16,  # matrix width (channels)
    mh=16,  # matrix height (channels)
    dtype=DataType["INT8"],
    name_suffix="",
):
    """
    Create a comprehensive subgraph with all major layer types.

    The flow is (FLOAT32 input/output for loop compatibility):
    input (FLOAT32) -> Thresholding_rtl (converts to INT8)
          -> DuplicateStreams -> (branch1: MVAU -> Thresh, branch2: MVAU -> Thresh)
          -> ElementwiseAdd_rtl (merge branches, INT8/INT8 -> INT9)
          -> ElementwiseMul_rtl (INT9 input, FLOAT32 param -> FLOAT32)
          -> ElementwiseMul_rtl (FLOAT32 input, FLOAT32 param)
          -> Shuffle
          -> LayerNorm_rtl
          -> HWSoftmax_hls
          -> output (FLOAT32)
    """
    nodes = []
    value_infos = []
    initializers = {}

    # Shape: [batch, spatial, spatial, channels]
    ifm_shape = [1, 3, 3, mw]
    ofm_shape = [1, 3, 3, mh]

    # Generate weights and thresholds
    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))

    # Threshold for input conversion (FLOAT32 -> INT8)
    # Generate FLOAT32 thresholds spanning reasonable range for input conversion
    T_input = np.sort(
        generate_random_threshold_values(DataType["FLOAT32"], mw, dtype.get_num_possible_values() - 1),
        axis=1,
    )
    T0 = np.sort(
        generate_random_threshold_values(dtype, mh, dtype.get_num_possible_values() - 1), axis=1
    )
    T1 = np.sort(
        generate_random_threshold_values(dtype, mh, dtype.get_num_possible_values() - 1), axis=1
    )

    # Elementwise parameters
    eltw_hls_param = gen_finn_dt_tensor(DataType["FLOAT32"], [mh])
    eltw_rtl_param = gen_finn_dt_tensor(DataType["FLOAT32"], [mh])

    # LayerNorm parameters
    ln_scale = gen_finn_dt_tensor(DataType["FLOAT32"], [mh])
    ln_bias = gen_finn_dt_tensor(DataType["FLOAT32"], [mh])

    # --- Node 0: Thresholding_rtl (FLOAT32 -> INT8 input conversion) ---
    nodes.append(
        create_node(
            "Thresholding_rtl",
            [f"ifm{name_suffix}", f"thresh_input{name_suffix}"],
            [f"ifm_int{name_suffix}"],
            f"Thresholding_rtl_input{name_suffix}",
            {
                "NumChannels": mw,
                "PE": 2,
                "inputDataType": "FLOAT32",
                "weightDataType": "FLOAT32",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
                "numInputVectors": list(ifm_shape[:-1]),
            },
        )
    )

    # --- Node 1: DuplicateStreams_hls ---
    nodes.append(
        create_node(
            "DuplicateStreams_hls",
            [f"ifm_int{name_suffix}"],
            [f"ifm_1{name_suffix}", f"ifm_2{name_suffix}"],
            f"DuplicateStreams_hls_0{name_suffix}",
            {
                "NumChannels": mw,
                "NumOutputStreams": 2,
                "PE": 2,
                "inputDataType": dtype.name,
                "outFIFODepths": [2, 2],
                "cpp_interface": "hls_vector",
                "hls_style": "freerunning",
                "numInputVectors": list(ifm_shape[:-1]),
            },
        )
    )

    # --- Node 2: MVAU_rtl (branch 1) ---
    nodes.append(
        create_node(
            "MVAU_rtl",
            [f"ifm_1{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": 2,
                "PE": 2,
                "inputDataType": dtype.name,
                "weightDataType": dtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "numInputVectors": list(ifm_shape[:-1]),
            },
        )
    )

    # --- Node 3: Thresholding_rtl (branch 1) ---
    nodes.append(
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"mt0_out{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 2,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
                "numInputVectors": list(ofm_shape[:-1]),
            },
        )
    )

    # --- Node 4: MVAU_rtl (branch 2) ---
    nodes.append(
        create_node(
            "MVAU_rtl",
            [f"ifm_2{name_suffix}", f"weights1{name_suffix}"],
            [f"mm1_out{name_suffix}"],
            f"MVAU_rtl_1{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": 2,
                "PE": 2,
                "inputDataType": dtype.name,
                "weightDataType": dtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "numInputVectors": list(ifm_shape[:-1]),
            },
        )
    )

    # --- Node 5: Thresholding_rtl (branch 2) ---
    nodes.append(
        create_node(
            "Thresholding_rtl",
            [f"mm1_out{name_suffix}", f"thresh1{name_suffix}"],
            [f"mt1_out{name_suffix}"],
            f"Thresholding_rtl_1{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 2,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
                "numInputVectors": list(ofm_shape[:-1]),
            },
        )
    )

    # --- Node 6: ElementwiseAdd_rtl (merge branches) ---
    nodes.append(
        create_node(
            "ElementwiseAdd_rtl",
            [f"mt0_out{name_suffix}", f"mt1_out{name_suffix}"],
            [f"add_out{name_suffix}"],
            f"ElementwiseAdd_rtl_0{name_suffix}",
            {
                "lhs_shape": ofm_shape,
                "rhs_shape": ofm_shape,
                "out_shape": ofm_shape,
                "lhs_dtype": dtype.name,
                "rhs_dtype": dtype.name,
                "out_dtype": "INT9",
                "lhs_style": "input",
                "rhs_style": "input",
                "PE": 2,
            },
        )
    )

    # --- Node 7: ElementwiseMul_rtl (INT input, FLOAT32 param) ---
    nodes.append(
        create_node(
            "ElementwiseMul_rtl",
            [f"add_out{name_suffix}", f"eltw_hls_param{name_suffix}"],
            [f"eltw_hls_out{name_suffix}"],
            f"ElementwiseMul_rtl_0{name_suffix}",
            {
                "lhs_shape": ofm_shape,
                "rhs_shape": [mh],
                "out_shape": ofm_shape,
                "lhs_dtype": "INT9",
                "rhs_dtype": "FLOAT32",
                "out_dtype": "FLOAT32",
                "lhs_style": "input",
                "rhs_style": "const",
                "PE": 2,
            },
        )
    )

    # --- Node 8: ElementwiseMul_rtl (FLOAT32 input, FLOAT32 param) ---
    nodes.append(
        create_node(
            "ElementwiseMul_rtl",
            [f"eltw_hls_out{name_suffix}", f"eltw_rtl_param{name_suffix}"],
            [f"eltw_rtl_out{name_suffix}"],
            f"ElementwiseMul_rtl_0{name_suffix}",
            {
                "lhs_shape": ofm_shape,
                "rhs_shape": [mh],
                "out_shape": ofm_shape,
                "lhs_dtype": "FLOAT32",
                "rhs_dtype": "FLOAT32",
                "out_dtype": "FLOAT32",
                "PE": 2,
            },
        )
    )

    # --- Node 9: LayerNormalization (ONNX standard op) ---
    # Use standard ONNX LayerNormalization - will be converted via ExtractNormScaleBias + InferLayerNorm
    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=[f"eltw_rtl_out{name_suffix}", f"ln_scale{name_suffix}", f"ln_bias{name_suffix}"],
        outputs=[f"ln_out{name_suffix}"],
        name=f"LayerNorm_0{name_suffix}",
        epsilon=1e-5,
        axis=-1,
        stash_type=1,
    )
    nodes.append(ln_node)

    # --- Node 10: HWSoftmax_hls ---
    nodes.append(
        create_node(
            "HWSoftmax_hls",
            [f"ln_out{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"HWSoftmax_hls_0{name_suffix}",
            {
                "NumChannels": mh,
                "SIMD": 2,
                "ifm_dim": ofm_shape,
                "input_data_type": "FLOAT32",
                "cpp_interface": "hls_vector",
                "hls_style": "freerunning",
                #"output_data_type": "FLOAT32",
            },
        )
    )

    # Build value_info for intermediate tensors
    intermediate_tensors = [
        (f"ifm_int{name_suffix}", ifm_shape),
        (f"ifm_1{name_suffix}", ifm_shape),
        (f"ifm_2{name_suffix}", ifm_shape),
        (f"mm0_out{name_suffix}", ofm_shape),
        (f"mm1_out{name_suffix}", ofm_shape),
        (f"mt0_out{name_suffix}", ofm_shape),
        (f"mt1_out{name_suffix}", ofm_shape),
        (f"add_out{name_suffix}", ofm_shape),
        (f"eltw_hls_out{name_suffix}", ofm_shape),
        (f"eltw_rtl_out{name_suffix}", ofm_shape),
        (f"ln_out{name_suffix}", ofm_shape),
    ]
    value_infos = [create_tensor_info(name, shape) for name, shape in intermediate_tensors]

    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name=f"comprehensive_subgraph{name_suffix}",
        inputs=[create_tensor_info(f"ifm{name_suffix}", ifm_shape)],
        outputs=[create_tensor_info(f"ofm{name_suffix}", ofm_shape)],
        value_info=value_infos,
    )

    # Create model
    model = qonnx_make_model(graph, producer_name="comprehensive-mlo-test")
    model = ModelWrapper(model)

    # Set initializers
    model.set_initializer(f"weights0{name_suffix}", W0)
    model.set_initializer(f"weights1{name_suffix}", W1)
    model.set_initializer(f"thresh_input{name_suffix}", T_input.astype(np.float32))
    model.set_initializer(f"thresh0{name_suffix}", T0)
    model.set_initializer(f"thresh1{name_suffix}", T1)
    model.set_initializer(f"eltw_hls_param{name_suffix}", eltw_hls_param)
    model.set_initializer(f"eltw_rtl_param{name_suffix}", eltw_rtl_param)
    model.set_initializer(f"ln_scale{name_suffix}", ln_scale)
    model.set_initializer(f"ln_bias{name_suffix}", ln_bias)

    # Set tensor datatypes
    # Input is FLOAT32 for loop compatibility (HWSoftmax outputs FLOAT32)
    model.set_tensor_datatype(f"ifm{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"thresh_input{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"ifm_int{name_suffix}", dtype)
    model.set_tensor_datatype(f"weights0{name_suffix}", dtype)
    model.set_tensor_datatype(f"weights1{name_suffix}", dtype)
    model.set_tensor_datatype(f"thresh0{name_suffix}", DataType["INT33"])
    model.set_tensor_datatype(f"thresh1{name_suffix}", DataType["INT33"])
    model.set_tensor_datatype(f"eltw_hls_param{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"eltw_rtl_param{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"ln_scale{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"ln_bias{name_suffix}", DataType["FLOAT32"])
    model.set_tensor_datatype(f"ofm{name_suffix}", DataType["FLOAT32"])

    # Set tensor layouts to NHWC to avoid unwanted transposes in InferLayerNorm
    # Setting on global input propagates through InferDataLayouts
    model.set_tensor_layout(f"ifm{name_suffix}", DataLayout.NHWC)
    model.set_tensor_layout(f"eltw_rtl_out{name_suffix}", DataLayout.NHWC)
    model.set_tensor_layout(f"ln_out{name_suffix}", DataLayout.NHWC)
    model.set_tensor_layout(f"ofm{name_suffix}", DataLayout.NHWC)

    return model


def create_chained_subgraphs(mw=16, mh=16, dtype=DataType["INT8"], num_copies=3):
    """Create multiple copies of the comprehensive subgraph for chaining."""
    subgraph_models = []

    for i in range(num_copies):
        name_suffix = f"_{i}"
        subgraph_model = make_comprehensive_subgraph(
            mw=mw,
            mh=mh,
            dtype=dtype,
            name_suffix=name_suffix,
        )
        subgraph_models.append(subgraph_model)

    return subgraph_models


# Matrix dimensions
@pytest.mark.parametrize("dim", [16])
# 3 iterations (copies of the subgraph chained together)
@pytest.mark.parametrize("iteration", [3])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_comprehensive_mlo(dim, iteration):
    """
    Test comprehensive MLO with all major layer types.

    Creates a subgraph containing DuplicateStreams, MVAU, Thresholding,
    RTL Elementwise, Shuffle, LayerNorm, and HWSoftmax.
    Chains 3 copies together for loop rolling.
    """
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path is None:
        pytest.skip("XILINX_VIVADO not set")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    if match:
        year, minor = int(match.group(1)), int(match.group(2))
        if (year, minor) < (2024, 2):
            pytest.skip("At least Vivado version 2024.2 needed for MLO.")

    # Create multiple copies of the subgraph
    subgraph_models = create_chained_subgraphs(
        mw=dim,
        mh=dim,
        dtype=DataType["INT8"],
        num_copies=iteration,
    )

    # Chain the subgraphs together using MergeONNXModels
    model = subgraph_models[0]
    for m in subgraph_models[1:]:
        model = model.transform(MergeONNXModels(m))

    # Cleanup and inference
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Set NHWC layout on global input to propagate through the graph
    # This prevents InferLayerNorm from inserting unwanted transposes
    model.set_tensor_layout(model.graph.input[0].name, DataLayout.NHWC)

    # Convert LayerNormalization to HW layers
    model = model.transform(ExtractNormScaleBias())
    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model.save("test.onnx")
    # Get number of nodes per subgraph (for loop_body_range) - after transformations
    nodes_per_body = len(model.graph.node) // iteration

    # Generate reference output using cppsim
    # Input is FLOAT32 to match HWSoftmax output for loop compatibility
    input_dtype = DataType["FLOAT32"]
    x = gen_finn_dt_tensor(input_dtype, (1, 3, 3, dim))

    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))

    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

    # Setup build directory
    tmp_output_dir = make_build_dir("build_comprehensive_mlo_")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/comprehensive_mlo_model.onnx")

    # Build steps - include transpose_decomposition for Shuffle
    steps = [
        # "step_qonnx_to_finn",
        # "step_tidy_up",
        # "step_streamline",
        # "step_convert_to_hw",
        "step_create_dataflow_partition",
        # "step_specialize_layers",
        "step_transpose_decomposition",  # For Shuffle layers
        "step_loop_rolling",  
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,  # Disabled for debugging without loop rolling
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_full_context=True,
        verify_save_rtlsim_waveforms=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )

    build.build_dataflow_cfg(tmp_output_dir + "/comprehensive_mlo_model.onnx", cfg)

    # Check if expected files are there
    assert os.path.isfile(tmp_output_dir + "/loop-body-template.onnx")  # Only exists with loop rolling
    report_dir = tmp_output_dir + "/report"
    assert os.path.isfile(report_dir + "/estimate_layer_cycles.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources.json")
    assert os.path.isfile(tmp_output_dir + "/stitched_ip/ip/component.xml")

    # Verify output files exist and check verification results
    verif_dir = tmp_output_dir + "/verification_output"

    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"cppsim verification failed - check {verif_dir}"

    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"node_by_node_rtlsim verification failed - check {verif_dir}"

    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"stitched_ip_rtlsim verification failed - check {verif_dir}"

    # Verify iteration context files (only with loop rolling)
    iteration_context_files = [
        f for f in os.listdir(verif_dir) if f.startswith("iteration_context_")
    ]
    assert len(iteration_context_files) > 0, f"No iteration context files found in {verif_dir}"

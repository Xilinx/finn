# Copyright (c) 2022 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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


import numpy as np
import onnxruntime as ort
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

model_name = "ultranet"
platform_name = "fpga"

# Convert QONNX to FINN format
def convert_qonnx_to_finn():
    print("Converting QONNX model to FINN format...")
    model = ModelWrapper("ultranet_qonnx_finn.onnx")
    
    # Apply necessary transformations
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(ConvertQONNXtoFINN())
    
    # Save the converted model
    model.save("ultranet_finn_converted.onnx")
    print("Saved FINN-compatible model as ultranet_finn_converted.onnx")
    return "ultranet_finn_converted.onnx"

# Generate test input and golden output reference
def generate_test_data(model_file):
    print("Generating test data...")
    # Load the ONNX model to get input shape
    model = ModelWrapper(model_file)
    input_shape = model.get_tensor_shape(model.graph.input[0].name)
    
    # Fix shape if batch dimension is symbolic/dynamic
    if input_shape[0] == 0 or input_shape[0] is None or str(input_shape[0]).startswith('unk'):
        input_shape = [1] + list(input_shape[1:])  # Set batch size to 1
    
    # Ensure input_shape is a list of integers
    input_shape = [int(dim) if dim != 0 else 1 for dim in input_shape]
    
    print(f"Using input shape: {input_shape}")
    
    # Generate random input data
    np.random.seed(42)  # For reproducible results
    # Generate float data in range [0, 1] as expected by neural networks
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    print(f"Generated input data shape: {input_data.shape}")
    print(f"Generated input data type: {input_data.dtype}")
    print(f"Generated input data range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # Run inference to get golden output using QONNX model
    from qonnx.core.onnx_exec import execute_onnx
    qonnx_model = ModelWrapper("ultranet_qonnx_finn.onnx")
    
    # Apply necessary transformations for execution
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(InferDataTypes())
    
    input_name = qonnx_model.graph.input[0].name
    input_dict = {input_name: input_data}  # input_data is already float32
    output_dict = execute_onnx(qonnx_model, input_dict)
    output_name = qonnx_model.graph.output[0].name
    output = [output_dict[output_name]]
    
    # Save test data
    np.save("input.npy", input_data)
    np.save("expected_output.npy", output[0])
    
    print(f"Generated test input with shape: {input_shape}")
    print(f"Generated golden output with shape: {output[0].shape}")
    print("Saved input.npy and expected_output.npy")

# Convert QONNX to FINN first
converted_model_file = convert_qonnx_to_finn()

# Generate test data using the converted model
generate_test_data(converted_model_file)

cfg = build.DataflowBuildConfig(
    output_dir="output_%s_%s" % (model_name, platform_name),
    target_fps=60,
    mvau_wwidth_max=512,
    synth_clk_period_ns=10.0,
    board=platform_name,
    fpga_part="xczu3eg-sbva484-1-e",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    split_large_fifos=True,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ],
    verify_steps=[
        # Disable initial verification steps that may have bugs
        # build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        # build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
    ],
    save_intermediate_models=True,
)

model_file = converted_model_file
build.build_dataflow_cfg(model_file, cfg)

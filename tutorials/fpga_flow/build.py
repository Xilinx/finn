# Copyright (c) 2022 Advanced Micro Devices, Inc.
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


# This file is intended to serve as an example showing how to set up custom builds
# using FINN. The custom build can be launched like this:
# ./run-docker.sh build_custom /path/to/folder


import numpy as np
import os

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.util.data_packing as dpk
from finn.custom_op.registry import getCustomOp

model_name = "tfc_w1a1"
platform_name = "fpga"


def custom_step_gen_tb_and_io(model, cfg):
    sim_output_dir = cfg.output_dir + "/sim"
    os.makedirs(sim_output_dir, exist_ok=True)
    # load the provided input data
    inp_data = np.load("input.npy")
    batchsize = inp_data.shape[0]
    # permute input image from NCHW -> NHWC format (needed by FINN)
    # this example (MNIST) only has 1 channel, which means this doesn't
    # really do anything in terms of data layout changes, but provided for
    # completeness
    inp_data = np.transpose(inp_data, (0, 2, 3, 1))
    # this network is an MLP and takes in flattened input
    inp_data = inp_data.reshape(batchsize, -1)
    # query the parallelism-dependent folded input shape from the
    # node consuming the graph input
    inp_name = model.graph.input[0].name
    inp_node = getCustomOp(model.find_consumer(inp_name))
    inp_shape_folded = list(inp_node.get_folded_input_shape())
    inp_stream_width = inp_node.get_instream_width_padded()
    # fix first dimension (N: batch size) to correspond to input data
    # since FINN model itself always uses N=1
    inp_shape_folded[0] = batchsize
    inp_shape_folded = tuple(inp_shape_folded)
    inp_dtype = model.get_tensor_datatype(inp_name)
    # now re-shape input data into the folded shape and do hex packing
    inp_data = inp_data.reshape(inp_shape_folded)
    inp_data_packed = dpk.pack_innermost_dim_as_hex_string(
        inp_data, inp_dtype, inp_stream_width, prefix="", reverse_inner=True
    )
    np.savetxt(sim_output_dir + "/input.dat", inp_data_packed, fmt="%s", delimiter="\n")
    # load expected output and calculate folded shape
    exp_out = np.load("expected_output.npy")
    out_name = model.graph.output[0].name
    out_node = getCustomOp(model.find_producer(out_name))
    out_shape_folded = list(out_node.get_folded_output_shape())
    out_stream_width = out_node.get_outstream_width_padded()
    out_shape_folded[0] = batchsize
    out_shape_folded = tuple(out_shape_folded)
    out_dtype = model.get_tensor_datatype(out_name)
    exp_out = exp_out.reshape(out_shape_folded)
    out_data_packed = dpk.pack_innermost_dim_as_hex_string(
        exp_out, out_dtype, out_stream_width, prefix="", reverse_inner=True
    )
    np.savetxt(
        sim_output_dir + "/expected_output.dat",
        out_data_packed,
        fmt="%s",
        delimiter="\n",
    )
    # fill in testbench template
    with open("templates/finn_testbench.template.sv", "r") as f:
        testbench_sv = f.read()
    testbench_sv = testbench_sv.replace("@N_SAMPLES@", str(batchsize))
    testbench_sv = testbench_sv.replace("@IN_STREAM_BITWIDTH@", str(inp_stream_width))
    testbench_sv = testbench_sv.replace("@OUT_STREAM_BITWIDTH@", str(out_stream_width))
    testbench_sv = testbench_sv.replace(
        "$IN_BEATS_PER_SAMPLE@", str(np.prod(inp_shape_folded[:-1]))
    )
    testbench_sv = testbench_sv.replace(
        "$OUT_BEATS_PER_SAMPLE@", str(np.prod(out_shape_folded[:-1]))
    )
    testbench_sv = testbench_sv.replace("@TIMEOUT_CYCLES@", "1000")
    with open(sim_output_dir + "/finn_testbench.sv", "w") as f:
        f.write(testbench_sv)
    # fill in testbench project creator template
    with open("templates/make_sim_proj.template.tcl", "r") as f:
        testbench_tcl = f.read()
    testbench_tcl = testbench_tcl.replace("@DCP_ROOT@", "../stitched_ip")
    with open(sim_output_dir + "/make_sim_proj.tcl", "w") as f:
        f.write(testbench_tcl)

    return model


build_steps = build_cfg.default_build_dataflow_steps + [custom_step_gen_tb_and_io]


cfg = build.DataflowBuildConfig(
    steps=build_steps,
    start_step="custom_step_gen_tb_and_io",
    board=platform_name,
    output_dir="output_%s_%s" % (model_name, platform_name),
    synth_clk_period_ns=10.0,
    folding_config_file="folding_config.json",
    fpga_part="xczu3eg-sbva484-1-e",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    stitched_ip_gen_dcp=True,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
    ],
    verify_steps=[
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
    ],
    save_intermediate_models=True,
)
model_file = "model.onnx"
build.build_dataflow_cfg(model_file, cfg)

# Copyright (c) 2020 Xilinx, Inc.
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


import finn.util.build_dataflow as build

model_name = "tfc_w1a1"
platform_name = "Pynq-Z1"

cfg = build.DataflowBuildConfig(
    output_dir="output_%s_%s" % (model_name, platform_name),
    target_fps=100000,
    mvau_wwidth_max=10000,
    # can specify detailed folding/FIFO/etc config with:
    # folding_config_file="folding_config.json",
    synth_clk_period_ns=10.0,
    board=platform_name,
    shell_flow_type=build.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build.DataflowOutputType.PYNQ_DRIVER,
        build.DataflowOutputType.STITCHED_IP,
        build.DataflowOutputType.BITFILE,
    ],
    save_intermediate_models=True,
)
model_file = "model.onnx"
build.build_dataflow_cfg(model_file, cfg)

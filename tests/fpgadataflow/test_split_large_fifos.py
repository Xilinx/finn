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


import pytest

import json
import shutil
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from math import ceil

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir
from finn.util.test import get_trained_network_and_ishape


def fetch_test_model(topology, wbits=2, abits=2):
    tmp_output_dir = make_build_dir("build_fifosizing_%s_" % topology)
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir + "/model.onnx"
    BrevitasONNXManager.export(model, ishape, chkpt_name)
    return tmp_output_dir


def get_folding_cfg(depth=65536):
    cfg = dict()
    cfg["Defaults"] = dict()
    for i in range(3):
        key = "StreamingFIFO_" + str(i)
        cfg[key] = {"depth": depth, "ram_style": "auto", "impl_style": "rtl"}
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("depth", [16384, 65536, 45000])
def test_split_large_fifos(depth):
    tmp_output_dir = fetch_test_model("tfc")
    folding_cfg = get_folding_cfg(depth)
    with open(tmp_output_dir + "/folding_config.json", "w") as f:
        json.dump(folding_cfg, f, indent=2)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=False,
        split_large_fifos=True,
        folding_config_file=tmp_output_dir + "/folding_config.json",
        target_fps=10000,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        rtlsim_batch_size=100,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
        default_mem_mode=build_cfg.ComputeEngineMemMode.DECOUPLED,
    )
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        est_data = json.load(f)
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        sim_data = json.load(f)
    assert (
        float(sim_data["throughput[images/s]"])
        / float(est_data["estimated_throughput_fps"])
        > 0.9
    )
    with open(tmp_output_dir + "/final_hw_config.json") as f:
        hw_config = json.load(f)
    n = 0
    for key in hw_config:
        if "StreamingFIFO" in key:
            n += 1
    assert n == 3 * ceil(depth / 32768) + 1

    shutil.rmtree(tmp_output_dir)

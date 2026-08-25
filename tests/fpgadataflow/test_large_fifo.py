# Copyright (C) 2026, Advanced Micro Devices, Inc.
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
import torch
from brevitas.export import export_qonnx

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir, part_map, robust_rmtree
from finn.util.test import get_trained_network_and_ishape

# deep enough to need more than one BRAM/URAM primitive, and not a power of two,
# so it also covers fifo.sv's lo+hi memory space decomposition
DEPTH = 45000


def fetch_test_model(topology, wbits=2, abits=2):
    tmp_output_dir = make_build_dir("build_large_fifo_%s_" % topology)
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir + "/model.onnx"
    export_qonnx(model, torch.randn(ishape), chkpt_name)
    return tmp_output_dir


def get_folding_cfg(depth=DEPTH, ram_style="auto"):
    cfg = dict()
    cfg["Defaults"] = {
        "depth": [depth, ["StreamingFIFO_rtl"]],
        "ram_style": [ram_style, ["StreamingFIFO_rtl"]],
    }
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
# "ultra" only on the boards that have URAM. AUP-ZU3_8GB is an xczu3eg, which has no
# URAM, and is left at "auto": fifo.sv would still elaborate the ultra branch at this
# depth, but Vivado drops the attribute (Synth 8-12187) and backs the array with BRAM,
# so asking for it explicitly would test nothing the other two boards do not.
@pytest.mark.parametrize(
    "board, ram_style", [("AUP-ZU3_8GB", "auto"), ("ZCU104", "ultra"), ("VEK280", "ultra")]
)
def test_large_fifo_is_not_split(board, ram_style):
    versal = board in ("VEK280", "VCK190")
    tmp_output_dir = fetch_test_model("tfc")
    folding_cfg = get_folding_cfg(DEPTH, ram_style)
    with open(tmp_output_dir + "/folding_config.json", "w") as f:
        json.dump(folding_cfg, f, indent=2)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=False,
        folding_config_file=tmp_output_dir + "/folding_config.json",
        target_fps=10000,
        synth_clk_period_ns=10.0,
        board=None if versal else board,
        fpga_part=part_map[board] if versal else None,
        rtlsim_batch_size=100,
        shell_flow_type=None if versal else build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        est_data = json.load(f)
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        sim_data = json.load(f)
    assert (
        float(sim_data["throughput[images/s]"]) / float(est_data["estimated_throughput_fps"]) > 0.9
    )
    # every FIFO must build as one instance at exactly the requested depth:
    # no chain of smaller FIFOs, no rounding up to a power of two
    with open(tmp_output_dir + "/final_hw_config.json") as f:
        hw_cfg = json.load(f)
    fifos = {k: v for k, v in hw_cfg.items() if k.startswith("StreamingFIFO_rtl")}
    assert len(fifos) > 0
    for name, attrs in fifos.items():
        assert attrs["depth"] == DEPTH, "%s was split or rounded: %d" % (name, attrs["depth"])

    robust_rmtree(tmp_output_dir)

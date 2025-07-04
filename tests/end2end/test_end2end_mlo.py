# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# * Neither the name of FINN nor the names of its
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

from qonnx.transformation.general import GiveUniqueNodeNames

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

verif_steps = [
    # "folded_hls_cppsim",
    # "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


def custom_give_unique_names(model, cfg):
    model = model.transform(GiveUniqueNodeNames(), apply_to_subgraphs=True)
    return model


steps = [
    #    "step_convert_to_hw",
    #    "step_create_dataflow_partition",
    #    "step_specialize_layers",
    #    "step_target_fps_parallelization",
    #    "step_apply_folding_config",
    "step_minimize_bit_width",
    custom_give_unique_names,
    #    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    # "step_measure_rtlsim_performance",
]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mlo():
    tmp_output_dir = make_build_dir("build_mlo")
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        verify_steps=verif_steps,
        verify_save_rtlsim_waveforms=True,
        # stitched_ip_gen_dcp=True,
        generate_outputs=[
            # build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            # build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    build.build_dataflow_cfg("finn_loop.onnx", cfg)

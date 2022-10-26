# Copyright (c) 2020, Xilinx
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

import pkg_resources as pk

import pytest

import numpy as np
import os
from shutil import copytree

from finn.builder.build_dataflow import build_dataflow_directory
from finn.util.basic import make_build_dir


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_build_dataflow_directory():
    test_dir = make_build_dir("test_build_dataflow_directory_")
    target_dir = test_dir + "/build_dataflow"
    example_data_dir = pk.resource_filename("finn.qnn-data", "build_dataflow/")
    copytree(example_data_dir, target_dir)
    build_dataflow_directory(target_dir)
    # check the generated files
    output_dir = target_dir + "/output_tfc_w1a1_Pynq-Z1"
    assert os.path.isfile(output_dir + "/time_per_step.json")
    assert os.path.isfile(output_dir + "/auto_folding_config.json")
    assert os.path.isfile(output_dir + "/final_hw_config.json")
    assert os.path.isfile(output_dir + "/stitched_ip/ip/component.xml")
    assert os.path.isfile(output_dir + "/driver/driver.py")
    assert os.path.isfile(output_dir + "/report/estimate_layer_cycles.json")
    assert os.path.isfile(output_dir + "/report/estimate_layer_resources.json")
    assert os.path.isfile(output_dir + "/report/rtlsim_perf_batch_1.vcd")
    assert os.path.isfile(
        output_dir + "/report/estimate_layer_config_alternatives.json"
    )
    assert os.path.isfile(output_dir + "/report/estimate_network_performance.json")
    assert os.path.isfile(output_dir + "/report/ooc_synth_and_timing.json")
    assert os.path.isfile(output_dir + "/report/rtlsim_performance.json")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.bit")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.hwh")
    assert os.path.isfile(output_dir + "/report/post_synth_resources.xml")
    assert os.path.isfile(output_dir + "/report/post_route_timing.rpt")
    # verification outputs
    verif_batchsize = np.load(target_dir + "/input.npy").shape[0]
    for i in range(verif_batchsize):
        verify_out_dir = output_dir + "/verification_output"
        assert os.path.isfile(
            verify_out_dir + f"/verify_initial_python_{i}_SUCCESS.npy"
        )
        assert os.path.isfile(
            verify_out_dir + f"/verify_streamlined_python_{i}_SUCCESS.npy"
        )
        assert os.path.isfile(
            verify_out_dir + f"/verify_folded_hls_cppsim_{i}_SUCCESS.npy"
        )
        assert os.path.isfile(
            verify_out_dir + f"/verify_stitched_ip_rtlsim_{i}_SUCCESS.npy"
        )
        assert os.path.isfile(output_dir + f"/report/verify_rtlsim_{i}.vcd")

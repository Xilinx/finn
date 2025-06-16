#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	Unit test for FINN XSI via rtlsim_multi_io adapter
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import pytest

try:
    import finn_xsi.adapter as finnxsi
except ModuleNotFoundError:
    finnxsi = None

import os
import shutil

from finn.util.basic import get_finn_root, make_build_dir


@pytest.mark.vivado
@pytest.mark.util
def test_finnxsi():
    testcase_root_dir = get_finn_root() + "/finn_xsi/testcase"
    sim_dir = make_build_dir("rtlsim_test_finnxsi_")
    shutil.copytree(testcase_root_dir, sim_dir, dirs_exist_ok=True)
    top_module_name = "StreamingEltwise_hls_0"
    source_list = [
        "StreamingEltwise_hls_0_flow_control_loop_pipe_no_ap_cont.v",
        "StreamingEltwise_hls_0_regslice_both.v",
        "StreamingEltwise_hls_0.v",
    ]
    sim_out_dir, out_so_relative_path = finnxsi.compile_sim_obj(
        top_module_name, source_list, sim_dir, debug=True
    )
    assert os.path.isfile(sim_out_dir + "/rtlsim.prj"), "Simulation .prj not found"
    assert os.path.isfile(sim_out_dir + "/" + out_so_relative_path), "Simulation object not found"
    sim = finnxsi.load_sim_obj(sim_out_dir, out_so_relative_path, tracefile="sim.wdb")
    finnxsi.reset_rtlsim(sim)

    i0 = [4, 6, 8, 10]
    i1 = [1, 2, 3, 4]
    exp = [x - y for x, y in zip(i0, i1)]
    io_dict = {"inputs": {"in0": i0, "in1": i1}, "outputs": {"out": []}}
    sname = "_V"
    cycles = finnxsi.rtlsim_multi_io(sim, io_dict, 4, sname)
    output = io_dict["outputs"]["out"]
    assert output == exp
    assert cycles == 8
    shutil.rmtree(sim_dir)

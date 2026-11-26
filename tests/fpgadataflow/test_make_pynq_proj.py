# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os

from finn.transformation.fpgadataflow import templates
from finn.transformation.fpgadataflow.make_pynq_proj import (
    VERSAL_SHELL_MAX_AXILITE,
    ZYNQ_SHELL_MAX_AXILITE,
    check_shell_axilite_count,
)
from finn.util.basic import make_build_dir, pynq_part_map, robust_rmtree
from finn.util.launcher import run_tool


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "versal, max_axilite", [(False, ZYNQ_SHELL_MAX_AXILITE), (True, VERSAL_SHELL_MAX_AXILITE)]
)
def test_shell_axilite_count_limit(versal, max_axilite):
    check_shell_axilite_count(max_axilite, versal, "board")
    with pytest.raises(ValueError, match=f"at most {max_axilite} AXI-Lite interfaces"):
        check_shell_axilite_count(max_axilite + 1, versal, "board")


@pytest.mark.fpgadataflow
def test_shell_templates_match_axilite_limit():
    # the Tcl checks of the shells are a backstop for the Python check
    for template, max_axilite in [
        (templates.custom_zynq_shell_template, ZYNQ_SHELL_MAX_AXILITE),
        (templates.custom_versal_shell_template, VERSAL_SHELL_MAX_AXILITE),
    ]:
        assert "if {$NUM_AXILITE > %d}" % max_axilite in template


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_zynq_shell_interconnect_accepts_axilite_limit():
    # the AXI Interconnect of the Zynq shell accepts the maximum and rejects one more
    build_dir = make_build_dir(prefix="axilite_limit_")
    tcl = os.path.join(build_dir, "check.tcl")
    with open(tcl, "w") as f:
        f.write(
            "create_project -in_memory -part %s\n"
            "create_bd_design t\n"
            'set v [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" '
            "-filter design_tool_contexts=~*IPI*]]\n"
            "create_bd_cell -type ip -vlnv $v ic\n"
            "foreach n {%d %d} {\n"
            "    set ok [expr {![catch {set_property CONFIG.NUM_MI $n [get_bd_cells ic]}]}]\n"
            '    puts "FINN_NUM_MI $n $ok"\n'
            "}\n"
            % (pynq_part_map["AUP-ZU3_8GB"], ZYNQ_SHELL_MAX_AXILITE, ZYNQ_SHELL_MAX_AXILITE + 1)
        )
    run_tool(["vivado", "-mode", "batch", "-nojournal", "-source", tcl], cwd=build_dir)
    # Read Vivado's own log rather than the transcript: a tool dispatcher may hand
    # back the tool's standard output incompletely (finn-lsf copies the LSF job
    # output before it is complete on NFS).
    with open(os.path.join(build_dir, "vivado.log")) as f:
        log = f.read()
    results = [line.split()[1:] for line in log.splitlines() if line.startswith("FINN_NUM_MI")]
    assert results == [
        [str(ZYNQ_SHELL_MAX_AXILITE), "1"],
        [str(ZYNQ_SHELL_MAX_AXILITE + 1), "0"],
    ], log[-2000:]
    robust_rmtree(build_dir)

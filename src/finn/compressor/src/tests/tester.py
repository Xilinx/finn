#!/usr/bin/env python
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vivado XSim wrapper for testing generated compressors."""

import subprocess
import re


def tester(test_loc, comp_loc):
    """Run Vivado XSim simulation to test a compressor.

    Args:
        test_loc: Path to testbench SystemVerilog file
        comp_loc: Path to compressor SystemVerilog file
    """
    args = (
        f"""rm -r xsim.dir/ &&
        xvlog -work work -sv ../res/glbl.v {test_loc} {comp_loc} -L unisims_ver --nolog &&
        xelab -L work -L unisims_ver -relax --nolog glbl tb &&
        xsim --nolog work.glbl#work.tb -R""").replace("\n", " ")
    print(args)
    try:
        ret = subprocess.run(args, capture_output=True, text=True, timeout=300,
                             shell=True, check=True)
    except subprocess.CalledProcessError as e:
        if e.returncode == 127:
            raise RuntimeError(
                "Could not call Vivado simulation tools. Did you source Vivado?")
        else:
            raise RuntimeError("Something failed during simulation.")
    if "$finish called at time" in ret.stdout:
        print("Simulation SUCCESS!")
    else:
        print("ERROR in Compressor Simulation!")
        error = re.findall("Error:.*\n.*\n", ret.stdout)[0].split("\n")
        print(f">> {error[0]}\n>> {error[1]}")
        exit(-2)

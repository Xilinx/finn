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

import os
import re

from finn.util.basic import launch_process_helper, which


def _extract_util_from_report(vivado_proj_folder, row_name):
    """Extract the Used column for a row in Vivado's utilization report."""

    log_path = os.path.join(vivado_proj_folder, "vivado.log")
    if not os.path.isfile(log_path):
        return None

    row_pattern = re.compile(r"^\|\s*%s\s*\|\s*([0-9.]+)\s*\|" % re.escape(row_name))
    with open(log_path, "r") as f:
        for line in f:
            match = row_pattern.match(line)
            if match is not None:
                return float(match.group(1))
    return None


def out_of_context_synth(
    verilog_dir,
    top_name,
    fpga_part="xczu3eg-sbva484-1-e",
    clk_name="ap_clk_0",
    clk_period_ns=5.0,
):
    "Run out-of-context Vivado synthesis, return resources and slack."

    # ensure that the OH_MY_XILINX envvar is set
    if "OHMYXILINX" not in os.environ:
        raise Exception("The environment variable OHMYXILINX is not defined.")
    # ensure that vivado is in PATH: source $VIVADO_PATH/settings64.sh
    if which("vivado") is None:
        raise Exception("vivado is not in PATH, ensure settings64.sh is sourced.")
    omx_path = os.environ["OHMYXILINX"]
    script = "vivadocompile.sh"
    # vivadocompile.sh <top-level-entity> <fp-ip-tcl-list> <clock-name>
    #                  <fpga-part> <clk-period-ns>
    call_omx = [
        "zsh",
        os.path.join(omx_path, script),
        top_name,
        "",
        clk_name,
        fpga_part,
        "%f" % float(clk_period_ns),
    ]
    launch_process_helper(call_omx, proc_env=os.environ.copy(), cwd=verilog_dir)

    vivado_proj_folder = "%s/results_%s" % (verilog_dir, top_name)
    res_counts_path = vivado_proj_folder + "/res.txt"

    with open(res_counts_path, "r") as myfile:
        res_data = myfile.read().split("\n")
    ret = {}
    ret["vivado_proj_folder"] = vivado_proj_folder
    util_report_rows = {
        "DSP": "DSPs",
    }
    for res_line in res_data:
        res_fields = res_line.split("=")
        print(res_fields)
        try:
            ret[res_fields[0]] = float(res_fields[1])
        except ValueError:
            util_value = None
            if res_fields[0] in util_report_rows:
                util_value = _extract_util_from_report(
                    vivado_proj_folder, util_report_rows[res_fields[0]]
                )
            if util_value is None:
                raise
            ret[res_fields[0]] = util_value
        except IndexError:
            ret[res_fields[0]] = 0
    if ret["WNS"] == 0:
        ret["fmax_mhz"] = 0
    else:
        ret["fmax_mhz"] = 1000.0 / (clk_period_ns - ret["WNS"])
    return ret

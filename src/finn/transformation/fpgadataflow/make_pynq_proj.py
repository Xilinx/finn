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
import warnings
import subprocess

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.util.basic import get_by_name, make_build_dir, roundup_to_integer_multiple

from . import templates


class MakePYNQProject(Transformation):
    """Create a Vivado PYNQ overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CreateStitchedIP transformation must have been previously run on
    the graph.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(
        self,
        platform,
        clk_name="ap_clk",
        rst_name="ap_rst_n",
        s_axis_if_name="s_axis_0",
        m_axis_if_name="m_axis_0",
        s_aximm_if_name="s_axi_control",
    ):
        super().__init__()
        self.platform = platform
        self.clk_name = clk_name
        self.rst_name = rst_name
        self.s_axis_if_name = s_axis_if_name
        self.m_axis_if_name = m_axis_if_name
        self.s_aximm_if_name = s_aximm_if_name

    def apply(self, model):
        pynq_shell_path = os.environ["PYNQSHELL_PATH"]
        if not os.path.isdir(pynq_shell_path):
            raise Exception("Ensure the PYNQ-HelloWorld utility repo is cloned.")
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
            raise Exception(
                "No stitched IPI design found, apply CreateStitchedIP first."
            )
        vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
        if vivado_stitch_vlnv is None:
            raise Exception(
                "No vlnv for stitched IP found, apply CreateStitchedIP first."
            )

        # collect list of all IP dirs
        ip_dirs = ["list"]
        for node in model.graph.node:
            ip_dir_attribute = get_by_name(node.attribute, "ip_path")
            assert (
                ip_dir_attribute is not None
            ), """Node attribute "ip_path" is
            empty. Please run transformation HLSSynth_ipgen first."""
            ip_dir_value = ip_dir_attribute.s.decode("UTF-8")
            assert os.path.isdir(
                ip_dir_value
            ), """The directory that should
            contain the generated ip blocks doesn't exist."""
            ip_dirs += [ip_dir_value]
        ip_dirs += [ipstitch_path + "/ip"]
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))

        # extract HLSCustomOp instances to get i/o stream widths
        i_tensor_name = model.graph.input[0].name
        o_tensor_name = model.graph.output[0].name
        first_node = getCustomOp(model.find_consumer(i_tensor_name))
        last_node = getCustomOp(model.find_producer(o_tensor_name))
        i_bits_per_cycle = first_node.get_instream_width()
        o_bits_per_cycle = last_node.get_outstream_width()
        # ensure i/o is padded to bytes
        i_bits_per_cycle_padded = roundup_to_integer_multiple(i_bits_per_cycle, 8)
        o_bits_per_cycle_padded = roundup_to_integer_multiple(o_bits_per_cycle, 8)
        assert (
            i_bits_per_cycle_padded % 8 == 0
        ), """Padded input bits are not a
        multiple of 8."""
        assert (
            o_bits_per_cycle_padded % 8 == 0
        ), """Padded output bits are not a
        multiple of 8."""
        in_bytes = i_bits_per_cycle_padded / 8
        out_bytes = o_bits_per_cycle_padded / 8
        in_if_name = self.s_axis_if_name
        out_if_name = self.m_axis_if_name
        clk_name = self.clk_name
        nrst_name = self.rst_name
        axi_lite_if_name = self.s_aximm_if_name
        vivado_ip_cache = os.getenv("VIVADO_IP_CACHE", default="")

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_pynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)
        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)

        # get metadata property clk_ns to calculate clock frequency
        clk_ns = float(model.get_metadata_prop("clk_ns"))
        if clk_ns not in [5.0, 10.0, 20.0]:
            warnings.warn(
                """The chosen frequency may lead to failure due to clock divider
                constraints."""
            )
        fclk_mhz = 1 / (clk_ns * 0.001)

        ip_config_tcl = templates.ip_config_tcl_template % (
            vivado_pynq_proj_dir,
            ip_dirs_str,
            vivado_pynq_proj_dir,
            synth_report_filename,
            vivado_stitch_vlnv,
            in_bytes,
            out_bytes,
            in_if_name,
            out_if_name,
            clk_name,
            nrst_name,
            axi_lite_if_name,
            vivado_ip_cache,
            fclk_mhz,
        )

        with open(vivado_pynq_proj_dir + "/ip_config.tcl", "w") as f:
            f.write(ip_config_tcl)
        # create a shell script for project creation and synthesis
        make_project_sh = vivado_pynq_proj_dir + "/make_project.sh"
        working_dir = os.environ["PWD"]
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        with open(make_project_sh, "w") as f:
            f.write(
                templates.call_pynqshell_makefile_template
                % (pynq_shell_path, self.platform, ipcfg, "block_design", working_dir)
            )
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        with open(synth_project_sh, "w") as f:
            f.write(
                templates.call_pynqshell_makefile_template
                % (pynq_shell_path, self.platform, ipcfg, "bitstream", working_dir)
            )
        # call the project creation script
        # synthesis script will be called with a separate transformation
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)

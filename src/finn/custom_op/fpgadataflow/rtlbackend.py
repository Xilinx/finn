# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

import numpy as np
import os
import shutil
import subprocess
from abc import ABC, abstractmethod

from finn import xsi
from finn.custom_op.fpgadataflow import templates
from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

finnxsi = xsi if xsi.is_available() else None


class RTLBackend(ABC):
    """RTLBackend class all custom ops that correspond to a module in finn-rtllib
    are using functionality of. Contains different functions every RTL
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new RTL custom op node."""

    def get_nodeattr_types(self):
        return {
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }

    @abstractmethod
    def generate_hdl(self, model, fpgapart, clk):
        pass

    def prepare_rtlsim(self, behav=False):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        verilog_files = self.get_rtl_file_list(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = finnxsi.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug, behav
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def get_verilog_paths(self):
        """Returns path to code gen directory. Can be overwritten to
        return additional paths to relevant verilog files"""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [code_gen_dir]

    @abstractmethod
    def get_rtl_file_list(self, abspath=False):
        """Returns list of rtl files. Needs to be filled by each node."""
        pass

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def code_generation_pack_ip(self, fpgapart):
        """Pack RTL as IP"""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # bundle all RTL source files into the codegen dir so the packaging tcl
        # (which globs *.v/*.sv/*.dat in the dir) picks them up. This keeps the
        # packaging uniform across ops without requiring per-op file copies.
        for src in self.get_rtl_file_list(abspath=True):
            dst = os.path.join(code_gen_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy(src, code_gen_dir)
        # prepare the IP packaging tcl template
        template = templates.ip_package_tcl
        self.code_gen_dict.clear()
        self.code_gen_dict["$TOPNAME$"] = [self.get_nodeattr("gen_top_module")]
        self.code_gen_dict["$PART$"] = [fpgapart]
        # note: setting the root dir as absolute can cause path problems
        # the ipgen script will be invoked from the sources dir so root_dir=. is OK
        self.code_gen_dict["$VERILOG_DIR$"] = ["."]
        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        f = open(os.path.join(code_gen_dir, "package_ip.tcl"), "w")
        f.write(template)
        f.close()
        # create a shell script and call Vivado to invoke the IP pkg script
        make_project_sh = code_gen_dir + "/make_ip.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(code_gen_dir))
            f.write("vivado -mode batch -source package_ip.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # set ipgen_path and ip_path to point to the new packaged IP
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        vlnv = "xilinx.com:hls:%s:1.0" % (self.get_nodeattr("gen_top_module"))
        self.set_nodeattr("ip_vlnv", vlnv)
        self.code_gen_dict.clear()

    def pack_as_ip(self):
        """Whether this node is instantiated as a standalone packaged IP (via the
        base code_generation_ipi, create_bd_cell -type ip -vlnv) and therefore
        needs to be packaged during code_generation_ipgen. Ops that override
        code_generation_ipi to add raw sources / build their own RTL hierarchy are
        not black-boxed and must not be packaged; those inherit this default,
        which returns False for them because they override code_generation_ipi."""
        return type(self).code_generation_ipi is RTLBackend.code_generation_ipi

    def code_generation_ipgen(self, model, fpgapart, clk):
        self.generate_hdl(model, fpgapart, clk)
        if self.pack_as_ip():
            self.code_generation_pack_ip(fpgapart)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        if mode == "rtlsim":
            node = self.onnx_node
            inputs = {}
            for i, inp in enumerate(node.input):
                exp_ishape = tuple(self.get_normal_input_shape(i))
                folded_ishape = self.get_folded_input_shape(i)
                inp_val = context[inp]
                assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
                assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
                export_idt = self.get_input_datatype(i)

                reshaped_input = inp_val.reshape(folded_ishape)
                np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
                nbits = self.get_instream_width(i)
                rtlsim_inp = npy_to_rtlsim_input(
                    "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
                )
                inputs["in%s" % i] = rtlsim_inp
            outputs = {}
            for o, outp in enumerate(node.output):
                outputs["out%s" % o] = []
            # assembled execution context
            io_dict = {"inputs": inputs, "outputs": outputs}

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)
            for o, outp in enumerate(node.output):
                rtlsim_output = io_dict["outputs"]["out%s" % o]
                odt = self.get_output_datatype(o)
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width(o)
                out_npy_path = "{}/output.npy".format(code_gen_dir)
                out_shape = self.get_folded_output_shape(o)
                rtlsim_output_to_npy(
                    rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
                )
                # load and reshape output
                exp_oshape = tuple(self.get_normal_output_shape(o))
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

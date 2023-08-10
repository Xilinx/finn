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
import math
import numpy as np
import os
import subprocess
import warnings
from qonnx.core.datatype import DataType
from shutil import copy

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_finn_root
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

from . import templates


class StreamingFIFO(HLSCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.strm_fifo_wrapper = templates.strm_fifo_wrapper

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                # FIFO depth
                "depth": ("i", True, 0),
                # folded shape of input/output
                "folded_shape": ("ints", True, []),
                # FINN DataTypes for inputs/outputs
                "dataType": ("s", True, ""),
                # Toggle between hls or IPI implementation
                # rtl - use the hls generated IP during stitching
                # vivado - use the AXI Infrastructure FIFO
                "impl_style": ("s", False, "rtl", {"rtl", "vivado"}),
                # FPGA resource type for FIFOs when impl_style is vivado
                # auto -- let Vivado decide
                # block -- use BRAM
                # distributed -- use LUTRAM
                # ultra -- use URAM (on UltraScale+)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "block", "distributed", "ultra"},
                ),
                # whether depth monitoring is enabled (impl_style=rtl only)
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
            }
        )

        return my_attrs

    def get_adjusted_depth(self):
        impl = self.get_nodeattr("impl_style")
        depth = self.get_nodeattr("depth")
        if impl == "vivado":
            old_depth = depth
            # round up depth to nearest power-of-2
            # Vivado FIFO impl may fail otherwise
            depth = (1 << (depth - 1).bit_length()) if impl == "vivado" else depth
            if old_depth != depth:
                warnings.warn(
                    "%s: rounding-up FIFO depth from %d to %d for impl_style=vivado"
                    % (self.onnx_node.name, old_depth, depth)
                )

        return depth

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == tuple(exp_ishape), "Unexpect input shape for StreamingFIFO."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        is_rtl = self.get_nodeattr("impl_style") == "rtl"
        is_depth_monitor = self.get_nodeattr("depth_monitor") == 1
        if is_rtl and is_depth_monitor:
            ret["ap_none"] = ["maxcount"]
        return ret

    def get_verilog_top_module_name(self):
        "Return the Verilog top module name for this node."

        node = self.onnx_node
        prefixed_top_name = "%s" % (node.name)
        return prefixed_top_name

    def code_generation_ipgen(self, model, fpgapart, clk):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_dir = "{}/project_{}/sol1/impl/verilog".format(code_gen_dir, self.onnx_node.name)
        os.makedirs(verilog_dir)
        # copy Q_srl.v from finn-rtllib to verilog directory
        memstream_dir = get_finn_root() + "/finn-rtllib/memstream/hdl/"
        Q_file = os.path.join(memstream_dir, "Q_srl.v")
        copy(Q_file, verilog_dir)

        # empty code gen dictionary for new entries
        self.code_gen_dict.clear()
        self.code_gen_dict["$TOPNAME$"] = ["{}".format(self.onnx_node.name)]
        self.code_gen_dict["$LAYER_NAME$"] = [
            "{}_{}".format(self.onnx_node.name, self.onnx_node.name)
        ]
        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()
        count_width = int(self.get_nodeattr("depth") - 1).bit_length()
        self.code_gen_dict["$COUNT_RANGE$"] = ["[{}:0]".format(count_width - 1)]
        self.code_gen_dict["$IN_RANGE$"] = ["[{}:0]".format(in_width - 1)]
        self.code_gen_dict["$OUT_RANGE$"] = ["[{}:0]".format(in_width - 1)]
        self.code_gen_dict["$WIDTH$"] = [str(in_width)]
        self.code_gen_dict["$DEPTH$"] = [str(self.get_nodeattr("depth"))]
        self.code_gen_dict["$HLS_SNAME$"] = [self.hls_sname()]

        template = self.strm_fifo_wrapper

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        f = open(os.path.join(verilog_dir, "{}.v".format(self.onnx_node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def ipgen_singlenode_code(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_dir = "{}/project_{}/sol1/impl/verilog".format(code_gen_dir, self.onnx_node.name)
        # prepare the IP packaging tcl template
        template = templates.ip_package_tcl
        self.code_gen_dict.clear()
        self.code_gen_dict["$TOPNAME$"] = ["{}".format(self.onnx_node.name)]
        # note: setting the root dir as absolute can cause path problems
        # the ipgen script will be invoked from the sources dir so root_dir=. is OK
        self.code_gen_dict["$VERILOG_DIR$"] = ["."]
        self.code_gen_dict["$HLS_SNAME$"] = [self.hls_sname()]
        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        f = open(os.path.join(verilog_dir, "package_ip.tcl"), "w")
        f.write(template)
        f.close()
        # create a shell script and call Vivado to invoke the IP pkg script
        make_project_sh = verilog_dir + "/make_ip.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(verilog_dir))
            f.write("vivado -mode batch -source package_ip.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # set ipgen_path and ip_path to point to the new packaged IP
        self.set_nodeattr("ipgen_path", verilog_dir)
        self.set_nodeattr("ip_path", verilog_dir)
        vlnv = "xilinx.com:hls:%s:1.0" % (self.onnx_node.name)
        self.set_nodeattr("ip_vlnv", vlnv)
        self.code_gen_dict.clear()

    def get_normal_input_shape(self, ind=0):
        depth = self.get_adjusted_depth()
        assert depth >= 2, """Depth is too low"""
        if depth > 256 and self.get_nodeattr("impl_style") == "rtl":
            warnings.warn("Depth is high, set between 2 and 256 for efficient SRL implementation")
        # derive normal shape from folded shape
        # StreamingFIFOs are inserted in between fpgadataflow nodes
        # the folded shape could be for example (1, nf, pe)
        # with nf (neuron folding): mh // pe
        # the normal input shape is in this case (1, mh)
        # so to achieve this the two inner dimensions are multiplied
        # and together with all previous dimensions
        # this gives the normal input shape

        folded_shape = self.get_nodeattr("folded_shape")
        # extract inner dimension
        inner_dim = folded_shape[-1]
        # multiply with the next inner dimension
        folding_factor = folded_shape[-2] * inner_dim
        normal_ishape = []
        # create the normal_ishape
        for i in range(len(folded_shape) - 2):
            normal_ishape.append(folded_shape[i])
        normal_ishape.append(folding_factor)

        return normal_ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_instream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        inp = context[node.input[0]]
        exp_shape = self.get_normal_input_shape()

        if mode == "cppsim":
            output = inp
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[node.output[0]] = output
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # create a npy file for the input of the node
            assert (
                str(inp.dtype) == "float32"
            ), """Input datatype is
                not float32 as expected."""
            expected_inp_shape = self.get_folded_input_shape()
            reshaped_input = inp.reshape(expected_inp_shape)
            if DataType[self.get_nodeattr("dataType")] == DataType["BIPOLAR"]:
                # store bipolar activations as binary
                reshaped_input = (reshaped_input + 1) / 2
                export_idt = DataType["BINARY"]
            else:
                export_idt = DataType[self.get_nodeattr("dataType")]
            # make copy before saving the array
            reshaped_input = reshaped_input.copy()
            np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            output = self.rtlsim(sim, inp)
            odt = DataType[self.get_nodeattr("dataType")]
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)
            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def global_includes(self):
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass

    def code_generation_ipi(self):
        impl_style = self.get_nodeattr("impl_style")
        if impl_style == "rtl":
            return super().code_generation_ipi()
        elif impl_style == "vivado":
            cmd = []
            node_name = self.onnx_node.name
            depth = self.get_adjusted_depth()
            ram_style = self.get_nodeattr("ram_style")
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate and configure DWC
            cmd.append(
                "create_bd_cell -type ip "
                "-vlnv xilinx.com:ip:axis_data_fifo:2.0 /%s/fifo" % node_name
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_DEPTH {%d}] "
                "[get_bd_cells /%s/fifo]" % (depth, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {%s}] "
                "[get_bd_cells /%s/fifo]" % (ram_style, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.TDATA_NUM_BYTES {%d}] "
                "[get_bd_cells /%s/fifo]" % (np.ceil(self.get_outstream_width() / 8), node_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/M_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, dout_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/S_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aresetn]" % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aclk]" % (node_name, clk_name, node_name)
            )
            return cmd
        else:
            raise Exception(
                "FIFO implementation style %s not supported, please use rtl or vivado" % impl_style
            )

    def bram_estimation(self):
        """Calculates resource estimation for BRAM"""
        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "block"):
            # Non-BRAM based implementation
            return 0

        if W == 1:
            return math.ceil(depth / 16384)
        elif W == 2:
            return math.ceil(depth / 8192)
        elif W <= 4:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 4))
        elif W <= 9:
            return (math.ceil(depth / 2048)) * (math.ceil(W / 9))
        elif W <= 18 or depth > 512:
            return (math.ceil(depth / 1024)) * (math.ceil(W / 18))
        else:
            return (math.ceil(depth / 512)) * (math.ceil(W / 36))

    def uram_estimation(self):
        """Calculates resource estimation for URAM"""

        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "ultra"):
            # Non-BRAM based implementation
            return 0
        else:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 72))

    def bram_efficiency_estimation(self):
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * depth
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        address_luts = 2 * math.ceil(math.log(depth, 2))

        if impl == "rtl" or (impl == "vivado" and ram_type == "distributed"):
            ram_luts = (math.ceil(depth / 32)) * (math.ceil(W / 2))
        else:
            ram_luts = 0

        return int(address_luts + ram_luts)

    def prepare_rtlsim(self):
        assert self.get_nodeattr("impl_style") != "vivado", (
            "StreamingFIFO impl_style "
            "cannot be vivado for rtlsim. Only impl_style=rtl supported."
        )
        super().prepare_rtlsim()

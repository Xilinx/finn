############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import os
import shutil
import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class ElementwiseBinary_rtl(ElementwiseBinaryOperation, RTLBackend):
    """Base CustomOp wrapper for the finn-rtllib eltwisef component."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(ElementwiseBinaryOperation.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update({
            "wmem": ("i", False, 0),
            "numInputVectors": ("ints", False, [1]),
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        })
        return my_attrs

    def generate_hdl(self, model, fpgapart, clk):
        rhs_style = self.get_nodeattr("rhs_style")
        if rhs_style == "const":
            self.set_nodeattr("mem_mode", "internal_decoupled")
            self.set_nodeattr("numInputVectors", self.calc_numInputVectors())

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)

        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwisef'
        template_path = f"{rtlsrc}/eltwisef_template.v"
        pe = self.get_nodeattr("PE")

        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": self._get_rtl_op_name(),
            "B_SCALE": 1.0,
            "FORCE_BEHAVIORAL": 0,
            "STREAM_BITS": pe * 32,
        }

        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            template = template.replace(f"${key_name}$", str(code_gen_dict[key_name]))

        with open(os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"), "w") as f:
            f.write(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        if rhs_style == "const":
            self.generate_hdl_memstream(fpgapart)

        for sv_file in ["eltwisef.sv", "binopf.sv", "queue.sv"]:
            shutil.copy(f"{rtlsrc}/{sv_file}", code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = f"{self.get_nodeattr('code_gen_dir_ipgen')}/"
            rtllib_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwisef'
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = self.get_nodeattr("gen_top_module")
        return [
            f"{rtllib_dir}/eltwisef.sv",
            f"{rtllib_dir}/binopf.sv",
            f"{rtllib_dir}/queue.sv",
            f"{code_gen_dir}{top_module}.v",
        ]

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        rhs_style = self.get_nodeattr("rhs_style")
        if rhs_style == "const":
            self.set_nodeattr("mem_mode", "internal_decoupled")

        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        if rhs_style == "const":
            node_name = self.onnx_node.name
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
            
            # instantiate the RTL block
            self.instantiate_ip(cmd)
            
            # connect elementwise core
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, din_name, node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, dout_name, node_name, node_name, dout_name)
            )

            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"
            
            strm_tmpl = None
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
            
            if strm_tmpl is None:
                raise Exception(f"No memstream wrapper found in {code_gen_dir}")
                
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name)
            )
            if runtime_writable:
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
                    % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        else:
            self.instantiate_ip(cmd)
        return cmd

    def instantiate_ip(self, cmd):
        node_name = self.onnx_node.name
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/eltwisef/")
        top_module = self.get_nodeattr("gen_top_module")
        source_target = "./ip/verilog/rtl_ops/%s" % node_name

        sourcefiles = [
            os.path.join(code_gen_dir, f"{top_module}.v"),
            rtllib_dir + "eltwisef.sv",
            rtllib_dir + "binopf.sv",
            rtllib_dir + "queue.sv",
        ]

        for f in sourcefiles:
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, f))
        cmd.append("create_bd_cell -type hier -reference %s %s" % (top_module, node_name))

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            lhs = context[node.input[0]]
            rhs = context[node.input[1]]

            assert list(lhs.shape) == self.get_normal_input_shape(ind=0), \
                f"Input shape mismatch for {node.input[0]}"
            assert list(rhs.shape) == self.get_normal_input_shape(ind=1), \
                f"Input shape mismatch for {node.input[1]}"

            out_shape = self.get_normal_output_shape(ind=0)
            if self.rhs_style == "const":
                rhs = np.broadcast_to(rhs, out_shape)
            if self.lhs_style == "const":
                lhs = np.broadcast_to(lhs, out_shape)

            lhs = lhs.reshape(self.get_folded_input_shape(ind=0))
            rhs = rhs.reshape(self.get_folded_input_shape(ind=0))

            lhs_filename = os.path.join(code_gen_dir, "input_0.npy")
            rhs_filename = os.path.join(code_gen_dir, "input_1.npy")
            np.save(lhs_filename, lhs)
            np.save(rhs_filename, rhs)

            io_dict = {"inputs": {}, "outputs": {"out0": []}}
            lhs_dtype = self.get_input_datatype(ind=0)
            lhs_width = self.get_instream_width(ind=0)
            rhs_dtype = self.get_input_datatype(ind=1)
            rhs_width = self.get_instream_width(ind=1)

            if self.lhs_style == "input" or self.lhs_style == "const":
                io_dict["inputs"]["in0"] = npy_to_rtlsim_input(lhs_filename, lhs_dtype, lhs_width)
            if self.rhs_style == "input" or self.rhs_style == "const":
                io_dict["inputs"]["in1"] = npy_to_rtlsim_input(rhs_filename, rhs_dtype, rhs_width)

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            out = io_dict["outputs"]["out0"]
            dtype = self.get_output_datatype(ind=0)
            width = self.get_outstream_width(ind=0)
            shape = self.get_folded_output_shape(ind=0)
            filename = os.path.join(code_gen_dir, "output_0.npy")
            rtlsim_output_to_npy(out, filename, dtype, shape, width, dtype.bitwidth())
            out = np.load(filename)
            context[node.output[0]] = out.reshape(self.get_normal_output_shape(ind=0)).astype(np.float32)
        else:
            ElementwiseBinaryOperation.execute_node(self, context, graph)

    def generate_params(self, model, code_gen_dir):
        weights = model.get_initializer(self.onnx_node.input[1])
        if weights is not None:
            self.make_weight_file(weights, "decoupled_npy", f"{code_gen_dir}/input_1.npy")
            self.make_weight_file(weights, "decoupled_verilog_dat", f"{code_gen_dir}/memblock.dat")
            self.set_nodeattr("wmem", self.calc_wmem())

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        from finn.util.data_packing import pack_innermost_dim_as_hex_string
        from finn.util.basic import roundup_to_integer_multiple

        folded_weight_shape = self.get_folded_input_shape(1)
        weight_tensor = weights.reshape(folded_weight_shape).copy()

        if weight_file_mode == "decoupled_verilog_dat":
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            weight_tensor = np.tile(weight_tensor, (num_w_reps,) + (1,) * (len(folded_weight_shape) - 1))

        export_wdt = self.get_input_datatype(1)
        weight_width = self.get_instream_width(1)
        weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

        if weight_file_mode == "decoupled_verilog_dat":
            shape = weight_tensor.shape
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, shape[-1]),
                export_wdt,
                weight_width_padded,
                prefix=""
            )
        else:
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, folded_weight_shape[-1]),
                export_wdt,
                weight_width_padded,
                prefix=""
            )

        weight_stream = weight_tensor_hex.flatten()
        with open(weight_file_name, "w") as f:
            for val in weight_stream:
                f.write(val + "\n")

    def calc_wmem(self):
        base_wmem = super().calc_wmem()
        num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
        return base_wmem * num_w_reps

    def calc_numInputVectors(self):
        if self.get_nodeattr("rhs_style") == "const":
            folded_lhs = self.get_folded_input_shape(0)
            if len(folded_lhs) >= 2:
                return list(folded_lhs[:-1])
        return [1]

    def minimize_weight_bit_width(self, model):
        super().minimize_weight_bit_width(model)
        self.set_nodeattr("mem_mode", "internal_decoupled")
        self.set_nodeattr("numInputVectors", self.calc_numInputVectors())

    def _get_rtl_op_name(self):
        """Override in subclasses to return the correct RTL operation name."""
        raise NotImplementedError("Subclasses must implement _get_rtl_op_name")


class ElementwiseAdd_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise addition for FLOAT32."""
    
    _operation = "Add", np.add, "({0} + {1})", '"ADD"'
    
    def _get_rtl_op_name(self):
        return '"ADD"'


class ElementwiseSub_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise subtraction for FLOAT32."""
    
    _operation = "Sub", np.subtract, "({0} - {1})", '"SUB"'
    
    def _get_rtl_op_name(self):
        return '"SUB"'


class ElementwiseMul_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise multiplication for FLOAT32."""
    
    _operation = "Mul", np.multiply, "({0} * {1})", '"MUL"'
    
    def _get_rtl_op_name(self):
        return '"MUL"'

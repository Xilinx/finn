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
        # Add memstream-related attributes
        my_attrs.update({
            # Weight memory depth for constants (calculated)
            "wmem": ("i", False, 0),
            # Number of input vectors to process (for constant repetition)
            "numInputVectors": ("ints", False, [1]),
            # Runtime writeable weights support (like MVAU)
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        })
        return my_attrs

    def generate_hdl(self, model, fpgapart, clk):
        # Force internal_decoupled mode for RTL elementwise operations at HDL generation time
        rhs_style = self.get_nodeattr("rhs_style")
        print(f"DEBUG: generate_hdl called for {self.onnx_node.name}, rhs_style={rhs_style}")
        
        if rhs_style == "const":
            print(f"DEBUG: Forcing mem_mode to internal_decoupled for constants in {self.onnx_node.name}")
            self.set_nodeattr("mem_mode", "internal_decoupled")
            # Calculate and set numInputVectors for memstream repetition
            num_input_vecs = self.calc_numInputVectors()
            print(f"DEBUG: Setting numInputVectors to {num_input_vecs} for {self.onnx_node.name}")
            self.set_nodeattr("numInputVectors", num_input_vecs)
        
        # Generate parameter files first
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)
        
        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwisef'
        template_path = f"{rtlsrc}/eltwisef_template.v"
        dt = DataType[self.get_nodeattr("out_dtype")]
        pe = self.get_nodeattr("PE")
        
        op_name = self._get_rtl_op_name()
        
        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": op_name,
            "B_SCALE": 1.0,
            "FORCE_BEHAVIORAL": 0,
            "STREAM_BITS": pe * 32,
        }
        
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = f"${key_name}$"
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"), "w") as f:
            f.write(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # Generate memstream for internal_decoupled mode
        mem_mode = self.get_nodeattr("mem_mode")
        print(f"DEBUG: generate_hdl - mem_mode={mem_mode}, rhs_style={rhs_style}")
        
        if mem_mode == "internal_decoupled" and rhs_style == "const":
            print(f"DEBUG: Calling generate_hdl_memstream for {self.onnx_node.name}")
            try:
                self.generate_hdl_memstream(fpgapart)
                print(f"DEBUG: generate_hdl_memstream completed successfully")
            except Exception as e:
                print(f"DEBUG: generate_hdl_memstream failed: {e}")
                raise
        else:
            print(f"DEBUG: Skipping generate_hdl_memstream - mem_mode={mem_mode}, rhs_style={rhs_style}")

        sv_files = ["eltwisef.sv", "binopf.sv", "queue.sv"]
        for sv_file in sv_files:
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
        print(f"DEBUG: code_generation_ipi called for {self.onnx_node.name}")
        
        # Force internal_decoupled for constants one more time to be sure
        rhs_style = self.get_nodeattr("rhs_style")
        if rhs_style == "const":
            print(f"DEBUG: Forcing mem_mode to internal_decoupled in code_generation_ipi for {self.onnx_node.name}")
            self.set_nodeattr("mem_mode", "internal_decoupled")
        
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        mem_mode = self.get_nodeattr("mem_mode")
        print(f"DEBUG: mem_mode = {mem_mode} for {self.onnx_node.name}")
        sname = "V"

        # check if memstream components are needed
        if mem_mode == "internal_decoupled":
            rhs_style = self.get_nodeattr("rhs_style")
            print(f"DEBUG: Taking internal_decoupled path for memstream generation, rhs_style={rhs_style}")
            
            # Only generate memstream if we actually have constants
            if rhs_style != "const":
                print(f"DEBUG: WARNING - mem_mode is internal_decoupled but rhs_style is {rhs_style}, not 'const'")
                print(f"DEBUG: Falling back to basic instantiation")
                self.instantiate_ip(cmd)
                return cmd
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            # clock and reset
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            # streams
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
            print(f"DEBUG: Looking for memstream wrapper in {code_gen_dir}")
            
            # memstream for constants
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            print(f"DEBUG: runtime_writeable_weights = {runtime_writable}")
            
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"
            
            # automatically find memstream verilog component in code generation directory
            strm_tmpl = None
            print(f"DEBUG: Files in {code_gen_dir}:")
            for fname in os.listdir(code_gen_dir):
                print(f"DEBUG:   - {fname}")
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
                    print(f"DEBUG: Found memstream template: {strm_tmpl}")
            
            if strm_tmpl is None:
                print(f"DEBUG: ERROR - No memstream wrapper found with suffix {file_suffix}")
                raise Exception(f"No memstream wrapper found in {code_gen_dir}")
                
            strm_tmpl_name = strm_tmpl[:-2]
            print(f"DEBUG: Using memstream template: {strm_tmpl_name}")
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
            strm_inst = node_name + "_wstrm"
            # instantiate the memstream cell
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )
            # connect memstream
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
            # Connect memstream output to elementwise second input (constants)
            connection_cmd = (
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_%s]"
                % (node_name, strm_inst, node_name, node_name, sname)
            )
            print(f"DEBUG: Adding memstream connection: {connection_cmd}")
            cmd.append(connection_cmd)
            # runtime writeable weights
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
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")

            # save bd
            cmd.append("save_bd_design")
        elif mem_mode == "internal_embedded" or mem_mode == "external":
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)
        else:
            raise Exception("Unrecognized mem_mode for ElementwiseBinary_rtl")
        return cmd

    def instantiate_ip(self, cmd):
        """Instantiate the RTL IP for the elementwise operation."""
        node_name = self.onnx_node.name
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/eltwisef/")
        top_module = self.get_nodeattr("gen_top_module")
        
        sourcefiles = [
            os.path.join(code_gen_dir, f"{top_module}.v"),
            rtllib_dir + "eltwisef.sv",
            rtllib_dir + "binopf.sv", 
            rtllib_dir + "queue.sv",
        ]
        
        source_target = "./ip/verilog/rtl_ops/%s" % node_name
        for f in sourcefiles:
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, f))
        
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (top_module, node_name, node_name)
            )
        else:
            cmd.append(
                "create_bd_cell -type hier -reference %s %s"
                % (top_module, node_name)
            )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            ElementwiseBinaryOperation.execute_node(self, context, graph)

    def generate_params(self, model, code_gen_dir):
        """Generate weight/constant parameter files for memstream."""
        mem_mode = self.get_nodeattr("mem_mode")
        print(f"DEBUG: generate_params called for {self.onnx_node.name}, mem_mode={mem_mode}")
        
        if mem_mode == "internal_decoupled":
            # Generate weight data file for constants (input 1)
            rhs_style = self.get_nodeattr("rhs_style")
            print(f"DEBUG: rhs_style={rhs_style}, input[1]={self.onnx_node.input[1]}")
            
            weights = model.get_initializer(self.onnx_node.input[1])
            print(f"DEBUG: Found weights: {weights is not None}")
            if weights is not None:
                print(f"DEBUG: Weights shape: {weights.shape}, generating parameter files")
                # Save for simulation
                self.make_weight_file(weights, "decoupled_npy", f"{code_gen_dir}/input_1.npy")
                # Save for RTL synthesis
                self.make_weight_file(weights, "decoupled_verilog_dat", f"{code_gen_dir}/memblock.dat")
                # Update wmem attribute
                wmem_val = self.calc_wmem()
                print(f"DEBUG: Setting wmem={wmem_val}")
                self.set_nodeattr("wmem", wmem_val)
            else:
                print(f"DEBUG: WARNING - No weights found for constants input {self.onnx_node.input[1]}")
        else:
            print(f"DEBUG: Skipping param generation for mem_mode={mem_mode}")

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given constants in appropriate format for memstream.
        For internal_decoupled mode, replicate constants based on numInputVectors."""
        from finn.util.data_packing import pack_innermost_dim_as_hex_string
        from finn.util.basic import roundup_to_integer_multiple
            
        folded_weight_shape = self.get_folded_input_shape(1)
        weight_tensor = weights.reshape(folded_weight_shape).copy()
        
        # For memstream replay: replicate constants for each input vector
        if weight_file_mode == "decoupled_verilog_dat":
            # Calculate how many times to replicate constants
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            # Replicate the weight tensor for each input vector
            replicated_weights = np.tile(weight_tensor, (num_w_reps,) + (1,) * (len(folded_weight_shape) - 1))
            weight_tensor = replicated_weights
            
        # Get data type and width information
        export_wdt = self.get_input_datatype(1)
        weight_width = self.get_instream_width(1)
        weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
        
        # Pack as hex strings
        if weight_file_mode == "decoupled_verilog_dat":
            # For replicated weights, reshape to account for replication
            total_shape = weight_tensor.shape
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, total_shape[-1]), 
                export_wdt, 
                weight_width_padded, 
                prefix=""
            )
        else:
            # Original logic for other modes
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, folded_weight_shape[-1]), 
                export_wdt, 
                weight_width_padded, 
                prefix=""
            )
            
        # Write to file
        weight_stream = weight_tensor_hex.flatten()
        with open(weight_file_name, "w") as f:
            for val in weight_stream:
                f.write(val + "\n")

    def calc_wmem(self):
        """Calculate WMEM accounting for constant repetition in internal_decoupled mode."""
        # Base memory depth for one set of constants
        base_wmem = super().calc_wmem()
        
        # For internal_decoupled mode, multiply by repetition count
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            return base_wmem * num_w_reps
        else:
            return base_wmem

    def calc_numInputVectors(self):
        """Calculate numInputVectors based on input shapes for constant repetition."""
        lhs_shape = self.get_nodeattr("lhs_shape") 
        rhs_shape = self.get_nodeattr("rhs_shape")
        
        # For elementwise operations, calculate how many times constants need repetition
        # This should be the number of transactions on the primary input stream
        if self.get_nodeattr("rhs_style") == "const":
            # Calculate based on folded input shape to match streaming pattern
            folded_lhs = self.get_folded_input_shape(0)
            
            # For elementwise binary RTL operations, the constants need to be repeated
            # for each set of input vectors that don't match the constant dimension
            # folded_lhs is typically [batch_size, num_vectors, vector_size]
            # For [1, 16, 48] with constant [48], we need [1, 16] repetitions
            if len(folded_lhs) >= 2:
                # All dimensions except the last (parallelized) dimension need repetition
                num_input_vecs = list(folded_lhs[:-1])
            else:
                num_input_vecs = [1]
        else:
            num_input_vecs = [1]
        
        return num_input_vecs

    def minimize_weight_bit_width(self, model):
        """Override to also set numInputVectors when styles are determined."""
        # Call parent implementation to set lhs_style/rhs_style
        super().minimize_weight_bit_width(model)
        
        lhs_style = self.get_nodeattr("lhs_style")
        rhs_style = self.get_nodeattr("rhs_style")
        print(f"DEBUG: After parent call for {self.onnx_node.name} - lhs_style: {lhs_style}, rhs_style: {rhs_style}")
        
        # Force internal_decoupled mode for RTL elementwise operations to ensure memstream
        print(f"DEBUG: Setting mem_mode to internal_decoupled for {self.onnx_node.name}")
        self.set_nodeattr("mem_mode", "internal_decoupled")
        
        # Calculate and set numInputVectors for memstream repetition
        num_input_vecs = self.calc_numInputVectors()
        print(f"DEBUG: Setting numInputVectors to {num_input_vecs} for {self.onnx_node.name}")
        self.set_nodeattr("numInputVectors", num_input_vecs)

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

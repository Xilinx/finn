# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
import shutil
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name, is_finn_op, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class FINNLoop(HWCustomOp, RTLBackend):
    """Class that corresponds to the meta/container node FINN loop
    which is a placeholder for a group of fpgadataflow nodes that have been separated
    out into a FINN-ONNX model of its own and are meant to be executed in a loop."""

    def get_nodeattr_types(self):
        my_attrs = {
            "body": ("g", True, ""),
            "iteration": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # FINN output datatype
            "outputDataType": ("s", True, ""),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_nodeattr(self, name):
        """Get a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types.
        Default value is returned if attribute is not set."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    ret = ModelWrapper(qonnx_make_model(ret))
                    return ret
                else:
                    return super().get_nodeattr(name)
            else:
                if req:
                    raise Exception(
                        """Required attribute %s unspecified in
                    a %s node"""
                        % (name, self.onnx_node.op_type)
                    )
                else:
                    # not set, return default value
                    return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    attr.g.CopyFrom(value)
                else:
                    super().set_nodeattr(name, value)
            else:
                super().set_nodeattr(name, value)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def get_normal_input_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            if is_finn_op(node.domain):
                inst = getCustomOp(node)
                ishape = inst.get_normal_input_shape(0)
            else:
                ishape = loop_body.get_tensor_shape(node.input[0])
        else:
            loop_body = self.get_nodeattr("body")
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if is_finn_op(param_node.domain):
                inst = getCustomOp(param_node)
                ishape = inst.get_normal_input_shape(1)
            else:
                ishape = loop_body.get_tensor_shape(tensor)
        return ishape

    def get_normal_output_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        if is_finn_op(node.domain):
            inst = getCustomOp(node)
            oshape = inst.get_normal_output_shape(0)
        else:
            oshape = loop_body.get_tensor_shape(node.output[0])
        return oshape

    def get_folded_input_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = getCustomOp(node)
            ishape = inst.get_folded_input_shape(0)
        else:
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            inst = getCustomOp(param_node)
            ishape = inst.get_folded_input_shape(1)
        return ishape

    def get_folded_output_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_folded_output_shape(0)

    def infer_node_datatype(self, model):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            idt = DataType[self.get_nodeattr("inputDataType")]
        else:
            loop_body = self.get_nodeattr("body")
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if is_finn_op(param_node.domain):
                inst = getCustomOp(param_node)
                idt = inst.get_input_datatype(1)
            else:
                idt = loop_body.get_tensor_datatype(tensor)
        return idt

    def get_output_datatype(self, ind=0):
        odt = DataType[self.get_nodeattr("outputDataType")]
        return odt

    def get_instream_width(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = getCustomOp(node)
            iwidth = inst.get_instream_width(0)
        else:
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            inst = getCustomOp(param_node)
            iwidth = inst.get_instream_width(1)
        return iwidth

    def get_outstream_width(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_outstream_width(0)

    def get_number_output_values(self):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output values
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_number_output_values()

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        loop_body = self.get_nodeattr("body")
        # for each iteration run execution
        iteration = self.get_nodeattr("iteration")
        for i_iter in range(iteration):
            # set the right parameters
            input_dict = {}
            for i, inp in enumerate(node.input):
                if i == 0:
                    input_dict[loop_body.graph.input[i].name] = inp_values
                else:
                    params = context[node.input[i]]
                    input_dict[loop_body.graph.input[i].name] = params[i_iter]
            outp_dict = oxe.execute_onnx(loop_body, input_dict)
            inp_values = outp_dict[loop_body.graph.output[0].name]
        result = outp_dict[loop_body.graph.output[0].name]
        context[node.output[0]] = np.asarray(result, dtype=np.float32)

    def generate_hdl(self, model, fpgapart, clk):
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)
        
        code_gen_dict = {}
        code_gen_dict["$LOOP_CONTROL_WRAPPER_NAME$"] = [f"{self.onnx_node.name}_loop_cont_wrapper"]
        code_gen_dict["$N_MAX_LAYERS$"] = str(self.get_nodeattr("iteration")),
        code_gen_dict["$ILEN_BITS$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$OLEN_BITS$"] = [str(self.get_output_datatype(0).bitwidth())]
        
        input_elements = np.prod(self.get_normal_input_shape(0))
        input_bytes    = (input_elements * self.get_input_datatype(0).bitwidth() + 8 - 1) // 8
        # round up to next power of 2
        input_bytes_rounded_to_power_of_2 = 2**(math.ceil(math.log2(input_bytes)))
        code_gen_dict["$LAYER_OFFS_INT$"] = [str(input_bytes_rounded_to_power_of_2)] # need to get correct value        
        
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mlo/loop_control_wrapper.v" 
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.onnx_node.name + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper)
            
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def generate_params(self, model, path):
        iteration = self.get_nodeattr("iteration")
        loop_node = self.onnx_node
        loop_body = self.get_nodeattr("body")
        for i, inp in enumerate(loop_node.input[1:]):
            params = model.get_initializer(inp)
            param_dtype = model.get_tensor_datatype(inp)
            assert params.shape[0] == iteration
            # get node that initializer is attached to
            loop_tensor = loop_body.graph.input[i + 1].name
            param_node = loop_body.find_consumer(loop_tensor)
            for iter in range(iteration):
                loop_body.set_initializer(loop_tensor, params[iter])
                loop_body.set_tensor_datatype(loop_tensor, param_dtype)
                inst = getCustomOp(param_node)
                inst.generate_params(loop_body, path)
                # if param_node.op_type.startswith("MVAU"):
                param_file = "{}/memblock.dat".format(path)
                new_param_file = "{}/memblock_{}.dat".format(path, iter)
                if param_node.op_type.startswith("MVAU"):
                    # rename so it doesn't get overwritten
                    shutil.move(param_file, new_param_file)
                else:
                    # get all generated Thresholding dat files
                    pe = inst.get_nodeattr("PE")
                    output_data_type = inst.get_nodeattr("outputDataType")
                    o_bitwidth = DataType[output_data_type].bitwidth()
                    param_files = []
                    for stage in range(o_bitwidth):
                        for pe_value in range(pe):
                            param_files.append(
                                path
                                + "/%s_threshs_%s_%s.dat"
                                % (
                                    param_node.name,
                                    pe_value,
                                    stage,
                                )
                            )
                    for param_file in param_files:
                        param_path = Path(param_file)
                        new_param_file = param_path.with_name(
                            param_path.stem + "_i" + str(iter) + param_path.suffix
                        )
                        shutil.move(param_path, new_param_file)
            if param_node.op_type.startswith("MVAU"):
                # concatinate all .dat files together
                param_file = "{}/memblock_{}.dat".format(path, param_node.name)
                with open(param_file, "w") as outfile:
                    for iter in range(iteration):
                        memblock_file = "{}/memblock_{}.dat".format(path, iter)
                        with open(memblock_file, "r") as infile:
                            for line in infile:
                                outfile.write(line)
                        os.remove(memblock_file)
            if param_node.op_type.startswith("Thresholding"):
                # concatinate all .dat files together
                pe = inst.get_nodeattr("PE")
                output_data_type = inst.get_nodeattr("outputDataType")
                o_bitwidth = DataType[output_data_type].bitwidth()
                for stage in range(o_bitwidth):
                    for pe_value in range(pe):
                        param_file = path + "/%s_threshs_%s_%s.dat" % (
                            param_node.name,
                            pe_value,
                            stage,
                        )
                        with open(param_file, "w") as outfile:
                            for iter in range(iteration):
                                iter_file = "{}/{}_threshs_{}_{}_i{}.dat".format(
                                    path, param_node.name, pe_value, stage, iter
                                )
                                with open(iter_file, "r") as infile:
                                    for line in infile:
                                        outfile.write(line)
                                os.remove(iter_file)

    def code_generation_ipi(self):
        
        # AXI regs
        cmd = [
            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_reg_32",
            "set_property CONFIG.TDATA_NUM_BYTES {4} [get_ips axis_reg_32]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_8",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_8]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_16",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_16]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_32",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_32]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_64",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_64]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_128",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_128]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_256",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_256]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_512",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_512]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_8",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_8]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_16",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_16]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_32",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_32]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_64",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_64]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_128",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_128]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_256",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_256]",

            "create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_512",
            "set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_512]",

            "create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_256",
            "set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {256} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_256]",

            "create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_512",
            "set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_512]",

            "create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axil_register_slice_64",
            "set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {64} CONFIG.HAS_PROT {0} CONFIG.DATA_WIDTH {64} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_W {1} CONFIG.REG_R {1} CONFIG.REG_B {1} ] [get_ips axil_register_slice_64]"
        ]
        
        source_files = [
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_macros.svh",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_intf.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/queue.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/cdma_top.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/cdma_u_wr.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_dma_wr_u.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/intermediate_frames.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/mux_in.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/mux_out.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/loop_control.sv",
            f"{self.get_nodeattr('code_gen_dir_ipgen')}/{self.onnx_node.name}_wrapper.v"
        ]
        for f in source_files:
            cmd += [f"add_files -norecurse {f}"]
        
        cmd.append(f"create_bd_cell -type module -reference {self.onnx_node.name}_loop_cont_wrapper {self.onnx_node.name}")
        
        # Connect the Loop Control Wrapper to the Loop Body
        loop_body = self.get_nodeattr("body")
        first_node_name = loop_body.find_consumer(loop_body.model.graph.input[0].name).name 
        last_node_name  = loop_body.find_producer(loop_body.model.graph.output[0].name).name 
        
        cmd.append(f"connect_bd_intf_net [get_bd_intf_pins {self.onnx_node.name}/m_axis_core_in] [get_bd_intf_pins {first_node_name}/in0_V]")
        cmd.append(f"connect_bd_intf_net [get_bd_intf_pins {last_node_name}/out0_V] [get_bd_intf_pins {self.onnx_node.name}/s_axis_core_out]")
        
        cmd.append(f"make_bd_intf_pins_external [get_bd_intf_pins {self.onnx_node.name}/m_axi_hbm]")
        cmd.append(f"make_bd_intf_pins_external [get_bd_intf_pins {self.onnx_node.name}/idx_fs]")
        cmd.append(f"make_bd_intf_pins_external [get_bd_intf_pins {self.onnx_node.name}/idx_se]")
        cmd.append(f"make_bd_pins_external [get_bd_pins {self.onnx_node.name}/n_layers]")
        cmd.append(f"make_bd_pins_external [get_bd_pins {self.onnx_node.name}/done_if]")
        
        return cmd

    def get_rtl_file_list(self, abspath=False):
        pass

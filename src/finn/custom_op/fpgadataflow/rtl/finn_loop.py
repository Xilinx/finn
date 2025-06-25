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
from qonnx.util.basic import (
    get_by_name,
    is_finn_op,
    qonnx_make_model,
    roundup_to_integer_multiple,
)

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.create import adjacency_list


def collect_ip_dirs(model, ipstitch_path):
    # collect list of all IP dirs
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        assert os.path.isdir(
            ip_dir_value
        ), """The directory that should
        contain the generated ip blocks doesn't exist."""
        ip_dirs += [ip_dir_value]
        if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
            if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
    return ip_dirs


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

    def get_exp_cycles(self):
        loop_body = self.get_nodeattr("body")
        check_if_cycles_annotated = False

        for node in loop_body.graph.node:
            cnode = getCustomOp(node)
            if cnode.get_nodeattr("cycles_estimate") is not None:
                check_if_cycles_annotated = True
                break
        if not check_if_cycles_annotated:
            loop_body = loop_body.transform(AnnotateCycles())

        return loop_body.analysis(dataflow_performance)["critical_path_cycles"] * self.get_nodeattr(
            "iteration"
        )

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
        self.generate_hdl_stream_tap()
        self.generate_params(model, code_gen_dir)
        code_gen_dict = {}
        code_gen_dict["$LOOP_CONTROL_WRAPPER_NAME$"] = [f"{self.onnx_node.name}_loop_cont_wrapper"]
        code_gen_dict["$N_MAX_LAYERS$"] = (str(self.get_nodeattr("iteration")),)
        code_gen_dict["$ILEN_BITS$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$OLEN_BITS$"] = [str(self.get_output_datatype(0).bitwidth())]

        input_elements = np.prod(self.get_normal_input_shape(0))
        input_bytes = (input_elements * self.get_input_datatype(0).bitwidth() + 8 - 1) // 8
        # round up to next power of 2
        input_bytes_rounded_to_power_of_2 = 2 ** (math.ceil(math.log2(input_bytes)))
        code_gen_dict["$LAYER_OFFS_INT$"] = [
            str(input_bytes_rounded_to_power_of_2)
        ]  # need to get correct value

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
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
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

    def generate_hdl_stream_tap(self):
        """Helper function to generate verilog code for stream tap components."""
        template_path = (
            os.environ["FINN_ROOT"] + "/finn-rtllib/stream_tap/hdl/stream_tap_wrapper_template.v"
        )
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        iteration = self.get_nodeattr("iteration")
        loop_body = self.get_nodeattr("body")
        graph_inputs = [x.name for x in loop_body.graph.input]
        # TODO check if this needs to be padded
        data_width = DataType.get_smallest_possible(iteration).bitwidth()
        # pad to nearest multiple of 8
        data_width = roundup_to_integer_multiple(data_width, 8)
        for node in loop_body.graph.node:
            node_inst = getCustomOp(node)
            if node_inst.get_nodeattr("mlo_max_iter"):
                stname = "IN_%s" % graph_inputs.index(node.input[1])
                code_gen_dict = {
                    "$MODULE_NAME$": [stname],
                    "$DATA_WIDTH$": [str(data_width)],
                    "$TAP_REP$": [str(iteration)],
                }
                # apply code generation to template
                with open(template_path, "r") as f:
                    template_wrapper = f.read()
                for key in code_gen_dict:
                    # transform list into long string separated by '\n'
                    code_gen_line = "\n".join(code_gen_dict[key])
                    template_wrapper = template_wrapper.replace(key, code_gen_line)
                with open(
                    os.path.join(code_gen_dir, stname + "_stream_tap_wrapper.v"),
                    "w",
                ) as f:
                    f.write(template_wrapper)

    def get_verilog_top_module_intf_names(self):
        # from wrapper template
        addr_bits = 64
        data_bits = 256

        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]

        intf_names["s_axis"] = []
        # AXI4S slave interface from outside loop to loop control externalize
        # to block diagram interface port and connect to fetch_start component
        intf_names["s_axis"].append(("in0_V", self.get_instream_width_padded(0)))
        # AXI4S slave interface for idx_fs
        # This interface should be externalized to an interface port on the block diagram
        # and connected to the fetch_start component
        intf_names["s_axis"].append(("idx_fs", str(data_bits)))

        intf_names["m_axis"] = []
        # AXI4S master interface to drive final loop output externalize
        # to block diagram interface port and connect to store_end component
        intf_names["m_axis"].append(("out0_V", self.get_outstream_width_padded(0)))
        # AXI4S master interface for idx_se
        # This interface should be externalized to an interface port on the block diagram
        # and connected to the store_end component
        intf_names["m_axis"].append(("idx_se", str(data_bits)))

        intf_names["aximm"] = []
        # AXI4 master interface for intermediate buffering between layers
        # TODO: rename because it might not be hbm?
        intf_names["aximm"].append(["m_axi_hbm", str(addr_bits)])
        intf_names["axilite"] = []

        # using ap_none field to add control signals
        intf_names["ap_none"] = []
        # n_layers and done_if should be externalize to a block diagram port
        # and connected to the axil_iw_slv_mlo component
        intf_names["ap_none"].append("n_layers")
        intf_names["ap_none"].append("done_if")

        loop_body = self.get_nodeattr("body")
        loop_body_intf = eval(loop_body.get_metadata_prop("vivado_stitch_ifnames"))
        for intf in loop_body_intf["aximm"]:
            intf_names["aximm"] += intf

        return intf_names

    def code_generation_ipi(self):
        # AXI regs
        cmd = [
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_reg_32""",
            "set_property CONFIG.TDATA_NUM_BYTES {4} [get_ips axis_reg_32]",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_8""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_8]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_16""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_16]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_32""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_32]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_64""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_64]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_128""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_128]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_256""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_256]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axis_register_slice_512""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_512]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_8""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} \
            CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_8]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_16""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} \
            CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_16]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_32""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} \
            CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_32]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_64""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} \
            CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_64]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_128""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} \
            CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_128]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_256""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} \
            CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_256]""",
            """create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 \
            -module_name axisf_register_slice_512""",
            """set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} \
            CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} \
            CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_512]""",
            """create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 \
            -module_name axi_register_slice_256""",
            """set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {256} \
            CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} \
            CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} \
            CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] \
             [get_ips axi_register_slice_256]""",
            """create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 \
            -module_name axi_register_slice_512""",
            """set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} \
            CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} \
            CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} \
            CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] \
             [get_ips axi_register_slice_512]""",
            """create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 \
            -module_name axil_register_slice_64""",
            """set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {64} \
            CONFIG.HAS_PROT {0} CONFIG.DATA_WIDTH {64} CONFIG.REG_AW {1} CONFIG.REG_AR {1} \
            CONFIG.REG_W {1} CONFIG.REG_R {1} CONFIG.REG_B {1} ] \
             [get_ips axil_register_slice_64]""",
            """create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 \
              -module_name cdma_datamover_rd""",
            "set_property -dict [list \
              CONFIG.c_addr_width {64} \
              CONFIG.c_enable_s2mm {0} \
              CONFIG.c_include_mm2s_dre {true} \
              CONFIG.c_m_axi_mm2s_data_width {256} \
              CONFIG.c_m_axis_mm2s_tdata_width {256} \
              CONFIG.c_mm2s_burst_size {64} \
             ] [get_ips cdma_datamover_rd]",
            """create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 \
             -module_name cdma_datamover_wr""",
            "set_property -dict [list \
             CONFIG.c_addr_width {64} \
             CONFIG.c_enable_mm2s {0} \
             CONFIG.c_include_s2mm_dre {true} \
             CONFIG.c_m_axi_s2mm_data_width {256} \
             CONFIG.c_s2mm_burst_size {64} \
             CONFIG.c_s_axis_s2mm_tdata_width {256} \
            ] [get_ips cdma_datamover_wr]",
        ]

        source_files = [
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_macros.svh",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_intf.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/queue.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_top.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/axi_dma_rd_a.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/axi_dma_rd_u.sv",
            # f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/axi_dma_wr_a.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/krnl_counter.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_a/cdma_a.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_a/cdma_a_rd.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_a/cdma_a_wr.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_u/cdma_u.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_u/cdma_u_wr.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_u/cdma_u_rd.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_x/cdma_x.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_x/cdma_x_rd.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/cdma/cdma_x/cdma_x_wr.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_adapter.v",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_dwc.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_fifo.v",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_fifo_adapter.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_reg_array_rtl.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_reg_array_tmplt.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_reg_rtl.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/axis_reg_tmplt.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/common/ram_p_c.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/axi_dma_wr_u.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/intermediate_frames.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/mux_in.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/infrastructure/mux_out.sv",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/mlo/loop_control.sv",
            f"{self.get_nodeattr('code_gen_dir_ipgen')}/{self.onnx_node.name}_wrapper.v",
            f"{os.environ['FINN_ROOT']}/finn-rtllib/fifo/hdl/Q_srl.v",
        ]
        for f in source_files:
            cmd += [f"add_files -norecurse {f}"]

        # create and instantiate FINNLoop node overarching block design
        cmd.append("create_bd_cell -type hier %s" % (self.onnx_node.name))
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        # clock and reset
        cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (self.onnx_node.name, clk_name))
        cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (self.onnx_node.name, rst_name))
        # interfaces
        node_intf = self.get_verilog_top_module_intf_names()
        m_axis_intfs = node_intf["m_axis"]
        s_axis_intfs = node_intf["s_axis"]
        control_intfs = node_intf["ap_none"]
        mm_intfs = node_intf["aximm"]
        for intf in m_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        for intf in s_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        for intf in mm_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        cnt_bits = 16
        for intf in control_intfs:
            if intf == "n_layers":
                cmd.append(
                    "create_bd_pin -from %d -to 0 -dir I -type data /%s/%s"
                    % (cnt_bits - 1, self.onnx_node.name, intf)
                )
            elif intf == "done_if":
                cmd.append(
                    "create_bd_pin -from 1 -to 0 -dir O -type data /%s/%s"
                    % (self.onnx_node.name, intf)
                )

        # instantiate loop shell
        loop_shell_name = f"{self.onnx_node.name}/{self.onnx_node.name}_loop_cont_wrapper"
        cmd.append(
            f"""create_bd_cell -type module -reference \
            {self.onnx_node.name}_loop_cont_wrapper {loop_shell_name}"""
        )
        # connect loop shell to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, loop_shell_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, loop_shell_name, clk_name)
        )
        # "externalize" some of the loop shell signals
        ext_intf_signals = ["in0_V", "out0_V", "m_axi_hbm", "idx_se", "idx_fs"]
        ext_signals = ["n_layers", "done_if"]
        for sig in ext_intf_signals:
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s]"
                % (self.onnx_node.name, sig, loop_shell_name, sig)
            )
        for sig in ext_signals:
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
                % (self.onnx_node.name, sig, loop_shell_name, sig)
            )

        # stream tap graph generation
        loop_body = self.get_nodeattr("body")
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd.append("file mkdir %s" % source_target)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # create a hierarchy for this layer, with the same port names
        stg_intf = {}
        stg_intf["clk"] = self.get_verilog_top_module_intf_names()["clk"]
        stg_intf["rst"] = self.get_verilog_top_module_intf_names()["rst"]
        bd_name = f"{self.onnx_node.name}/stream_tap_graph"
        cmd.append("create_bd_cell -type hier %s" % bd_name)
        # clock and reset
        cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (bd_name, clk_name))
        cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (bd_name, rst_name))
        # streams
        cmd.append(
            "create_bd_intf_pin -mode Master "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/m_axis_0" % bd_name
        )
        cmd.append(
            "create_bd_intf_pin -mode Slave "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/s_axis_0" % bd_name
        )
        for id, inp in enumerate(loop_body.graph.input[1:]):
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/m_axis_%d" % (bd_name, id + 1)
            )
        # get stream tap components
        stream_tap_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/stream_tap/hdl/")
        file_suffix = "_stream_tap_wrapper.v"
        # automatically find stream tap verilog components in code generation directory
        st_tmpl_names = []
        st_verilog_files = []
        for fname in os.listdir(code_gen_dir):
            if fname.endswith(file_suffix):
                st_verilog_files.append(os.path.join(code_gen_dir, fname))
                st_tmpl_names.append(fname[:-2])
        sourcefiles = st_verilog_files + [
            stream_tap_dir + "stream_tap.sv",
        ]
        for f in sourcefiles:
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]

        adj_list = adjacency_list(
            loop_body,
            lambda node: node.op_type == "Thresholding_rtl"
            or (
                node.op_type == "MVAU_rtl"
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            ),
        )
        # create map that maps each stream tap to its param node
        st_map = {}
        for id, inp in enumerate(loop_body.graph.input[1:]):
            consumer = loop_body.find_consumer(inp.name)
            st_map[consumer.name] = "IN_%d_stream_tap_wrapper" % (id + 1)

        producer = "__INPUT0__"
        for id, inp in enumerate(loop_body.graph.input[1:]):
            node_name = adj_list[producer][0]
            if node_name == "__OUTPUT0__":
                break
            inst_name = st_map[node_name]
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (st_map[node_name], bd_name, inst_name)
            )
            # connect
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (bd_name, clk_name, bd_name, inst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (bd_name, rst_name, bd_name, inst_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_%s] "
                "[get_bd_intf_pins %s/%s/m_axis_1]" % (bd_name, id + 1, bd_name, inst_name)
            )

            if "INPUT" not in producer:
                src_intf_name = "m_axis_0"
                dst_intf_name = "s_axis_0"
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (bd_name, st_map[producer], src_intf_name, bd_name, inst_name, dst_intf_name)
                )
            producer = node_name
            if id == 0:
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_0] "
                    "[get_bd_intf_pins %s/%s/s_axis_0]" % (bd_name, bd_name, inst_name)
                )
            elif id == len(loop_body.graph.input[1:]) - 1:
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
                    "[get_bd_intf_pins %s/%s/m_axis_0]" % (bd_name, bd_name, inst_name)
                )

        # connect stream tap graph to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, bd_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, bd_name, clk_name)
        )

        loop_body_ipstitch_path = loop_body.get_metadata_prop("vivado_stitch_proj")
        loop_body_vlnv = loop_body.get_metadata_prop("vivado_stitch_vlnv")
        loop_body_intf_names = eval(loop_body.get_metadata_prop("vivado_stitch_ifnames"))
        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
        cmd.append(
            "set_property ip_repo_paths "
            "[concat [get_property ip_repo_paths [current_project]] %s] "
            "[current_project]" % ip_dirs_str
        )
        cmd.append("update_ip_catalog -rebuild -scan_changes")
        finn_ip_name = f"{self.onnx_node.name}/finn_design_mlo"
        cmd.append("create_bd_cell -type ip -vlnv %s %s" % (loop_body_vlnv, finn_ip_name))
        # connect finn ip to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, finn_ip_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, finn_ip_name, clk_name)
        )
        # "externalize" some of the loop shell signals
        ext_signals = loop_body_intf_names["aximm"]
        for sig in ext_signals:
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s]"
                % (self.onnx_node.name, sig[0][0], finn_ip_name, sig[0][0])
            )
        # connect components with each other
        # stream tap with finn ip
        connect_signals = loop_body_intf_names["s_axis"]
        for id, sig in enumerate(connect_signals[:-1]):
            cmd.append(
                "connect_bd_intf_net "
                "[get_bd_intf_pins %s/m_axis_%d] [get_bd_intf_pins %s/s_axis_%d]"
                % (bd_name, id + 1, finn_ip_name, id + 1)
            )
        # connect stream tap with loop wrapper
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/s_axis_0] [get_bd_intf_pins %s/m_axis_core_in_fw_idx]"
            % (bd_name, loop_shell_name)
        )
        # connect loop wrapper with finn ip
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/m_axis_core_in] [get_bd_intf_pins %s/s_axis_0]"
            % (loop_shell_name, finn_ip_name)
        )
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/m_axis_0] [get_bd_intf_pins %s/s_axis_core_out]"
            % (finn_ip_name, loop_shell_name)
        )

        return cmd

    def get_rtl_file_list(self, abspath=False):
        pass

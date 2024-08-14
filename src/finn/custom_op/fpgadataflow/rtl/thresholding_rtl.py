# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import shutil
from pyverilator.util.axi_utils import reset_rtlsim, rtlsim_multi_io
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.basic import (
    get_memutil_alternatives,
    get_rtlsim_trace_depth,
    make_build_dir,
    mem_primitives_versal,
    pyverilate_get_liveness_threshold_cycles,
)
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class Thresholding_rtl(Thresholding, RTLBackend):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # memory depth triggers for threshold storage
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            # enable uniform thres optimization
            # doesn't actually do anything yet, only
            # for resource estimations
            "uniform_thres": ("i", False, 0, {0, 1}),
            # enable deep pipelining for easier timing closure
            # setting to 0 may save some FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_pe_mem_geometries(self):
        """return a list of (bitwidth, depth) for PE memory configurations to be used
        in resource estimation

        for each bitwidth, the depth is calculated as the
        number of thresholds that can be stored in a single
        memory block
        the bitwidth is the bitwidth of the threshold values
        the depth is the number of thresholds that can be stored
        in a single memory block
        the number of memory blocks is calculated as the number
        of thresholds divided by the depth
        the number of memory blocks is then multiplied by the
        number of PEs to get the total number of memory blocks
        required for the entire layer
        """
        pe = self.get_nodeattr("PE")
        wdt = self.get_weight_datatype()
        wdt_bits = wdt.bitwidth()
        odt = self.get_output_datatype()
        odt_bits = odt.bitwidth()
        t_channels = self.get_nodeattr("NumChannels")
        cf = t_channels / pe
        is_uniform = self.get_nodeattr("uniform_thres")
        if is_uniform:
            ret = [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        else:
            ret = [(wdt_bits, (cf) * 2**x) for x in range(odt_bits)]
        return ret

    def get_memory_estimate(self):
        """return the memory estimate for this node"""
        res_dict = {}
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        pe = self.get_nodeattr("PE")
        ret = self.get_pe_mem_geometries()
        for mem_cfg in ret:
            (width, depth) = mem_cfg
            primitives = mem_primitives_versal
            if depth_trigger_bram != 0 or depth_trigger_uram != 0:
                if depth >= depth_trigger_bram and depth < depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "BRAM" in k}
                elif depth >= depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "URAM" in k}
            alts = get_memutil_alternatives(mem_cfg, primitives)
            primary_alt = alts[0]
            res_type = primary_alt[0].split("_")[0]
            res_count, eff, waste = primary_alt[1]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count
        return res_dict

    def bram_estimation(self):
        """return the number of BRAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("BRAM", 0)

    def uram_estimation(self):
        """return the number of URAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("URAM", 0)

    def lut_estimation(self):
        """return the number of LUTs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("LUTRAM", 0)

    def get_all_meminit_filenames(self, abspath=False):
        "Return a list of all .dat memory initializer files used for this node"
        dat_files = []
        t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                dat_files.append(thresh_file)
        return dat_files

    def prepare_codegen_rtl_values(self, model):
        """All dictionary values produced in this function are to replace
        their key value(s) in the RTL template files"""
        code_gen_dict = {}

        thresholds = model.get_initializer(self.onnx_node.input[1])
        bias = self.get_nodeattr("ActVal")  # activation bias value
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        input_data_type = self.get_nodeattr("inputDataType")  # input/threshold precision
        o_bitwidth = DataType[output_data_type].bitwidth()

        t_path = self.get_nodeattr("code_gen_dir_ipgen")
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            thresh_file_name = f"{t_path}/memblock.dat"
            self.make_weight_file(thresholds, "decoupled", thresh_file_name)

        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type) and decrease the bias by 1.
        # Additionally, increase number of threshold steps to reflect new shape
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("numSteps")
        wdt = self.get_weight_datatype()
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
                bias = bias - 1
            # TODO: temporary fix for unsigned narrow quantization
            else:
                max_val = wdt.max()
                if max_val > DataType[input_data_type].max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
            n_thres_steps += 1

        # add dummy dimension as final dimension (that's what gets packed with next call)
        t_expand = np.expand_dims(thresholds, axis=-1)
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
        t_packed = pack_innermost_dim_as_hex_string(
            t_expand,
            wdt,
            bw_hexdigit,
            prefix="",
        )

        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")  # number of channels

        # If a single threshold value is found, broadcast the value
        if t_packed.shape[0] == 1:
            t_packed = np.broadcast_to(t_packed, (pe, expected_thresholds))
            num_channels = pe

        channel_fold = int(num_channels / pe)

        for stage in range(o_bitwidth):
            sn = o_bitwidth - stage - 1
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                threshs = np.zeros([channel_fold * (2**stage)], dtype="object")
                for ch in range(channel_fold):
                    for i in range(2**stage):
                        threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                            (i << (o_bitwidth - stage)) + 2**sn - 1
                        ]
                with open(thresh_file, "w") as f:
                    for val in threshs:
                        f.write(val + "\n")
        code_gen_dict["$THRESHOLDS_PATH$"] = ['"./%s_"' % self.onnx_node.name]

        # Identify the module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [
            self.get_verilog_top_module_name() + "_axi_wrapper"
        ]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        i_bitwidth = DataType[input_data_type].bitwidth()

        code_gen_dict["$N$"] = [str(o_bitwidth)]  # output precision - convert bitwidth to string
        code_gen_dict["$WT$"] = [
            str(wdt.bitwidth())
        ]  # threshold precision - convert bitwidth to string
        code_gen_dict["$WI$"] = [str(i_bitwidth)]  # input precision - convert bitwidth to string
        code_gen_dict["$C$"] = [str(num_channels)]  # number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # requires C = M*PE

        # Is the input datatype signed or unsigned?
        # The thresholding core needs to know this when comparing weights to inputs
        if self.get_input_datatype().signed():
            code_gen_dict["$SIGNED$"] = [str(1)]
        else:
            code_gen_dict["$SIGNED$"] = [str(0)]

        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )
        code_gen_dict["$O_BITS$"] = [str(int(o_bits))]

        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        code_gen_dict["$USE_AXILITE$"] = [str(rt_weights)]

        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        deep_pipeline = self.get_nodeattr("deep_pipeline")
        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(deep_pipeline)]
        return code_gen_dict

    def get_rtl_file_list(self):
        """Thresholding binary search RTL file list"""
        return [
            "axilite_if.v",
            "thresholding.sv",
            "thresholding_axi.sv",
            "thresholding_template_wrapper.v",
        ]

    def get_rtl_file_paths(self):
        """Get full path of all RTL files"""
        rtl_root_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl/"
        rtl_file_list = self.get_rtl_file_list()
        rtl_file_paths = [rtl_root_dir + file for file in rtl_file_list]
        return rtl_file_paths

    def get_rtl_template_data(self, path):
        """Return RTL file contents as a template"""
        with open(path, "r") as f:
            template = f.read()
        return template

    def fill_in_rtl_template_data(self, replace_dict, template_data):
        """Use attribute values to finn in RTL template placeholders"""
        template_data_cp = template_data
        for key in replace_dict:
            replacement_line = "\n".join(replace_dict[key])
            template_data_cp = template_data_cp.replace(key, replacement_line)
        return template_data_cp

    def dump_rtl_data(self, dest_dir, filename, data):
        """Dump filled-in-template RTL files for future synthesis step"""
        # when generating template files, handle a special case:
        # if the filename contains the word "template", replace that
        # with the node name to distinguish between instances
        if "template" in filename:
            filename = self.get_nodeattr("gen_top_module") + ".v"
        with open(os.path.join(dest_dir, filename), "w") as f:
            f.write(data)
        return

    def generate_hdl(self, model, fpgapart, clk):
        """Prepare HDL files from templates for synthesis"""
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values(model)

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Set the 'gen_top_module' attribute for use later
        # by PyVerilator and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])

        for rtl_file_path in self.get_rtl_file_paths():
            # read in original RTL template file
            template_data = self.get_rtl_template_data(rtl_file_path)
            # apply code generation to templates
            data = self.fill_in_rtl_template_data(code_gen_dict, template_data)
            # dump filled-in template to destination directory for compilation
            file_only_path = rtl_file_path.split("/")[-1]
            self.dump_rtl_data(code_gen_dir, file_only_path, data)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        return

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [
            x.replace("thresholding_template_wrapper", self.get_nodeattr("gen_top_module"))
            for x in self.get_rtl_file_list()
        ]
        dat_files = self.get_all_meminit_filenames(abspath=True)
        single_src_dir = make_build_dir("pyverilator_" + self.onnx_node.name + "_")
        for dat_file in dat_files:
            shutil.copy(dat_file, single_src_dir)

        # build the Verilator emulation library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=single_src_dir,
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_nodeattr("gen_top_module"),
            auto_eval=False,
        )

        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if mode == "cppsim":
            Thresholding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the thresholds
                if in_ind == 0:
                    assert (
                        str(context[inputs].dtype) == "float32"
                    ), """Input datatype is
                    not float32 as expected."""
                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)

                    if self.get_input_datatype() == DataType["BIPOLAR"]:
                        # store bipolar activations as binary
                        reshaped_input = (reshaped_input + 1) / 2
                        export_idt = DataType["BINARY"]
                    else:
                        export_idt = self.get_input_datatype()

                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                elif in_ind > 2:
                    raise Exception("Unexpected input found for Thresholding_rtl")
                in_ind += 1

            # Create a PyVerilator wrapper of the RTLSim .so
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            io_names = self.get_verilog_top_module_intf_names()
            istream_name = io_names["s_axis"][0][0]
            ostream_name = io_names["m_axis"][0][0]
            io_dict = {
                "inputs": {istream_name: inp},
                "outputs": {ostream_name: []},
            }

            trace_file = self.get_nodeattr("rtlsim_trace")
            if trace_file == "default":
                trace_file = self.onnx_node.name + ".vcd"
            sname = "_"

            # Change into so directory to ensure threshold files can be found
            rtlsim_so = self.get_nodeattr("rtlsim_so")
            so_dir = os.path.dirname(os.path.realpath(rtlsim_so))
            olcwd = os.getcwd()
            os.chdir(so_dir)
            num_out_values = self.get_number_output_values()
            reset_rtlsim(sim)
            total_cycle_count = rtlsim_multi_io(
                sim,
                io_dict,
                num_out_values,
                trace_file=trace_file,
                sname=sname,
                liveness_threshold=pyverilate_get_liveness_threshold_cycles(),
            )
            self.set_nodeattr("cycles_rtlsim", total_cycle_count)
            os.chdir(olcwd)
            output = io_dict["outputs"][ostream_name]

            # Manage output data
            odt = self.get_output_datatype()
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

    def code_generation_ipi(self):
        """Constructs and returns the TCL commands for node instantiation as an RTL
        block."""
        rtl_file_list = [
            x.replace("thresholding_template_wrapper", self.get_nodeattr("gen_top_module"))
            for x in self.get_rtl_file_list()
        ]
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        for rtl_file in rtl_file_list:
            cmd.append(
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.path.join(code_gen_dir, rtl_file))
            )

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )

        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            intf_names["axilite"] = ["s_axilite"]

        return intf_names

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_name : filename for the weight file to be generated

        """
        thresholds = weights
        pe = self.get_nodeattr("PE")
        ch = self.get_nodeattr("NumChannels")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type) and decrease the bias by 1.
        # Additionally, increase number of threshold steps to reflect new shape
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("numSteps")
        wdt = self.get_weight_datatype()
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
            # TODO: temporary fix for unsigned narrow quantization
            else:
                max_val = wdt.max()
                if max_val > self.get_input_datatype().max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
            n_thres_steps += 1

        # If a single threshold value is found, broadcast the value
        if thresholds.shape[0] == 1:
            thresholds = np.broadcast_to(thresholds, (pe, expected_thresholds))
            ch = pe

        width_padded = roundup_to_integer_multiple(thresholds.shape[1], 2**o_bitwidth)
        thresh_padded = np.zeros((thresholds.shape[0], width_padded))
        thresh_padded[: thresholds.shape[0], :n_thres_steps] = thresholds
        thresh_stream = []
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
        padding = np.zeros(width_padded, dtype=np.int32)

        chan_ind = 0
        cf = ch // pe
        for fold in range(cf):
            for c in range(2 ** (pe - 1).bit_length()):
                if (c == 0 or c % pe != 0) and c < pe:
                    for t in thresh_padded[chan_ind]:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [t], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
                    chan_ind += 1
                else:
                    for z in padding:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [z], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
        with open(weight_file_name, "w") as f:
            for val in thresh_stream:
                f.write(val + "\n")

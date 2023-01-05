# Copyright (C) 2022, Advanced Micro Devices, Inc.
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
import textwrap
import warnings
from math import ceil, log2
from qonnx.core.datatype import DataType
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir, find_next_power_of_2
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

"""@package thresholding_binary_search
- ONNX i/o tensor shape assumptions for Thresholding:
- input 0 is the input tensor, shape (..., NumChannels)
- input 1 is the threshold tensor, shape (NumChannels, n_thres)
- output 0 is the output tensor, shape (..., NumChannels) - same as input
- the '...' here can be any shape (representing groups of vectors)

This module creates an RTL IP, HLS is not supported. See 'thresholding_batch'
for a HLS equivalent.
"""


class Thresholding_Binary_Search(HLSCustomOp):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # parallelization; channels thresholded per cycle
            "PE": ("i", True, 0),
            # number of channels (each may have different thresholds)
            "NumChannels": ("i", True, 0),
            # number of steps in thresholding function. Used only in decoupled mode
            "numSteps": ("i", True, 1),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # input and output FIFO depths
            "inFIFODepth": ("i", False, 0),
            "outFIFODepth": ("i", False, 0),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # memory mode for the thresholds
            # const -- embedded thresholds, default
            # decoupled -- streaming thresholds with streamer packaged inside IP
            "mem_mode": ("s", False, "const", {"const", "decoupled"}),
            # (mem_mode = decoupled only) whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "gen_top_module": ("s", False, ""),
            "activation_bias": ("i", False, 0),
            "clkFreq": ("i", False, 200000000),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_tmem(self):
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe

    def make_shape_compatible_op(self, model):
        return []

    def infer_node_datatype(self, model):
        return

    def verify_node(self):
        return []

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return 0

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("outputDataType")]

    def get_weight_datatype(self):
        """The term 'weights' and 'thresholds' are used interchangably in this class."""
        return DataType[self.get_nodeattr("weightDataType")]

    def minimize_accumulator_width(self, model):
        return None

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("PE")

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_weightstream_width(self):
        # Only 'decoupled' mode is supported
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode != "decoupled":
            raise Exception(
                "Unrecognized memory mode for this node: {}".format(mem_mode)
            )
        pe = self.get_nodeattr("PE")
        wp = self.get_weight_datatype().bitwidth()
        n_thres_steps = self.get_nodeattr("numSteps")
        w_width = pe * wp * n_thres_steps
        return w_width

    def get_folded_input_shape(self, ind=0):
        fold = self.calc_tmem()
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        num_channels = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [num_channels])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        return 0

    def get_exp_cycles(self):
        return 0

    def get_template_param_values(self):
        return dict()

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr(
            "numSteps"
        ), "Mismatch in threshold steps"
        if not self.get_input_datatype().signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert np.equal(
            np.mod(orig_thres_matrix, 1), 0
        ).all(), "Need int threshold tensor"
        ret = orig_thres_matrix
        # workaround for vivado_hls threshold bug
        if ret[0][0] == 0 and n_thres_steps == 1:
            ret = np.copy(ret)
            ret[0][0] = 1
            warnings.warn(
                "Setting 0-valued first threshold to 1 to avoid vivado_hls bug"
            )
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert (
            ret.shape[0] == mh
        ), "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:
        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated
        """
        # There are 'decoupled_*' flavors, just make sure that the flavors
        # are decoupled related
        if "decoupled" not in weight_file_mode:
            raise Exception(
                "Unrecognized memory mode for this node: {}".format(weight_file_mode)
            )

        threshold_tensor = self.get_hls_compatible_threshold_tensor(weights)
        tdt = self.get_weight_datatype()
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)

        # streaming thresholds need to be organized differently
        # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
        decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
        # (1, tmem, pe, n_thres_steps) -(1, tmem, pe * n_thres_steps)
        pe = self.get_nodeattr("PE")
        n_thres_steps = self.get_nodeattr("numSteps")
        decoupled_thres_pe_flipped = np.flip(decoupled_thres, axis=-2)
        decoupled_thres = decoupled_thres.reshape(1, -1, pe * n_thres_steps)
        decoupled_thres = decoupled_thres.copy()
        decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.reshape(
            1, -1, pe * n_thres_steps
        )
        decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.copy()

        if weight_file_mode == "decoupled_npy":
            # save weight stream into npy for cppsim
            np.save(weight_file_name, decoupled_thres)
        elif weight_file_mode == "decoupled_verilog_dat":
            # convert weight values into hexstring
            weight_width = self.get_weightstream_width()
            # pad to nearest 4 bits to get hex strings
            weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
            weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
            )
            weight_stream = weight_tensor_pe_flipped.flatten()
            weight_stream = weight_stream.copy()
            with open(weight_file_name, "w") as f:
                for val in weight_stream:
                    f.write(val + "\n")
        elif weight_file_mode == "decoupled_runtime":
            # memstream axi-lite interface will map each mem line to
            # one or multiple 32-bit words
            weight_width = self.get_weightstream_width()
            words_per_memwidth = 2 ** ceil(log2(weight_width / 32))
            if words_per_memwidth < 1:
                words_per_memwidth = 1
            weight_width_padded = words_per_memwidth * 32
            # first, pack and ensure padding to 32 bits
            weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
            )
            weight_stream = weight_tensor_pe_flipped.flatten()
            weight_stream = weight_stream.copy()
            with open(weight_file_name, "w") as f:
                for val in weight_stream:
                    # split into groups of 8 hex digits (= 32 bits)
                    words_32b = textwrap.wrap(val, 8)
                    words_32b.reverse()
                    for word_32b in words_32b:
                        f.write(word_32b + "\n")
        else:
            raise Exception("Decoupled weight export not yet implemented")

    # Get the integer from the DataType and string-ify it
    # This assumes that the data is in the form "INTx" or similar
    def conv_datatype_to_str(self, data_type):
        # Handle the case that an int is passed to the function
        if isinstance(data_type, int):
            return str(data_type)
        return str(DataType[data_type].bitwidth())

    def prepare_codegen_rtl_values(self):
        """All dictionary values produced in this function are to replace
        their key value(s) in the RTL template files"""
        code_gen_dict = {}

        # Identify the module names
        code_gen_dict["$MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        code_gen_dict["$MODULE_NAME_AXI$"] = [
            self.get_verilog_top_module_name() + "_axi"
        ]
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [
            self.get_verilog_top_module_name() + "_axi_wrapper"
        ]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        input_data_type = self.get_nodeattr(
            "inputDataType"
        )  # input/threshold precision
        num_channels = self.get_nodeattr("NumChannels")  # number of channels
        bias = self.get_nodeattr("activation_bias")  # activation bias value

        code_gen_dict["$N$"] = [
            self.conv_datatype_to_str(output_data_type)
        ]  # output precision
        code_gen_dict["$M$"] = [
            self.conv_datatype_to_str(input_data_type)
        ]  # input/threshold precision
        code_gen_dict["$C$"] = [
            self.conv_datatype_to_str(num_channels)
        ]  # number of channels
        code_gen_dict["$BIAS$"] = [
            self.conv_datatype_to_str(bias)
        ]  # activation bias value

        # Is the input datatype signed or unsigned?
        # The thresholding core needs to know this when comparing weights to inputs
        if self.get_input_datatype().min() < 0:
            code_gen_dict["$SIGN$"] = ["signed"]
        else:
            code_gen_dict["$SIGN$"] = ["unsigned"]

        return code_gen_dict

    def get_rtl_file_list(self):
        return ["thresholding.sv", "thresholding_axi.sv", "thresholding_axi_wrapper.v"]

    def get_rtl_file_paths(self):
        rtl_root_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl/"
        rtl_file_list = self.get_rtl_file_list()
        rtl_file_paths = [rtl_root_dir + file for file in rtl_file_list]
        return rtl_file_paths

    def get_rtl_template_data(self, path):
        with open(path, "r") as f:
            template = f.read()
        return template

    def fill_in_rtl_template_data(self, replace_dict, template_data):
        template_data_cp = template_data
        for key in replace_dict:
            replacement_line = "\n".join(replace_dict[key])
            template_data_cp = template_data_cp.replace(key, replacement_line)
        return template_data_cp

    def dump_rtl_data(self, dest_dir, filename, data):
        with open(os.path.join(dest_dir, filename), "w") as f:
            f.write(data)
        return

    def generate_hdl(self):
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values()

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        for rtl_file_path in self.get_rtl_file_paths():
            # read in original RTL template file
            template_data = self.get_rtl_template_data(rtl_file_path)
            # apply code generation to templates
            data = self.fill_in_rtl_template_data(code_gen_dict, template_data)
            # dump filled-in template to destination directory for compilation
            file_only_path = rtl_file_path.split("/")[-1]
            self.dump_rtl_data(code_gen_dir, file_only_path, data)

        # Before we return - set the 'gen_top_module' attribute for use later
        # by PyVerilator and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])
        return

    def code_generation_ipgen(self, model, fpgapart, clk):
        self.generate_hdl()

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

        # Generate params for RTLSim
        self.generate_params(model, code_gen_dir)

    def generate_params(self, model, path):
        # Only 'decoupled' mode is supported
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode != "decoupled":
            raise Exception(
                "Unrecognized memory mode for this node: {}".format(mem_mode)
            )

        code_gen_dir = path
        weight_filename_sim = "{}/thresholds.npy".format(code_gen_dir)
        thresholds = model.get_initializer(self.onnx_node.input[1])
        self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)

        # Verilog.dat thresholds:
        # also save weights as Verilog .dat file
        # note that we provide two different .dat files, one for synth
        # and one for synthesis. this is because URAM-based weights always
        # need zero weights for synthesis, otherwise they get inferred
        # as BRAM
        weight_filename_rtl_synth = "{}/memblock_synth_0.dat".format(code_gen_dir)
        weight_filename_rtl_sim = "{}/memblock_sim_0.dat".format(code_gen_dir)
        # sim weights are always the true weights
        self.make_weight_file(
            thresholds, "decoupled_verilog_dat", weight_filename_rtl_sim
        )

        # Synthesis thresholds:
        synth_thresholds = thresholds
        self.make_weight_file(
            synth_thresholds, "decoupled_verilog_dat", weight_filename_rtl_synth
        )

        return

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = self.get_rtl_file_list()

        # build the Verilator emulation library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_nodeattr("gen_top_module"),
        )

        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def execute_node(self, context, graph):
        # Perform input checks
        if self.get_nodeattr("exec_mode") != "rtlsim":
            raise Exception(
                "Invalid exec_mode value: {}; exec_mode must be set to '{}'".format(
                    self.get_nodeattr("exec_mode"), "rtlsim"
                )
            )
        if self.get_nodeattr("mem_mode") != "decoupled":
            raise Exception(
                "Invalid mem_mode value: {}; mem_mode must be set to '{}'".format(
                    self.get_nodeattr("mem_mode"), "decoupled"
                )
            )

        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
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
                raise Exception("Unexpected input found for Thresholding_Batch")
            in_ind += 1

        # Create a PyVerilator wrapper of the RTLSim .so
        sim = self.get_rtlsim()
        nbits = self.get_instream_width()
        inp = npy_to_rtlsim_input(
            "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
        )

        super().reset_rtlsim(sim)
        super().toggle_clk(sim)

        wnbits = self.get_weightstream_width()
        export_wdt = self.get_weight_datatype()
        wei = npy_to_rtlsim_input(
            "{}/thresholds.npy".format(code_gen_dir), export_wdt, wnbits
        )
        num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
        io_dict = {
            "inputs": {"in0": inp, "weights": wei * num_w_reps},
            "outputs": {"s_axis": []},
        }
        self.rtlsim_multi_io(sim, io_dict)
        output = io_dict["outputs"]["out"]

        # Manage output data
        odt = self.get_output_datatype()
        target_bits = odt.bitwidth()
        packed_bits = self.get_outstream_width()
        out_npy_path = "{}/output.npy".format(code_gen_dir)
        out_shape = self.get_folded_output_shape()

        rtlsim_output_to_npy(
            output, out_npy_path, odt, out_shape, packed_bits, target_bits
        )

        # load and reshape output
        output = np.load(out_npy_path)
        oshape = self.get_normal_output_shape()
        output = np.asarray([output], dtype=np.float32).reshape(*oshape)
        context[node.output[0]] = output
        return

    def code_generation_ipi(self):
        """Constructs and returns the TCL commands for node instantiation as an RTL
        block."""
        cmd = []
        rtl_file_list = self.get_rtl_file_list()
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        node_name = self.onnx_node.name
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
        clock_freq = self.get_nodeattr("clkFreq")

        for rtl_file in rtl_file_list:
            cmd.append(
                "add_files -norecurse %s" % (os.path.join(code_gen_dir, rtl_file))
            )

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )

        cmd.append(
            "set_property -dict [list CONFIG.FREQ_HZ {%d}] [%s %s/%s]"
            % (clock_freq, "get_bd_intf_pins", node_name, din_name)
        )

        cmd.append(
            "set_property -dict [list CONFIG.FREQ_HZ {%d}] [%s %s/%s]"
            % (clock_freq, "get_bd_intf_pins", node_name, dout_name)
        )

        return cmd

    def get_verilog_top_module_intf_names(self):
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""

        intf_names = super().get_verilog_top_module_intf_names()
        # Only 'decoupled' mode is supported - check before adding axilite interface
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode != "decoupled":
            raise Exception(
                "Unrecognized memory mode for this node: {}".format(mem_mode)
            )
        intf_names["axilite"] = ["s_axilite"]
        intf_names["s_axis"] = [["s_axis"]]
        intf_names["m_axis"] = [["m_axis"]]

        return intf_names

    def get_dynamic_config(self, model, address_stride=1):
        """Returns a configuration dictionary containing axilite write commands
        in order to program the thresholds into the RTL core during runtime.
        The default address stride for the weights is 1 byte."""

        thresholds = model.get_initializer(self.onnx_node.input[1])
        num_channels, num_weights_per_channel = thresholds.shape

        weight_addr_boundary = find_next_power_of_2(num_weights_per_channel)
        # Make sure that the next power of 2 (output) is greater than the input
        assert weight_addr_boundary >= num_weights_per_channel

        config = {}
        channel_cntr = 0
        for channel in thresholds:
            channel_start_addr = channel_cntr * weight_addr_boundary * address_stride
            weight_cntr = 0
            addr = 0
            for weight in channel:
                key_name = "{}_{}{}_{}{}".format(
                    "axilite", "ch", str(channel_cntr), "w", str(weight_cntr)
                )
                config[key_name] = (
                    channel_start_addr + addr,
                    int(str(pack_innermost_dim_as_hex_string([weight], self.get_weight_datatype(), self.get_weight_datatype().bitwidth())), 0),
                )

                weight_cntr += 1
                addr += address_stride

            channel_cntr += 1

        return config

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        """This is needed for the HLSSynthIP() transformation.
        This is an IP, not a HLS node, so therefore provide an empty hook
        to prevent any HLS synthesis."""
        pass

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

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
import shutil
import textwrap
import warnings
from qonnx.core.datatype import DataType
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import (
    layout_nhwc_to_hw_depthwise,
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class VectorVectorActivation_rtl(HLSCustomOp):
    """Class that corresponds to finn-rtl Vector Vector Unit."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "SIMD": ("i", False, 1),
            "Dim": ("ints", True, []),  # [H, W]
            "Channels": ("i", True, 0),
            "Kernel": ("ints", True, []),  # [H, W]
            "resType": ("s", False, "dsp", {"auto", "lut", "dsp"}),
            "pumpedCompute": ("i", False, 0, {0, 1}),
            "pumpedMemory": ("i", False, 0, {0, 1}),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # memory mode for the layer weights
            # const -- embedded weights, default, long compile/synth times
            # decoupled -- streaming weights with weight streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": ("s", False, "decoupled", {"const", "decoupled", "external"}),
            # (mem_mode = decoupled only) whether weights will be writable through
            # an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            # FPGA resource type for memories in decoupled mode
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
            # see also https://www.xilinx.com/support/answers/38070.html
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def minimize_accumulator_width(self, model):
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product"""
        weights = model.get_initializer(self.onnx_node.input[1])
        k_h, k_w = self.get_nodeattr("Kernel")
        fm = self.get_nodeattr("Channels")
        # put weights into the shape expected by calculate_matvec_accumulator_range
        weights = weights.reshape(fm, k_h * k_w).transpose()
        idt = self.get_input_datatype()

        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)
        # if runtime-writeable weights, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if self.get_nodeattr("runtime_writeable_weights"):
            wdt = self.get_weight_datatype()
            lower_worst = wdt.min() * np.ones_like(weights)
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones_like(weights)
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(upper_range), max(upper_range))

        # if the acc_range is always greater than 0, then acc_max <= 2^P - 1
        if acc_min >= 0:
            acc_bit_width = np.log2(acc_max + 1)
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"UINT{acc_bit_width}"]
        # if the acc_range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            _acc_max = max(-acc_min, 1 + acc_max)
            acc_bit_width = np.log2(_acc_max) + 1
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"INT{acc_bit_width}"]

        # if this is the last node in the graph, then ensure the datatype is
        # divisibly by 8 bits
        if model.find_direct_successors(self.onnx_node) is None:
            bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
            new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
            adt = DataType[new_adt_name]
        # for no-activation nodes, output dt = acc dt
        self.set_nodeattr("outputDataType", adt.name)
        self.set_nodeattr("accDataType", adt.name)

        return DataType[self.get_nodeattr("accDataType")]

    def minimize_weight_bit_width(self, model):
        """Minimize the bit width based on the values of the weights"""
        if not self.get_nodeattr("runtime_writeable_weights"):
            weights = model.get_initializer(self.onnx_node.input[1])
            w_min = weights.min()
            w_max = weights.max()
            if w_min < 0:
                if abs(w_min) > w_max:
                    wdt = DataType.get_smallest_possible(w_min)
                else:
                    wdt = DataType.get_smallest_possible(-w_max - 1)
            else:
                wdt = DataType.get_smallest_possible(w_max)
            self.set_nodeattr("weightDataType", wdt.name)
        return DataType[self.get_nodeattr("weightDataType")]

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        ch = self.get_nodeattr("Channels")
        k_h, k_w = self.get_nodeattr("Kernel")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = (k_h * k_w * ch // pe) // simd
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        return 0

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
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
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            return DataType[self.get_nodeattr("weightDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator"""
        return DataType[self.get_nodeattr("accDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        in_width = i_bits * simd * pe
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    def get_folded_input_shape(self, ind=0):
        k_h, k_w = self.get_nodeattr("Kernel")
        dim_h, dim_w = self.get_nodeattr("Dim")
        ch = self.get_nodeattr("Channels")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        kernel_2 = k_h * k_w
        assert kernel_2 % simd == 0, "Requirement kernel (k_h * k_w) divisable by SIMD is violated."
        sf = kernel_2 // simd
        assert ch % pe == 0, "Requirement Channels divisable by PE is violated."
        nf = ch // pe

        if ind == 0:
            # calculate shape of input 0
            folded_input_shape = tuple([1, dim_h, dim_w, sf * nf, simd * pe])
        elif ind == 1 and self.get_nodeattr("mem_mode") == "external":
            # calculate shape of input 1 (weights)
            folded_input_shape = tuple([1, sf * nf, pe])
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        nf = ch // pe
        dim_h, dim_w = self.get_nodeattr("Dim")
        folded_output_shape = tuple([1, dim_h, dim_w, nf, pe])
        return folded_output_shape

    def get_normal_input_shape(self, ind=0):
        dim_h, dim_w = self.get_nodeattr("Dim")
        ch = self.get_nodeattr("Channels")
        k_h, k_w = self.get_nodeattr("Kernel")
        normal_input_shape = tuple([1, dim_h, dim_w, k_h * k_w * ch])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        ch = self.get_nodeattr("Channels")
        dim_h, dim_w = self.get_nodeattr("Dim")
        normal_output_shape = tuple([1, dim_h, dim_w, ch])
        return normal_output_shape

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    # TODO: fix exp_cycles estimations --> depends on fpga_part and clk
    def get_exp_cycles(self):
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        ch = self.get_nodeattr("Channels")
        dim_h, dim_w = self.get_nodeattr("Dim")
        k_h, k_w = self.get_nodeattr("Kernel")
        # currently FINN supports for vvau a batch size of 1
        batch_size = 1
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = ((ch * k_h * k_w) / pe / simd) * batch_size * (dim_h * dim_w) / mmv
        return int(exp_cycles)

    def get_hls_compatible_weight_tensor(self, orig_weight_matrix):
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        ch = self.get_nodeattr("Channels")
        k_h, k_w = self.get_nodeattr("Kernel")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            ch,
            1,
            k_h,
            k_w,
        ), """Weights matrix doesn't
        have expected shape (channels, 1, kernel_size, kernel_size)"""
        ret = orig_weight_matrix
        if self.get_weight_datatype() == DataType["BIPOLAR"]:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        ret = ret.reshape(ch, k_h * k_w)
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        ret = ret.reshape(1, pe, wmem, simd)
        return ret

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hls_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_weight_datatype() == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", True, True)
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "const FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "const BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()
        elif "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # SIMD & PE flip
            weight_tensor_pe_simd_flipped = np.flip(weight_tensor_pe_flipped, axis=-1)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            # SIMD & PE flipped
            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_weightstream_width()
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_simd_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_simd_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_simd_flipped.flatten()
                weight_stream = weight_stream.copy()
                if self.get_nodeattr("pumpedMemory"):
                    split_w_stream = np.zeros([weight_stream.shape[0] * 2], dtype=object)
                    k = 0
                    for i in range(len(weight_stream)):
                        weight = weight_stream[i]
                        split_w_stream[k] = weight[len(weight) // 2 :]
                        split_w_stream[k + 1] = weight[: len(weight) // 2]
                        k += 2
                    weight_stream = split_w_stream
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_weightstream_width()
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
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
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        if mem_mode == "const":
            # save hlslib-compatible weights in params.h
            weight_filename = "{}/params.h".format(code_gen_dir)
            self.make_weight_file(weights, "hls_header", weight_filename)
        elif mem_mode == "decoupled" or mem_mode == "external":
            weight_filename_sim = "{}/weights.npy".format(code_gen_dir)
            # save decoupled weights for cppsim
            self.make_weight_file(weights, "decoupled_npy", weight_filename_sim)
            if mem_mode == "decoupled":
                # also save weights as Verilog .dat file
                # This file will be ignored when synthesizing UltraScale memory.
                weight_filename_rtl = self.get_decoupled_weight_filename(abspath=False)
                weight_filename_rtl = code_gen_dir + "/" + weight_filename_rtl
                self.make_weight_file(weights, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
            )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        k = self.get_nodeattr("Kernel")
        total_kernel = np.prod(k)
        pe = self.get_nodeattr("PE")
        chans = self.get_nodeattr("Channels")
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

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
                inp = context[inputs]
                # transform conventional nhwc into depthwise layout used by HW
                inp = layout_nhwc_to_hw_depthwise(inp, total_kernel, chans, pe)
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = inp.reshape(expected_inp_shape)
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
                raise Exception("Unexpected input found for VectorVectorActivation")
            in_ind += 1

        if mode == "cppsim":
            assert False, "RTL components don't support cppsim"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)

            if mem_mode == "external" or mem_mode == "decoupled":
                wnbits = self.get_weightstream_width()
                export_wdt = self.get_weight_datatype()
                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_weight_datatype() == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]
                wei = npy_to_rtlsim_input("{}/weights.npy".format(code_gen_dir), export_wdt, wnbits)
                dim_h, dim_w = self.get_nodeattr("Dim")
                num_w_reps = dim_h * dim_w

                io_dict = {
                    "inputs": {"in0": inp, "weights": wei * num_w_reps},
                    "outputs": {"out": []},
                }
                self.rtlsim_multi_io(sim, io_dict)
                output = io_dict["outputs"]["out"]
            else:
                output = self.rtlsim(sim, inp)
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
            # note that VVU produces standard NHWC output, despite using special layout
            # for the input - so nothing extra needed here
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates C++ code and tcl script for IP generation.
        Here: Generates (System-)Verilog code for IP generation."""
        self.generate_hdl(model, fpgapart, clk)

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        pass

    def code_generation_cppsim(self, model):
        """Normally: Generates C++ code for simulation (cppsim)."""
        pass

    def compile_singlenode_code(self):
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

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        sname = self.hls_sname()
        if mem_mode == "external":
            intf_names["s_axis"].append(("weights_" + sname, self.get_weightstream_width_padded()))
        if mem_mode == "decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def code_generation_ipi(self):
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            # if self.get_nodeattr("ram_style") == "ultra":
            #     assert (
            #         runtime_writable == 1
            #     ), "Layer with URAM weights must have runtime_writeable_weights=1"
            node_name = self.onnx_node.name
            sname = self.hls_sname()
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            # if we need a 2x clock for either compute or memory, instantiate the 2x clk port
            if self.get_nodeattr("pumpedCompute") or self.get_nodeattr("pumpedMemory"):
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append("create_bd_pin -dir I -type clk2x /%s/%s" % (node_name, clk2x_name))
            else:
                clk2x_name = None
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
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
            sourcefiles = [
                os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
                rtllib_dir + "mvu_vvu_axi.sv",
                rtllib_dir + "replay_buffer.sv",
                rtllib_dir + "mvu_4sx4u.sv",
                rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
                rtllib_dir + "mvu_8sx8u_dsp48.sv",
                rtllib_dir + "mvu_vvu_lut.sv",
            ]
            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    self.onnx_node.name,
                    self.onnx_node.name,
                )
            )

            # instantiate a streamer and connect it to the HLS IP
            strm_vlnv = "amd.com:finn:memstream:1.0"
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s" % (strm_vlnv, node_name, strm_inst)
            )
            cmd.append(
                "set_property -dict [list "
                "CONFIG.DEPTH {%d} "
                "CONFIG.WIDTH {%d} "
                "CONFIG.INIT_FILE {%s} "
                "CONFIG.RAM_STYLE {%s} "
                "CONFIG.PUMPED_MEMORY {%s} "
                "] [get_bd_cells /%s/%s]"
                % (
                    self.calc_wmem(),
                    self.get_weightstream_width_padded(),
                    self.get_decoupled_weight_filename(abspath=False),
                    self.get_nodeattr("ram_style"),
                    self.get_nodeattr("pumpedMemory"),
                    node_name,
                    strm_inst,
                )
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/weights_%s]"
                % (node_name, strm_inst, node_name, node_name, sname)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            # if using 2x pumped memory, connect the memstreamer's 2x clk input
            # to the 2x clock port. otherwise connect it to the regular clock port.
            if self.get_nodeattr("pumpedMemory"):
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                    % (node_name, clk2x_name, node_name, strm_inst)
                )
            else:
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                    % (node_name, clk_name, node_name, strm_inst)
                )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            # if using 2x pumped compute, connect the MVU's 2x clk input
            # to the 2x clock port. otherwise connect it to the regular clock port.
            if self.get_nodeattr("pumpedCompute"):
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                    % (node_name, clk2x_name, node_name, node_name)
                )
            else:
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                    % (node_name, clk_name, node_name, node_name)
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
            if runtime_writable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode == "external":
            # instantiate the RTL block
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
            sourcefiles = [
                os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
                rtllib_dir + "mvu_vvu_axi.sv",
                rtllib_dir + "replay_buffer.sv",
                rtllib_dir + "mvu_4sx4u.sv",
                rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
                rtllib_dir + "mvu_8sx8u_dsp48.sv",
                rtllib_dir + "mvu_vvu_lut.sv",
            ]
            for f in sourcefiles:
                cmd.append("add_files -norecurse %s" % (f))
            cmd.append(
                "create_bd_cell -type module -reference %s %s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    self.onnx_node.name,
                )
            )
        else:
            raise Exception("Unrecognized mem_mode for VectorVectorActivation")
        return cmd

    def uram_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "decoupled" and mstyle != "ultra")
            or (mmode == "const")
            or (mmode == "external")
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self):
        """Calculates resource estimation for BRAM"""
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        mem_width = Q * W * P
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # since this is HLS memory, not using the full width of a BRAM
        # assuming memories up to 128 deep get implemented in LUTs
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "decoupled" and mstyle in ["distributed", "ultra"])
            or (mstyle == "auto" and self.calc_wmem() <= 128)
            or (mmode == "const" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0

        if mem_width == 1:
            return math.ceil(omega / 16384)
        elif mem_width == 2:
            return math.ceil(omega / 8192)
        elif mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        elif mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 8))
        elif mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 16))
        else:
            return (math.ceil(omega / 512)) * (math.ceil(mem_width / 32))

    def bram_efficiency_estimation(self):
        P = self.get_nodeattr("PE")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * P * omega
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    # TODO: fix estimations
    def lut_estimation(self):
        """Calculates resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        c2 = 0
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "decoupled" and mstyle == "distributed") or (
            mmode == "const" and self.calc_wmem() <= 128
        ):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        res_type = self.get_nodeattr("resType")
        if res_type == "dsp":
            mult_luts = 0
        else:
            mult_luts = Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
        # adder tree
        addertree_luts = (W + A) * (2 * Q - 1)
        # accumulator
        acc_datatype = self.get_accumulator_datatype()
        acc_bits = acc_datatype.bitwidth()
        k_h, k_w = self.get_nodeattr("Kernel")
        # if accDataType is not set, then it will default to INT32, which would
        # be a large overestimate in most (if not all) cases. In this scenario,
        # we would use the minimum accumulator as determined by the data types
        # bound, derived in https://arxiv.org/abs/2301.13376
        alpha = math.log(k_h * k_w, 2) + W + A - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits
        return int(c0 + c1 * (P * (mult_luts + addertree_luts + acc_luts)) + c2)

    # TODO: fix estimations
    def dsp_estimation(self):
        # multiplication
        P = self.get_nodeattr("PE")
        res_type = self.get_nodeattr("resType")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        if res_type == "dsp":
            mult_dsp = P * np.ceil((W + A) / 48)  # TODO: more accurate modelling
        else:
            mult_dsp = 0
        return int(mult_dsp)

    def get_weightstream_width(self):
        """Returns weight stream width. Used only in decoupled mode."""
        if (
            self.get_nodeattr("mem_mode") == "decoupled"
            or self.get_nodeattr("mem_mode") == "external"
        ):
            simd = self.get_nodeattr("SIMD")
            pe = self.get_nodeattr("PE")
            wp = self.get_weight_datatype().bitwidth()
            w_width = simd * pe * wp
            return w_width
        else:
            return 0

    def get_weightstream_width_padded(self):
        """Returns weight stream width padded to a multiple of 8. This is required
        by the AXI Stream spec. Used in decoupled mode."""
        weight_width = self.get_weightstream_width()
        return roundup_to_integer_multiple(weight_width, 8)

    def get_op_and_param_counts(self):
        k_h, k_w = self.get_nodeattr("Kernel")
        fm = self.get_nodeattr("Channels")
        dim_h, dim_w = self.get_nodeattr("Dim")
        weight_bits = self.get_weight_datatype().bitwidth()
        inp_bits = self.get_input_datatype().bitwidth()
        num_repetitions = int(dim_h * dim_w)
        mac_count = k_h * k_w * fm * num_repetitions
        # cannonicalize op type: highest bitwidth operand first s.t.
        # e.g. mac_8bx4b and mac_4bx8b don't appear as two different op types
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = "op_mac_%dbx%db" % (bw1, bw2)
        weight_param_type = "param_weight_%db" % (weight_bits)
        weight_count = k_h * k_w * fm
        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}
        return ret_dict

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out": []},
        }
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["decoupled", "external"]:
            n_weight_inps = self.calc_wmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["weights"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    def generate_hdl(self, model, fpgapart, clk):
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)

        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)
        # add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to template
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(0)))
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(1)))

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _resolve_segment_len(self, clk):
        # Insert pipeline registers in the DSP58 chain to meet target clock frequency
        # 0.741 ns seems the worst-case delay through first DSP
        # 0.605 ns seems to be (on average) delay for all subsequent DSPs
        # clk >= (critical_path_dsps - 1) * 0.605 + 0.741
        """assert (
            clk > 0.741
        ), "Infeasible clk target of {} ns has been set, consider
            lowering the targeted clock frequency!".format(
            clk
        )
        critical_path_dsps = np.floor((clk - 0.741) / 0.605 + 1)
        max_chain_len = np.ceil(self.get_nodeattr("SIMD") / 3)
        dsp_chain_len = critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len
        return dsp_chain_len"""
        # copying fixed behavior from RTL MVU
        return 1

    def _resolve_impl_style(self, fpgapart):
        # Based on target device and activation/weight-width, choose the
        # supported RTL compute core
        if self.get_nodeattr("resType") == "lut":
            return "mvu_vvu_lut"
        else:
            is_dsp_targeted = self.get_nodeattr("resType") == "dsp"
            is_versal = (
                fpgapart[0:4] in ["xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"]
                or fpgapart[0:5] == "xqrvc"
            )
            assert (
                is_dsp_targeted and is_versal
            ), "DSP-based (RTL) VVU currently only supported on Versal (DSP58) devices"
            return "mvu_vvu_8sx9_dsp58"

    def prepare_codegen_default(self, fpgapart, clk):
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mvu/mvu_vvu_axi_wrapper.v"

        code_gen_dict = {}
        code_gen_dict["$IS_MVU$"] = [str(0)]
        code_gen_dict["$COMPUTE_CORE$"] = [self._resolve_impl_style(fpgapart)]
        code_gen_dict["$PUMPED_COMPUTE$"] = [str(self.get_nodeattr("pumpedCompute"))]
        mw = int(np.prod(self.get_nodeattr("Kernel")))
        code_gen_dict["$MW$"] = [str(mw)]
        code_gen_dict["$MH$"] = [str(self.get_nodeattr("Channels"))]
        code_gen_dict["$PE$"] = [str(self.get_nodeattr("PE"))]
        code_gen_dict["$SIMD$"] = [str(self.get_nodeattr("SIMD"))]
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
        code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Path to (System-)Verilog files used by top-module & path to top-module
        verilog_paths = [code_gen_dir, os.environ["FINN_ROOT"] + "/finn-rtllib/mvu"]
        verilog_files = [self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"]
        single_src_dir = make_build_dir("pyverilator_" + self.onnx_node.name + "_")
        dat_files = self.get_all_meminit_filenames(abspath=True)
        for dat_file in dat_files:
            shutil.copy(dat_file, single_src_dir)

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=single_src_dir,
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
            auto_eval=False,
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)

        return sim

    def get_all_verilog_paths(self):
        "Return list of all folders containing Verilog code for this node."

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Path to (System-)Verilog files used by top-module & path to top-module
        verilog_paths = [code_gen_dir, os.environ["FINN_ROOT"] + "/finn-rtllib/mvu"]
        return verilog_paths

    def get_verilog_top_filename(self):
        "Return the Verilog top module filename for this node."

        verilog_file = "{}/{}_wrapper.v".format(
            self.get_nodeattr("code_gen_dir_ipgen"), self.get_nodeattr("gen_top_module")
        )
        return verilog_file

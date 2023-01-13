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
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

from . import templates

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class Thresholding_Batch(HLSCustomOp):
    """Class that corresponds to finn-hls Thresholding_Batch function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.decoupled_wrapper = templates.decoupled_wrapper

    def get_nodeattr_types(self):
        my_attrs = {
            # parallelization; channels thresholded per cycle
            "PE": ("i", True, 0),
            # number of channels (each may have different thresholds)
            "NumChannels": ("i", True, 0),
            # number of steps in thresholding function
            "numSteps": ("i", True, 1),
            # string defining memory type
            "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # initialization value for the thresholding accumulator
            "ActVal": ("i", False, 0),
            # memory mode for the thresholds
            # const -- embedded thresholds, default
            # decoupled -- streaming thresholds with  streamer packaged inside IP
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
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return mh // pe

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype().name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The required Threshold_Batch attributes do not exist."""
            )

        return info_messages

    def bram_estimation(self):
        """Calculates BRAM cost if resource set to BRAM"""
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return int(ceil(A * P / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    def lut_estimation(self):
        """Calculates LUT cost, taking memory resource type into account"""
        # TODO add in/out FIFO contributions
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        tmem = self.calc_tmem()
        # cost of comparators
        comparator_cost = A * P
        # cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = P * A * int(ceil(tmem / 64))
        else:
            lutram_cost = 0
        # total cost
        return comparator_cost + lutram_cost

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_weight_datatype(self):
        """Returns FINN DataType of thresholds, here called weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def minimize_accumulator_width(self, model):
        "Minimize threshold width ('accumulator width' here due to convention)"
        thresholds = model.get_initializer(self.onnx_node.input[1])
        threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()
        min_input = self.get_input_datatype().min()
        max_input = self.get_input_datatype().max()
        # get range required by threshold values
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        self.set_nodeattr("weightDataType", tdt.name)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("PE")

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_weightstream_width(self):
        """Returns weight stream width. Used only in decoupled mode."""
        if self.get_nodeattr("mem_mode") == "decoupled":
            pe = self.get_nodeattr("PE")
            wp = self.get_weight_datatype().bitwidth()
            n_thres_steps = self.get_nodeattr("numSteps")
            w_width = pe * wp * n_thres_steps
            return w_width
        else:
            return 0

    def get_weightstream_width_padded(self):
        """Returns weight stream width padded to a multiple of 8. This is required
        by the AXI Stream spec. Used in decoupled mode."""
        weight_width = self.get_weightstream_width()
        return roundup_to_integer_multiple(weight_width, 8)

    def get_ap_int_max_w(self):
        temp_value = super().get_ap_int_max_w()
        weightstream = self.get_weightstream_width()
        return max([weightstream, temp_value])

    def get_folded_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        fold = ich // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        # fill in TSrcI
        ret["TSrcI"] = "Slice<%s>" % inp_hls_str
        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

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
        threshold_tensor = self.get_hls_compatible_threshold_tensor(weights)
        tdt = self.get_weight_datatype()
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        if weight_file_mode == "hls_header":
            # save thresholds in thresh.h
            thresholds_hls_code = numpy_to_hls_code(
                threshold_tensor, tdt, "thresholds", False, True
            )
            # write thresholds into thresh.h
            f_thresh = open(weight_file_name, "w")
            tdt_hls = tdt.get_hls_datatype_str()
            # use binary to export bipolar activations
            export_odt = self.get_output_datatype()
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                export_odt = DataType["BINARY"]
            odt_hls = export_odt.get_hls_datatype_str()
            f_thresh.write(
                "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
                = ".format(
                    self.calc_tmem(),
                    self.get_nodeattr("PE"),
                    threshold_tensor.shape[-1],
                    tdt_hls,
                    odt_hls,
                    self.get_nodeattr("ActVal"),
                    "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                )
            )
            f_thresh.write(thresholds_hls_code)
            f_thresh.close()
        elif "decoupled" in weight_file_mode:
            # streaming thresholds need to be organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
            # TODO add flips/reversals as needed here
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
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        code_gen_dir = path
        thresholds = model.get_initializer(self.onnx_node.input[1])
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            # save thresholds in thresh.h
            weight_filename = "{}/thresh.h".format(code_gen_dir)
            self.make_weight_file(thresholds, "hls_header", weight_filename)
        elif mem_mode == "decoupled":
            # save decoupled weights for cppsim
            weight_filename_sim = "{}/thresholds.npy".format(code_gen_dir)
            self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)
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
            ram_style = self.get_nodeattr("ram_style")
            if ram_style == "ultra":
                # UltraRAM must have no memory initializer, or only zeroes
                # otherwise BRAM will be inferred instead of URAM
                # as a workaround we provide a zero-weight init here
                synth_thresholds = np.zeros_like(thresholds, dtype=np.float32)
            else:
                synth_thresholds = thresholds
            self.make_weight_file(
                synth_thresholds, "decoupled_verilog_dat", weight_filename_rtl_synth
            )
        else:
            raise Exception("Unrecognized mem_mode")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
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

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            oshape = self.get_normal_output_shape()
            assert (
                context[node.output[0]].shape == oshape
            ), """Output shape is not as expected"""
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            if self.get_nodeattr("mem_mode") == "decoupled":
                wnbits = self.get_weightstream_width()
                export_wdt = self.get_weight_datatype()
                wei = npy_to_rtlsim_input(
                    "{}/thresholds.npy".format(code_gen_dir), export_wdt, wnbits
                )
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp, "weights": wei * num_w_reps},
                    "outputs": {"out": []},
                }
                self.rtlsim_multi_io(sim, io_dict)
                output = io_dict["outputs"]["out"]
            elif self.get_nodeattr("mem_mode") == "const":
                output = self.rtlsim(sim, inp)
            else:
                raise Exception("Unrecognized mem_mode")
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
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        if self.get_nodeattr("mem_mode") == "const":
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    # TODO check and add whatever missing
    def defines(self, var):
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = int(np.prod(numInputVectors))
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumChannels1 {}\n #define PE1 {}\n #define numReps {}""".format(
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("PE"),
                numReps,
            )
        ]
        if self.get_nodeattr("mem_mode") == "decoupled":
            self.code_gen_dict["$DEFINES$"].append(
                "#define ActVal1 %d" % self.get_nodeattr("ActVal")
            )
            self.code_gen_dict["$DEFINES$"].append(
                "#define ThresType1 %s"
                % self.get_weight_datatype().get_hls_datatype_str()
            )
            self.code_gen_dict["$DEFINES$"].append(
                "#define NumSteps1 %d" % self.get_nodeattr("numSteps")
            )

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            tdt = self.get_weight_datatype()
            elem_bits = tdt.bitwidth()
            packed_bits = self.get_weightstream_width()
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = tdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/thresholds.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", weights, false, numReps);'
                % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
            )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> weights ("weights");'.format(
                    self.get_weightstream_width()
                )
            )

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        # TODO: why put some template parameters into defines and not others?
        # should ImgDim be defined or just filled in here like we do now?
        node = self.onnx_node
        inp_vecs = self.get_nodeattr("numInputVectors")
        total_spatial_size = int(np.prod(inp_vecs))
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<{}, NumChannels1, PE1, {}, {}>
                (in0, out, threshs, numReps);""".format(
                    node.op_type,
                    total_spatial_size,
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        elif mem_mode == "decoupled":
            # note that numReps is set to 1 in the invocation below, since
            # - for cppsim the repetition comes from the threshold stream reader+input
            # - for synth the unit runs continuously anyway (ap_ctrl_none)
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<{}, NumChannels1, PE1, {}, {}, ActVal1, ThresType1, NumSteps1>
                (in0, out, weights, 1);""".format(
                    "Thresholding_Stream_Batch",
                    total_spatial_size,
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        if self.get_nodeattr("mem_mode") == "const":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0,
                    hls::stream<ap_uint<{}>> &out
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.get_outstream_width(),
                )
            ]
        elif self.get_nodeattr("mem_mode") == "decoupled":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0,
                    hls::stream<ap_uint<{}>> &weights,
                    hls::stream<ap_uint<{}>> &out
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.get_weightstream_width(),
                    self.get_outstream_width(),
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

        if self.get_nodeattr("mem_mode") == "const":
            # the threshold tensor is acc_type [PE][TMEM][N_THRES]
            # partition for parallel access along PE and N_THRES
            # dimensions (dims 1 and 3)
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=1"
                )
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=3"
                )
            )
            # set resource type
            ram_style = self.get_nodeattr("ram_style")
            pe = self.get_nodeattr("PE")
            ich = self.get_nodeattr("NumChannels")
            # if PE less than NumChannels, assign cores according to ram_style;
            # otherwise if PE == NumChannels, Vivado HLS will unroll to FFs
            if pe < ich:
                if ram_style == "distributed":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        (
                            "#pragma HLS RESOURCE variable=threshs.m_thresholds "
                            "core=ROM_2P_LUTRAM"
                        )
                    )
                elif ram_style == "block":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        (
                            "#pragma HLS RESOURCE variable=threshs.m_thresholds "
                            "core=ROM_2P_BRAM"
                        )
                    )
                else:
                    raise Exception(
                        """Invalid value for attribute ram_style! Is currently set to: {}
                    has to be set to one of ("block", "distributed")""".format(
                            ram_style
                        )
                    )
        elif self.get_nodeattr("mem_mode") == "decoupled":
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=weights name=weights_"
                + self.hls_sname()
            )

    def code_generation_ipi(self):
        cmd = []
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            node_name = self.onnx_node.name
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            sname = self.hls_sname()
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
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
                % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate the hls ip
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (self.get_nodeattr("ip_vlnv"), node_name, node_name)
            )
            # instantiate a streamer and connect it to the HLS IP
            strm_vlnv = "xilinx.com:user:memstream:1.0"
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (strm_vlnv, node_name, strm_inst)
            )
            cmd.append(
                "set_property -dict [list "
                "CONFIG.NSTREAMS {1} "
                "CONFIG.MEM_DEPTH {%d} "
                "CONFIG.MEM_WIDTH {%d} "
                "CONFIG.MEM_INIT {%s} "
                "CONFIG.RAM_STYLE {%s} "
                "CONFIG.STRM0_DEPTH {%d} "
                "CONFIG.STRM0_WIDTH {%d} "
                "CONFIG.STRM0_OFFSET {0} "
                "] [get_bd_cells /%s/%s]"
                % (
                    self.calc_tmem(),
                    self.get_weightstream_width_padded(),
                    self.get_nodeattr("code_gen_dir_ipgen") + "/",
                    self.get_nodeattr("ram_style"),
                    self.calc_tmem(),
                    self.get_weightstream_width_padded(),
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
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aresetn]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aclk]"
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
            cmd.append("save_bd_design")
        elif mem_mode == "const":
            # base class impl sufficient for const mode
            return super().code_generation_ipi()
        else:
            raise Exception("Unrecognized mem_mode for Thresholding_Batch")
        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_op_and_param_counts(self):
        ret_dict = {}
        weight_bits = self.get_weight_datatype().bitwidth()
        out_features = self.get_nodeattr("NumChannels")
        num_steps = self.get_nodeattr("numSteps")
        # thresholds are called weights in this layer
        thres_param_type = "param_threshold_%db" % (weight_bits)
        thres_count = out_features * num_steps
        ret_dict[thres_param_type] = thres_count
        return ret_dict

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."

        return ["config_compile -pipeline_style frp"]

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
            n_weight_inps = self.calc_tmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["weights"] = [
                0 for i in range(num_w_reps * n_weight_inps)
            ]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

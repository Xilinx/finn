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
import warnings
from math import ceil
from onnx import helper

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
)

from . import templates

# ONNX i/o tensor shape assumptions for channelwise ops:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the channelwise parameter tensor, shape (NumChannels, params_per_channel)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


def get_smallest_possible(vals):
    """Returns smallest (fewest bits) possible DataType that can represent
    value. Prefers unsigned integers where possible."""
    vals = np.array(vals)
    for v in vals:
        assert int(v) == v, "Error float value"

    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            # not currently supported
            continue

        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt

    warnings.warn(
        """InferChannelwiseLinearLayer: Output values may not be
    representable with supported data types.
    Setting maximum width data type available.
    This will lead to errors if there are no constrains on the input
    """
    )

    if (0 <= vals).all():
        return DataType["UINT64"]
    else:
        return DataType["INT64"]


class ChannelwiseOp_Batch(HLSCustomOp):
    """Class that corresponds to finn-hls Thresholding_Batch function.
    It can implement a variety of channel-wise parametrized operations,
    including Add, Mul and multi-thresholding.
    """

    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.decoupled_wrapper = templates.decoupled_wrapper

    def get_nodeattr_types(self):
        my_attrs = {
            # channelwise "map" function to apply:
            # one of cmp_le, cmp_ge, add, mul
            "Func": ("s", False, "cmp_le", {"cmp_le", "cmp_ge", "add", "mul"}),
            "PE": ("i", True, 0),
            "NumChannels": ("i", True, 0),
            # string defining memory resource type for parameters
            "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "paramDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # input and output FIFO depths
            "inFIFODepth": ("i", False, 0),
            "outFIFODepth": ("i", False, 0),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_tmem(self):
        """Calculates and returns TMEM, the depth of the memory used
        to store the channelwise op parameters."""
        chn = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return chn // pe

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        # implement tensor with correct shape
        return helper.make_node(
            "RandomNormal",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            shape=list(oshape),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check input datatype against property
        idt = model.get_tensor_datatype(node.input[0])

        exp_idt_name = self.get_nodeattr("inputDataType")
        if exp_idt_name != idt.name:
            func = self.get_nodeattr("Func")
            assert func in ["add", "mul"], "Bad input DataType for ChannelwiseOp layer"

            self.set_nodeattr("inputDataType", idt.name)
            # update the func in ['add','mul'] cases

            # get parameter ranges
            param = model.get_initializer(node.input[1])
            param_min = min(param.flatten())
            param_max = max(param.flatten())

            # set function and determine output data type
            if func == "add":
                out_min = idt.min() + param_min
                out_max = idt.max() + param_max
                odt = get_smallest_possible([out_min, out_max])
            elif func == "mul":
                possible_limits = []
                possible_limits += [idt.min() * param_min]
                possible_limits += [idt.min() * param_max]
                possible_limits += [idt.max() * param_min]
                possible_limits += [idt.max() * param_max]
                odt = get_smallest_possible(possible_limits)

            self.set_nodeattr("outputDataType", odt.name)

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
            self.get_nodeattr("paramDataType")
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

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("PE")

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_folded_input_shape(self):
        ich = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        fold = ich // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self):
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

    def get_hls_compatible_parameter_tensor(self, orig_param_vector):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure chn % PE == 0
        * interleave rows between PEs
        * reshape into (PE, TMEM) and return
        """
        chn = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = chn // pe
        assert chn % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_param_vector.ndim == 1
        ), """Parameter vector dimension is {}.
        Expected dimension: 1.""".format(
            orig_param_vector.ndim
        )

        # if not self.get_input_datatype().signed():
        #     # ensure all thresholds are nonnegative
        #     assert (orig_param_vector >= 0).all()

        # ensure all thresholds are integer
        assert (orig_param_vector.astype(np.int32) == orig_param_vector).all()
        ret = orig_param_vector

        assert (
            ret.shape[0] == chn
        ), "Cardinality of parameter vector is not as expected (chn)"

        # distribute rows between PEs
        ret = ret.reshape(tmem, pe).transpose()
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""

        return ret.reshape(1, pe, tmem)

    def generate_params(self, model, path):
        code_gen_dir = path
        # save thresholds in params.h
        parameters = model.get_initializer(self.onnx_node.input[1])
        parameter_tensor = self.get_hls_compatible_parameter_tensor(parameters)
        pdt = DataType[self.get_nodeattr("paramDataType")]

        parameters_hls_code = numpy_to_hls_code(
            parameter_tensor, pdt, "parameters", False, True
        )
        # get input data type
        export_idt = self.get_input_datatype()
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            export_idt = DataType["BINARY"]
        idt_hls = export_idt.get_hls_datatype_str()

        # write parameters into params.h
        f_params = open("{}/params.h".format(code_gen_dir), "w")
        pdt_hls = pdt.get_hls_datatype_str()
        # use binary to export bipolar activations
        export_odt = self.get_output_datatype()
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            export_odt = DataType["BINARY"]
        odt_hls = export_odt.get_hls_datatype_str()
        # get desired function
        func = self.get_nodeattr("Func")
        if func == "cmp_le":
            func_str = "comp::less_equal"
        elif func == "cmp_ge":
            func_str = "std::greater_equal"
        elif func == "add":
            func_str = "std::plus"
        elif func == "mul":
            func_str = "std::multiplies"
        else:
            raise Exception(
                """Invalid value for attribute Func! Is currently set to: {}
            has to be set to one of the following value
            ("cmp_le", "cmp_ge", "add", "mul")""".format(
                    func
                )
            )
        f_params.write(
            "static ChannelWiseOperation<{},{},{},{},{},{}> threshs \
            = ".format(
                self.calc_tmem(),
                self.get_nodeattr("PE"),
                idt_hls,
                pdt_hls,
                odt_hls,
                "%s<%s>" % (func_str, odt_hls),
            )
        )
        f_params.write(parameters_hls_code)
        f_params.close()

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
                export_idt = self.get_input_datatype()
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for ChannelwiseOp_Batch")
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
            assert (
                context[node.output[0]].shape == self.get_folded_output_shape()
            ), """Output shape is not as expected"""
            # reshape output to have expected shape
            oshape = self.get_normal_output_shape()
            context[node.output[0]] = context[node.output[0]].reshape(*oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            output = self.rtlsim(sim, inp)
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
        self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']

    # TODO check and add whatever missing
    def defines(self, var):
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = numInputVectors[0]
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumChannels1 {}\n#define PE1 {}\n#define numReps {}""".format(
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("PE"),
                numReps,
            )
        ]

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

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        # TODO: why put some template parameters into defines and not others?
        # should ImgDim be defined or just filled in here like we do now?
        ishape = self.get_folded_input_shape()
        if len(ishape) == 3:
            imgdim_h = 1
            imgdim_w = 1
        elif len(ishape) == 5:
            imgdim_h = ishape[1]
            imgdim_w = ishape[2]
        else:
            raise Exception("""Unexpeted input shape""")
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """Thresholding_Batch<{}, {}, NumChannels1, PE1, {}, {}>
            (in0, out, threshs, numReps);""".format(
                imgdim_h,
                imgdim_w,
                tmpl_args["TSrcI"],
                tmpl_args["TDstI"],
            )
        ]

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
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0,
                hls::stream<ap_uint<{}>> &out
                )""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

        # the channelwise parameter tensor is acc_type [PE][TMEM][N_PARAMS_PER_CHANNEL]
        # partition for parallel access along PE and N_PARAMS_PER_CHANNEL
        # dimensions (dims 1 and 3)
        self.code_gen_dict["$PRAGMAS$"].append(
            (
                "#pragma HLS ARRAY_PARTITION variable=threshs.parameters "
                "complete dim=1"
            )
        )
        # self.code_gen_dict["$PRAGMAS$"].append(
        #     (
        #         "#pragma HLS ARRAY_PARTITION variable=threshs.parameters "
        #         "complete dim=3"
        #     )
        # )

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
                        "#pragma HLS RESOURCE variable=threshs.parameters "
                        "core=ROM_2P_LUTRAM"
                    )
                )
            elif ram_style == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    (
                        "#pragma HLS RESOURCE variable=threshs.parameters "
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

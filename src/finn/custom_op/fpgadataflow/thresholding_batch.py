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

from math import ceil
import os

import numpy as np

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
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
            "PE": ("i", True, 0),
            "NumChannels": ("i", True, 0),
            # string defining memory type
            "ram_style": ("s", False, "distributed"),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
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
        """Calculates and returns TMEM."""
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return mh // pe

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check input datatype against property
        idt_name = self.get_input_datatype().name
        exp_idt_name = self.get_nodeattr("inputDataType")
        assert exp_idt_name == idt_name, "Bad input DataType for Thresholding layer"
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        info_messages = []
        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

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
        """Calculates LUT cost, taking memory resource type into account """
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
        if not self.get_input_datatype().signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        ret = orig_thres_matrix
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

    def generate_params(self, model, path):
        code_gen_dir = path
        # save thresholds in thresh.h
        thresholds = model.get_initializer(self.onnx_node.input[1])

        threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
        tdt = DataType.INT32
        thresholds_hls_code = numpy_to_hls_code(
            threshold_tensor, tdt, "thresholds", False, True
        )
        # write thresholds into thresh.h
        f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
        tdt_hls = tdt.get_hls_datatype_str()
        # use binary to export bipolar activations
        export_odt = self.get_output_datatype()
        if self.get_output_datatype() == DataType.BIPOLAR:
            export_odt = DataType.BINARY
        odt_hls = export_odt.get_hls_datatype_str()
        f_thresh.write(
            "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
            = ".format(
                self.calc_tmem(),
                self.get_nodeattr("PE"),
                threshold_tensor.shape[-1],
                tdt_hls,
                odt_hls,
                export_odt.min(),
                "std::less_equal<%s>" % tdt_hls,
            )
        )
        f_thresh.write(thresholds_hls_code)
        f_thresh.close()

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
                if self.get_input_datatype() == DataType.BIPOLAR:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType.BINARY
                else:
                    export_idt = self.get_input_datatype()
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for StreamingFCLayer")
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType.BIPOLAR:
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
        self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    # TODO check and add whatever missing
    def defines(self, var):
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = numInputVectors[0]
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumChannels1 {}\n #define PE1 {}\n #define numReps {}""".format(
                self.get_nodeattr("NumChannels"), self.get_nodeattr("PE"), numReps,
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
        node = self.onnx_node
        ishape = self.get_folded_input_shape()
        if len(ishape) == 3:
            imgdim = 1
        elif len(ishape) == 5:
            imgdim = ishape[1]
        else:
            raise Exception("""Unexpeted input shape""")
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<{}, NumChannels1, PE1, {}, {}>
            (in0, out, threshs, numReps);""".format(
                node.op_type, imgdim, tmpl_args["TSrcI"], tmpl_args["TDstI"],
            )
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
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

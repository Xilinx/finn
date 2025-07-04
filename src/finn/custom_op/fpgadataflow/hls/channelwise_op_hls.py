# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
from math import ceil
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.channelwise_op import ChannelwiseOp
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import numpy_to_hls_code

# ONNX i/o tensor shape assumptions for channelwise ops:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the channelwise parameter tensor, shape (NumChannels, params_per_channel)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class ChannelwiseOp_hls(ChannelwiseOp, HLSBackend):
    """Class that corresponds to finn-hls Thresholding_Batch function.
    It can implement a variety of channel-wise parametrized operations,
    including Add, Mul and multi-thresholding.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(ChannelwiseOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

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
            info_messages.append("""The required Threshold_Batch attributes do not exist.""")

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

        assert ret.shape[0] == chn, "Cardinality of parameter vector is not as expected (chn)"

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

        parameters_hls_code = numpy_to_hls_code(parameter_tensor, pdt, "parameters", False, True)
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
            func_str = "comp::less_equal<%s, %s>" % (idt_hls, pdt_hls)
        elif func == "cmp_ge":
            func_str = "comp::greater_equal<%s, %s>" % (idt_hls, pdt_hls)
        elif func == "add":
            func_str = "comp::add<%s, %s, %s>" % (odt_hls, odt_hls, odt_hls)
        elif func == "mul":
            func_str = "comp::mul<%s, %s, %s>" % (odt_hls, odt_hls, odt_hls)
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
                func_str,
            )
        )
        f_params.write(parameters_hls_code)
        f_params.close()

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']

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
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_V, false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                npy_in,
            )
        )

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

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        # TODO: why put some template parameters into defines and not others?
        # should ImgDim be defined or just filled in here like we do now?
        ishape = self.get_folded_input_shape()
        if len(ishape) == 3:
            spatial_dim = 1
        elif len(ishape) == 5:
            spatial_dim = ishape[1] * ishape[2]
        else:
            raise Exception("""Unexpeted input shape""")
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """Thresholding_Batch<{}, NumChannels1, PE1, {}, {}>
            (in0_V, out0_V, threshs, numReps);""".format(
                spatial_dim,
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
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0_V, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0_V,
                hls::stream<ap_uint<{}>> &out0_V
                )""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        # the channelwise parameter tensor is acc_type [PE][TMEM][N_PARAMS_PER_CHANNEL]
        # partition for parallel access along PE and N_PARAMS_PER_CHANNEL
        # dimensions (dims 1 and 3)
        self.code_gen_dict["$PRAGMAS$"].append(
            ("#pragma HLS ARRAY_PARTITION variable=threshs.parameters " "complete dim=1")
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
                    ("#pragma HLS RESOURCE variable=threshs.parameters " "core=ROM_2P_LUTRAM")
                )
            elif ram_style == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.parameters " "core=ROM_2P_BRAM")
                )
            else:
                raise Exception(
                    """Invalid value for attribute ram_style! Is currently set to: {}
                has to be set to one of ("block", "distributed")""".format(
                        ram_style
                    )
                )

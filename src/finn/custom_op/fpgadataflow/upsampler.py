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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class UpsampleNearestNeighbour_Batch(HLSCustomOp):
    """
    Corresponds to finn-hlslib UpsampleNearestNeighbour_Batch function.
    Upsampling is done with the Nearest Neighbour algorithm.
    The layer expects square feature maps for the in and output.
    """

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # Size of the output feature map
            "OFMDim": ("i", True, 0),
            # Size of the input feature map
            "IFMDim": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "numInputVectors": ("i", False, 1),
            # Dimensionality mode: 0 = 2D square, 1 = 1D in H dim
            "DimMode": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        OFMDim = self.get_nodeattr("OFMDim")
        batch_size = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        reps = 1
        if is_2d:
            OFMDim = OFMDim * OFMDim
            reps = batch_size
        exp_cycles = OFMDim * reps
        return int(exp_cycles)

    def get_normal_input_shape(self):
        IFMDim = self.get_nodeattr("IFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        if is_2d:
            ishape = (batch, IFMDim, IFMDim, num_ch)
        else:
            ishape = (batch, IFMDim, 1, num_ch)
        return ishape

    def get_normal_output_shape(self):
        OFMDim = self.get_nodeattr("OFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        if is_2d:
            oshape = (batch, OFMDim, OFMDim, num_ch)
        else:
            oshape = (batch, OFMDim, 1, num_ch)
        return oshape

    def get_folded_input_shape(self):
        normal_ishape = list(self.get_normal_input_shape())
        return tuple(normal_ishape)

    def get_folded_output_shape(self):
        normal_oshape = list(self.get_normal_output_shape())
        return tuple(normal_oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert (
            ishape == exp_ishape
        ), "Unexpect input shape for UpsampleNearestNeighbour_Batch."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        ifm_ch = self.get_nodeattr("NumChannels")
        return ibits * ifm_ch

    def get_outstream_width(self):
        obits = self.get_output_datatype().bitwidth()
        ifm_ch = self.get_nodeattr("NumChannels")
        return obits * ifm_ch

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "upsample.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        ifm_ch = self.get_nodeattr("NumChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMChannels {}".format(ifm_ch)]

        ibits = self.get_input_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define Input_precision {}".format(ibits)]

        idim = self.get_nodeattr("IFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMDim {}".format(idim)]

        odim = self.get_nodeattr("OFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define OFMDim {}".format(odim)]

        batch_size = self.get_nodeattr("numInputVectors")
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(batch_size)]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
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
        is_2d = self.get_nodeattr("DimMode") == 0
        batch = self.get_nodeattr("numInputVectors")
        if is_2d:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """UpsampleNearestNeighbour_Batch<OFMDim, IFMDim, IFMChannels,
                ap_uint<Input_precision> > (in0, out, numReps);"""
            ]
        else:
            assert batch == 1, "1D upsampler currently needs numReps=1"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """UpsampleNearestNeighbour_1D<OFMDim, IFMDim, IFMChannels,
                ap_uint<Input_precision> > (in0, out);"""
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
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
            % (self.onnx_node.name, packed_hls_type, packed_hls_type)
        ]

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

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_oshape = self.get_folded_output_shape()

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

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (numInputVectors, ImgDim, ImgDim, NumChannels)."""
        export_idt = self.get_input_datatype()
        self.dynamic_input_to_npy(context, 1, target_dir=code_gen_dir)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == folded_oshape
            ), "cppsim did not produce expected folded output shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output shape doesn't match expected shape
            (1, OutputDim, OutputDim, NumChannels)."""

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
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class GlobalAccPool_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib AccPool_Batch function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 0),
            # FINN DataTypes for input
            "inputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        ch = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ch])
        return ishape

    def get_folded_input_shape(self, ind=0):
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        assert ch % pe == 0, "PE must divide NumChannels"
        folds = int(ch / pe)
        folded_ishape = tuple(vecs + [folds, pe])
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        ch = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        if len(vecs) == 1:
            oshape = tuple(vecs + [ch])
        elif len(vecs) == 3:
            oshape = tuple([vecs[0]] + [1, 1, ch])
        return oshape

    def get_folded_output_shape(self, ind=0):
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        unfolded_shape = list(self.get_normal_output_shape())
        assert ch % pe == 0, "PE must divide NumChannels"
        folds = int(ch / pe)
        oshape = tuple(unfolded_shape[:-1] + [folds, pe])
        return oshape

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."
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
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The required GlobalAccPool_Batch attributes do not exist."""
            )

        # verify that input data is 2D
        if len(self.get_nodeattr("numInputVectors")) != 3:
            info_messages.append("""GlobalAccPool_Batch requires 2D data input.""")
            raise Exception

        return info_messages

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        # determine data type from image size and input type
        idt = DataType[self.get_nodeattr("inputDataType")]
        vecs = list(self.get_nodeattr("numInputVectors"))
        npixels = vecs[-1] * vecs[-2]
        if idt.signed():
            extreme_value = npixels * idt.min()
        else:
            extreme_value = npixels * idt.max()
        return DataType.get_smallest_possible(extreme_value)

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        ibits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = pe * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        obits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        out_width = pe * obits
        return out_width

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[1:-1])

    def get_exp_cycles(self):
        # Channels/PE * batch size * idim * idim + Channels/PE
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        folds = int(ch / pe)
        return int(np.prod(self.get_folded_input_shape()[:-1]) + folds)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

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
        assert inp.shape == exp_ishape, """Input shape doesn't match expected shape ."""
        export_idt = self.get_input_datatype()
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim \
            did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = self.get_output_datatype()
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
        ), """Output shape doesn't match expected shape."""

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

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
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """AccPool_Batch<{}, {}, {}, {}, {}> (in0, out, 1);""".format(
                self.get_normal_input_shape()[1],
                self.get_nodeattr("NumChannels"),
                self.get_input_datatype().get_hls_datatype_str(),
                self.get_nodeattr("PE"),
                self.get_output_datatype().get_hls_datatype_str(),
            )
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
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
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0,
                hls::stream<ap_uint<{}>> &out)""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
            )
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

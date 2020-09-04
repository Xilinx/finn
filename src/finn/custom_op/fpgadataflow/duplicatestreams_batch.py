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

import os

import numpy as np

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp
from onnx import helper, TensorProto
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class DuplicateStreams_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib function of the same name."""

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

    def get_normal_input_shape(self):
        ch = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ch])
        return ishape

    def get_folded_input_shape(self):
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        assert ch % pe == 0, "PE must divide NumChannels"
        folds = int(ch / pe)
        folded_ishape = tuple(vecs + [folds, pe])
        return folded_ishape

    def get_normal_output_shape(self):
        return self.get_normal_input_shape()

    def get_folded_output_shape(self):
        return self.get_folded_input_shape()

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."

        oshape = self.get_normal_output_shape()
        values = np.zeros(oshape).astype(np.float32)
        split_input = np.concatenate((values, values), axis=0)

        split_in = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, oshape
        )

        model.graph.value_info.append(split_in)  # requires clean up
        model.set_initializer(split_in.name, split_input)

        shape_comp_node = helper.make_node(
            "Split",
            inputs=[split_in.name],
            outputs=[self.onnx_node.output[0], self.onnx_node.output[1]],
            axis=0,
        )

        return shape_comp_node

    def infer_node_datatype(self, model):
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)
        model.set_tensor_datatype(self.onnx_node.output[1], odt)

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

        return info_messages

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_instream_width(self):
        """Returns input stream width."""
        ibits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = pe * ibits
        return in_width

    def get_outstream_width(self):
        """Returns output stream width."""
        obits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        out_width = pe * obits
        return out_width

    def get_number_output_values(self):
        return 2 * np.prod(self.get_folded_output_shape()[1:-1])

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
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
            super().npy_to_dynamic_outputs(context, ["output0.npy", "output1.npy"])
            assert (
                context[node.output[0]].shape == folded_oshape
            ), "cppsim \
            did not produce expected ofolded utput shape"
            assert (
                context[node.output[1]].shape == folded_oshape
            ), "cppsim \
            did not produce expected ofolded utput shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
            context[node.output[1]] = context[node.output[1]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": [], "out1": []},
            }
            self.rtlsim_multi_io(sim, rtlsim_dict)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_shape = self.get_folded_output_shape()

            out_npy_path = "{}/output0.npy".format(code_gen_dir)
            rtlsim_output_to_npy(
                rtlsim_dict["outputs"]["out0"],
                out_npy_path,
                odt,
                out_shape,
                packed_bits,
                target_bits,
            )
            # load and reshape output 0
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            out_npy_path = "{}/output1.npy".format(code_gen_dir)
            rtlsim_output_to_npy(
                rtlsim_dict["outputs"]["out1"],
                out_npy_path,
                odt,
                out_shape,
                packed_bits,
                target_bits,
            )
            # load and reshape output 1
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[1]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output0 shape doesn't match expected shape."""
        assert (
            context[node.output[1]].shape == exp_oshape
        ), """Output1 shape doesn't match expected shape."""

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

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
            'hls::stream<ap_uint<{}>> out0 ("out0");'.format(self.get_outstream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out1 ("out1");'.format(self.get_outstream_width())
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """DuplicateStreams_Batch<{}, {}> (in0, out0, out1, 1);""".format(
                self.get_outstream_width(), self.get_number_output_values() // 2,
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
        npy_out = "%s/output0.npy" % code_gen_dir
        npy_out1 = "%s/output1.npy" % code_gen_dir
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
            )
        ]

        self.code_gen_dict["$DATAOUTSTREAM$"] += [
            'apintstream2npy<%s, %s, %d, %s>(out1, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out1,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0,
                hls::stream<ap_uint<{}>> &out0,
                hls::stream<ap_uint<{}>> &out1)""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
                self.get_outstream_width(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out1")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        intf_names["m_axis"] = ["out0_V_V", "out1_V_V"]
        return intf_names

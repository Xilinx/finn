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

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class DuplicateStreams_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib function of the same name."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 0),
            # how many duplicated output streams to create
            "NumOutputStreams": ("i", True, 0),
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

    def get_num_output_streams(self):
        return self.get_nodeattr("NumOutputStreams")

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
        num_out = self.get_num_output_streams()
        assert len(self.onnx_node.output) == num_out, "Unexpected number of outputs"

        oshape = self.get_normal_output_shape()
        ret = super().make_const_shape_op(oshape)
        ret.output[:] = self.onnx_node.output
        return ret

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
        for my_out in self.onnx_node.output:
            model.set_tensor_datatype(my_out, odt)

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
            self.get_nodeattr("NumOutputStreams")
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
        return self.get_num_output_streams() * np.prod(
            self.get_folded_output_shape()[1:-1]
        )

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def generate_params(self, model, path):
        n_outputs = self.get_num_output_streams()
        inp_streams = []
        commands = []
        o_stream_w = self.get_outstream_width()
        i_stream_w = self.get_instream_width()
        in_stream = "hls::stream<ap_uint<%d> > &in0" % (i_stream_w)
        inp_streams.append(in_stream)
        commands.append("ap_uint<%d> e = in0.read();" % i_stream_w)
        iters = self.get_number_output_values() // self.get_num_output_streams()
        for i in range(n_outputs):
            out_stream = "hls::stream<ap_uint<%d> > &out%d" % (o_stream_w, i)
            inp_streams.append(out_stream)
            cmd = "out%d.write(e);" % i
            commands.append(cmd)

        impl_hls_code = []
        impl_hls_code.append("void DuplicateStreamsCustom(")
        impl_hls_code.append(",".join(inp_streams))
        impl_hls_code.append(") {")
        impl_hls_code.append("for(unsigned int i = 0; i < %d; i++) {" % iters)
        impl_hls_code.append("#pragma HLS PIPELINE II=1")
        impl_hls_code.append("\n".join(commands))
        impl_hls_code.append("}")
        impl_hls_code.append("}")
        impl_hls_code = "\n".join(impl_hls_code)

        impl_filename = "{}/duplicate_impl.hpp".format(path)
        f_impl = open(impl_filename, "w")
        f_impl.write(impl_hls_code)
        f_impl.close()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
        folded_oshape = self.get_folded_output_shape()
        n_outputs = self.get_num_output_streams()

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
            super().npy_to_dynamic_outputs(
                context, ["output%d.npy" % i for i in range(n_outputs)]
            )
            for i in range(n_outputs):
                assert (
                    context[node.output[i]].shape == folded_oshape
                ), "cppsim \
                did not produce expected ofolded utput shape"
                context[node.output[i]] = context[node.output[i]].reshape(*exp_oshape)
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
                "outputs": {},
            }
            for i in range(n_outputs):
                rtlsim_dict["outputs"]["out%d" % i] = []
            self.rtlsim_multi_io(sim, rtlsim_dict)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_shape = self.get_folded_output_shape()
            for i in range(n_outputs):
                out_npy_path = "%s/output%d.npy" % (code_gen_dir, i)
                rtlsim_output_to_npy(
                    rtlsim_dict["outputs"]["out%d" % i],
                    out_npy_path,
                    odt,
                    out_shape,
                    packed_bits,
                    target_bits,
                )
                # load and reshape output 0
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[node.output[i]] = output

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
        self.code_gen_dict["$GLOBALS$"] = ['#include "duplicate_impl.hpp"']

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
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        for i in range(n_outputs):
            out_name = "out%d" % i
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<%d>> %s ("%s");'
                % (self.get_outstream_width(), out_name, out_name)
            )

    def docompute(self):
        n_outputs = self.get_num_output_streams()
        ostreams = ["out%d" % x for x in range(n_outputs)]
        dc = "DuplicateStreamsCustom(in0, %s);" % (",".join(ostreams))
        self.code_gen_dict["$DOCOMPUTE$"] = [dc]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        n_outputs = self.get_num_output_streams()
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
        outstrm_code = []

        for i in range(n_outputs):
            out_name = "out%d" % i
            npy_out = "%s/output%d.npy" % (code_gen_dir, i)
            outstrm_code.append(
                'apintstream2npy<%s, %s, %d, %s>(%s, %s, "%s");'
                % (
                    packed_hls_type,
                    elem_hls_type,
                    elem_bits,
                    npy_type,
                    out_name,
                    oshape_cpp_str,
                    npy_out,
                )
            )

        self.code_gen_dict["$DATAOUTSTREAM$"] = outstrm_code

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        n_outputs = self.get_num_output_streams()
        inp_streams = []
        o_stream_w = self.get_outstream_width()
        i_stream_w = self.get_instream_width()
        in_stream = "hls::stream<ap_uint<%d> > &in0" % (i_stream_w)
        inp_streams.append(in_stream)
        for i in range(n_outputs):
            out_stream = "hls::stream<ap_uint<%d> > &out%d" % (o_stream_w, i)
            inp_streams.append(out_stream)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}({})""".format(
                self.onnx_node.name,
                ",".join(inp_streams),
            )
        ]

    def pragmas(self):
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        for i in range(n_outputs):
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=out%d name=out%d_%s"
                % (i, i, self.hls_sname())
            )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        n_outputs = self.get_num_output_streams()
        sname = self.hls_sname()
        intf_names["m_axis"] = []
        for i in range(n_outputs):
            intf_names["m_axis"].append(
                ("out%d_%s" % (i, sname), self.get_outstream_width_padded())
            )
        return intf_names

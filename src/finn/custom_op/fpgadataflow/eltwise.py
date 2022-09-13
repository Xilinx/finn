# Copyright (c) 2022, Xilinx
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


class StreamingEltwise(HLSCustomOp):
    """Class that corresponds to finn-hlslib StreamingEltwise function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, ""),
            "PE": ("i", True, ""),
            # FINN DataTypes for inputs; output datatype inferred from input
            "inputDataType0": ("s", True, ""),
            "inputDataType1": ("s", True, ""),
            # type of EltwiseFunction for the operation
            "eltwiseOp": ("s", True, "", ["Add", "Sub", "AbsDiff"]),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_eltwise_op_lambda(self):
        eltwise_op = self.get_nodeattr("eltwiseOp")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        odt = self.get_output_datatype()
        tin0 = idt0.get_hls_datatype_str()
        tin1 = idt1.get_hls_datatype_str()
        tout = odt.get_hls_datatype_str()
        eltwise_ops = {
            # "Add": "[](auto a, auto b) { return  a + b; }",
            # "Sub": "[](auto a, auto b) { return  a - b; }",
            # "AbsDiff": "[](auto a, auto b) { return  a>b? a-b : b-a; }",
            "Add": f"add<{tin0}, {tin1}, {tout}>()",
            "Sub": f"sub<{tin0}, {tin1}, {tout}>()",
            "AbsDiff": f"absdiff<{tin0}, {tin1}, {tout}>()",
        }
        return eltwise_ops[eltwise_op]

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ich])
        return ishape

    def get_folded_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        assert ich % pe == 0, "PE must divide NumChannels"
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ich // pe, pe])
        return ishape

    def get_normal_output_shape(self):
        return self.get_normal_input_shape()

    def get_folded_output_shape(self):
        return self.get_folded_input_shape()

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input1 shape."
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[1]))
        assert ishape == exp_ishape, "Unexpected input2 shape."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt0 = model.get_tensor_datatype(node.input[0])
        if idt0 != self.get_input_datatype(0):
            warn_str = "inputDataType0 changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0)),
                str(idt0),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType0", idt0.name)
        idt1 = model.get_tensor_datatype(node.input[1])
        if idt1 != self.get_input_datatype(1):
            warn_str = "inputDataType1 changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(1)),
                str(idt1),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType1", idt1.name)
        # enforce output data type (calculated based on idt)
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
            self.get_nodeattr("inputDataType0")
            self.get_nodeattr("inputDataType1")
            self.get_nodeattr("eltwiseOp")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The required LabelSelect_Batch attributes do not exist."""
            )

        return info_messages

    def get_input_datatype(self, id=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType" + str(id))]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        op = self.get_nodeattr("eltwiseOp")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        assert idt0.signed() == idt1.signed(), (
            "%s: Inputs must have same signedness" % self.onnx_node.name
        )
        idt0_min, idt0_max = idt0.min(), idt0.max()
        idt1_min, idt1_max = idt1.min(), idt1.max()
        cands = [
            idt0_min - idt1_min,
            idt0_min - idt1_max,
            idt0_max - idt1_min,
            idt0_max - idt1_max,
        ]
        largest_magnitude = max(map(abs, cands))
        if op == "Add":
            if idt0.signed():
                return DataType.get_smallest_possible(idt0.min() + idt1.min())
            else:
                return DataType.get_smallest_possible(idt0.max() + idt1.max())
        elif op == "Sub":
            return DataType.get_smallest_possible(-largest_magnitude)
        elif op == "AbsDiff":
            return DataType.get_smallest_possible(largest_magnitude)
        else:
            raise Exception("%s: Unknown eltWiseOp = %s" % (self.onnx_node.name, op))

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        ibits = self.get_input_datatype(ind).bitwidth()
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
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

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
        assert (
            inp.shape == exp_ishape
        ), """Input0 shape doesn't match expected shape ."""
        export_idt0 = self.get_input_datatype(0)
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        # exact same thing for input1
        inp = context[node.input[1]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input1 shape doesn't match expected shape ."""
        export_idt1 = self.get_input_datatype(1)
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_1.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits0 = self.get_instream_width(0)
            nbits1 = self.get_instream_width(1)
            rtlsim_inp0 = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt0, nbits0
            )
            rtlsim_inp1 = npy_to_rtlsim_input(
                "{}/input_1.npy".format(code_gen_dir), export_idt1, nbits1
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp0, rtlsim_inp1)
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
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "eltwise.hpp"',
            '#include "interpret.hpp"',
        ]

        self.code_gen_dict["$GLOBALS$"].extend(
            [
                "template<typename TI1, typename TI2, typename TO>",
                "struct absdiff {",
                "TO operator()(TI1 const &a, TI2 const &b) const {",
                "#pragma HLS inline",
                "return  a>b? a-b : b-a;",
                "}",
                "};",
                "template<typename TI1, typename TI2, typename TO>",
                "struct sub {",
                "TO operator()(TI1 const &a, TI2 const &b) const {",
                "#pragma HLS inline",
                "return  a-b;",
                "}",
                "};",
                "template<typename TI1, typename TI2, typename TO>",
                "struct add {",
                "TO operator()(TI1 const &a, TI2 const &b) const {",
                "#pragma HLS inline",
                "return  a+b;",
                "}",
                "};",
            ]
        )

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        elem_bits_0 = idt0.bitwidth()
        elem_bits_1 = idt1.bitwidth()
        packed_bits_0 = self.get_instream_width(0)
        packed_hls_type_0 = "ap_uint<%d>" % packed_bits_0
        packed_bits_1 = self.get_instream_width(1)
        packed_hls_type_1 = "ap_uint<%d>" % packed_bits_1
        elem_hls_type_0 = idt0.get_hls_datatype_str()
        elem_hls_type_1 = idt1.get_hls_datatype_str()
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type_0, elem_hls_type_0, elem_bits_0, npy_type, npy_in)
        )
        npy_in = "%s/input_1.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in1);'
            % (packed_hls_type_1, elem_hls_type_1, elem_bits_1, npy_type, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in1 ("in1");'.format(self.get_instream_width(1))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        op = self.get_nodeattr("eltwiseOp")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        odt = self.get_output_datatype()
        elem_hls_type_0 = idt0.get_hls_datatype_str()
        elem_hls_type_1 = idt1.get_hls_datatype_str()
        out_hls_type = odt.get_hls_datatype_str()
        slice_in0 = "Slice<%s>" % elem_hls_type_0
        slice_in1 = "Slice<%s>" % elem_hls_type_1
        slice_out = "Slice<%s>" % out_hls_type
        eltwise_op_str = self.get_eltwise_op_lambda()
        "%sEltwiseFunction<%s, %s, %s>()" % (
            op,
            elem_hls_type_0,
            elem_hls_type_1,
            out_hls_type,
        )
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<{}, {}, {}, {}, {}, {}>(in0, in1, out, {});""".format(
                "StreamingEltwise",
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("PE"),
                self.get_number_output_values(),
                slice_in0,
                slice_in1,
                slice_out,
                eltwise_op_str,
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
            """void {}(hls::stream<ap_uint<{}>> &in0, hls::stream<ap_uint<{}>> &in1,
                hls::stream<ap_uint<{}>> &out)""".format(
                self.onnx_node.name,
                self.get_nodeattr("PE") * self.get_input_datatype(0).bitwidth(),
                self.get_nodeattr("PE") * self.get_input_datatype(1).bitwidth(),
                self.get_nodeattr("PE") * self.get_output_datatype().bitwidth(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=in1 name=in1_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        sname = self.hls_sname()
        swidth = self.get_instream_width_padded()
        intf_names["s_axis"] = [(x + "_" + sname, swidth) for x in ["in0", "in1"]]
        return intf_names

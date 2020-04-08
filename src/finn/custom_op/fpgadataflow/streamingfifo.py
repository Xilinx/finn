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
import os
import numpy as np
from shutil import copy

from pyverilator import PyVerilator
from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.core.datatype import DataType
from onnx import TensorProto, helper
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

from . import templates


class StreamingFIFO(HLSCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.strm_fifo_wrapper = templates.strm_fifo_wrapper

    def get_nodeattr_types(self):
        my_attrs = {
            # FIFO depth
            "depth": ("i", True, 0),
            # folded shape of input/output
            "folded_shape": ("ints", True, []),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())

        return my_attrs

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == tuple(exp_ishape), "Unexpect input shape for StreamingFIFO."
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
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        pass

    # overwrite not necessary functions
    def code_generation_ipgen(self, model, fpgapart, clk):
        # empty code gen dictionary for new entries
        self.code_gen_dict.clear()
        self.code_gen_dict["$TOPNAME$"] = ["top_{}".format(self.onnx_node.name)]
        self.code_gen_dict["$LAYER_NAME$"] = [
            "{}_{}".format(self.onnx_node.name, self.onnx_node.name)
        ]
        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width()
        if in_width % 8 != 0:
            in_width = math.floor(in_width / 8) + 8
        self.code_gen_dict["$IN_RANGE$"] = ["[{}:0]".format(in_width - 1)]
        self.code_gen_dict["$OUT_RANGE$"] = ["[{}:0]".format(in_width - 1)]
        self.code_gen_dict["$WIDTH$"] = [str(in_width)]
        self.code_gen_dict["$DEPTH$"] = [str(self.get_nodeattr("depth"))]

        template = self.strm_fifo_wrapper

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(
            os.path.join(code_gen_dir, "top_{}.v".format(self.onnx_node.name)), "w",
        )
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def ipgen_singlenode_code(self):
        pass

    def code_generation_npysim(self, model):
        pass

    def compile_singlenode_code(self):
        pass

    def get_normal_input_shape(self):
        depth = self.get_nodeattr("depth")
        assert (
            depth > 2
        ), """Depth is too low. Please set node attribute "depth" to a value
        between 2 and 256"""
        assert (
            depth < 256
        ), """Depth is too high. Please set node attribute "depth" to a value
        between 2 and 256"""
        folded_shape = self.get_nodeattr("folded_shape")
        inner_dim = folded_shape[-1]
        folding_factor = folded_shape[-2] * inner_dim
        normal_ishape = []
        for i in range(len(folded_shape) - 2):
            normal_ishape.append(folded_shape[i])
        normal_ishape.append(folding_factor)

        return normal_ishape

    def get_normal_output_shape(self):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self):
        return self.get_nodeattr("folded_shape")

    def get_folded_output_shape(self):
        return self.get_nodeattr("folded_shape")

    def get_instream_width(self):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        return folded_shape[-1] * dtype.bitwidth()

    def get_outstream_width(self):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        return folded_shape[-1] * dtype.bitwidth()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        inp = context[node.input[0]]
        exp_shape = self.get_normal_input_shape()

        if mode == "npysim":
            output = inp
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[node.output[0]] = output
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # create a npy file for the input of the node
            assert (
                str(inp.dtype) == "float32"
            ), """Input datatype is
                not float32 as expected."""
            expected_inp_shape = self.get_folded_input_shape()
            reshaped_input = inp.reshape(expected_inp_shape)
            if DataType[self.get_nodeattr("dataType")] == DataType.BIPOLAR:
                # store bipolar activations as binary
                reshaped_input = (reshaped_input + 1) / 2
                export_idt = DataType.BINARY
            else:
                export_idt = DataType[self.get_nodeattr("dataType")]
            # make copy before saving the array
            reshaped_input = reshaped_input.copy()
            np.save(
                os.path.join(code_gen_dir, "input_0.npy"), reshaped_input,
            )
            # copy Q_srl.v from finn-rtllib to code gen directory
            memstream_dir = "/workspace/finn/finn-rtllib/memstream/hdl/"
            Q_file = os.path.join(memstream_dir, "Q_srl.v")
            copy(Q_file, code_gen_dir)
            verilog_file = os.path.join(
                code_gen_dir, "top_{}.v".format(self.onnx_node.name)
            )
            if os.path.isfile(verilog_file):
                nbits = self.get_instream_width()
                inp = npy_to_rtlsim_input(
                    "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
                )
                sim = PyVerilator.build(verilog_file, verilog_path=[code_gen_dir],)
                super().reset_rtlsim(sim)
                super().toggle_clk(sim)
                output = self.rtlsim(sim, inp)
                odt = DataType[self.get_nodeattr("dataType")]
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
                    """Found no verilog files for this node,
                    did you run the codegen_ipgen transformation?"""
                )

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_number_input_values(self):
        folded_ishape = self.get_folded_input_shape()
        return np.prod(folded_ishape[:-1])

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

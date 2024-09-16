# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Performs transformations of input shapes to output shapes at both cppsim and rtlsim level
# Does padding and cropping if shapes mismatch using an intermediate inWidth+OutWidth buffer
# which is filled with zeroes. Only in hls-lib right now.


class StreamingDataWidthConverter(HWCustomOp):
    """Abstraction layer for HW implementation of StreamingDataWidthConverter"""

    def get_nodeattr_types(self):
        my_attrs = {
            # shapes of input/output tensors
            "in_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            # bit width of input and output streams
            "inWidth": ("i", True, 0),
            "outWidth": ("i", True, 0),
            "generalized_variant": ("i", True, 1),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("dataType")]

    def get_normal_input_shape(self, ind=0):
        ishape = self.get_nodeattr("in_shape")
        return ishape


    def get_num_in_words(self):
        shape = self.get_nodeattr("in_shape")
        out_els = self.get_nodeattr("inWidth") / self.get_output_datatype().bitwidth()
        num_words = int(shape[-1] // out_els)
        return num_words
    
    def get_num_words(self):
        shape = self.get_nodeattr("out_shape")
        out_els = self.get_nodeattr("outWidth") / self.get_input_datatype().bitwidth()
        num_words = int(shape[-1] // out_els)
        return num_words

    def get_normal_output_shape(self, ind=0):
        oshape = self.get_nodeattr("out_shape")
        return oshape

    def get_iowidth_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")

        return int(np.lcm(iwidth, owidth))

    def needs_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")

        # offset the resizing to get true values for DWC

        maxwidth = max(iwidth, owidth)
        minwidth = min(iwidth, owidth)
        return maxwidth % minwidth != 0

    def check_divisible_iowidths(self):
        pass

    def get_folded_input_shape(self, ind=0):
        self.check_divisible_iowidths()
        iwidth = self.get_nodeattr("inWidth")
        ishape = self.get_normal_input_shape()
        dummy_t = np.random.randn(*ishape)
        ibits = self.get_input_datatype().bitwidth()
        assert (
            iwidth % ibits == 0
        ), """DWC input width must be divisible by
        input element bitwidth"""
        ielems = int(iwidth // ibits)
        ichannels = ishape[-1]
        new_shape = []
        for i in ishape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ichannels // ielems))
        new_shape.append(ielems)

        dummy_t = dummy_t.reshape(new_shape)

        return dummy_t.shape

    def get_folded_output_shape(self, ind=0):
        self.check_divisible_iowidths()
        owidth = self.get_nodeattr("outWidth")

        oshape = self.get_normal_output_shape()

        obits = self.get_output_datatype().bitwidth()
        assert (
            owidth % obits == 0
        ), """DWC output width must be divisible by
        input element bitwidth"""
        oelems = int((owidth) // obits)
        ochannels = oshape[-1]
        new_shape = []
        for i in oshape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ochannels // oelems))
        new_shape.append(oelems)

        # reintroduce the resizing, this is the true final shape
        # we expect from the RTL
        # new_shape[-1] += resize

        return tuple(new_shape)

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_instream_width(self, ind=0):
        in_width = self.get_nodeattr("inWidth")
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.get_nodeattr("outWidth")
        return out_width

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()

        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == tuple(exp_ishape), "Unexpect input shape for StreamingDWC."
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
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingDWC needs 1 data input""")

        return info_messages

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_shape = self.get_normal_input_shape()
        out_shape = self.get_normal_output_shape()
        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == tuple(in_shape), "Input shape does not match expected shape."

        output = np.zeros((out_shape), dtype=np.float32)
        if out_shape[-1] > in_shape[-1]:
            output[..., : in_shape[-1]] = inp[..., : in_shape[-1]]
        else:
            output[..., : out_shape[-1]] = inp[..., : out_shape[-1]]

        output = np.asarray([output], dtype=np.float32).reshape(*out_shape)
        context[node.output[0]] = output

    
    def get_exp_cycles(self):

        out_shape = self.get_nodeattr("out_shape")
        out_width = self.get_nodeattr("outWidth")
        out_els = out_width / self.get_input_datatype().bitwidth()
        num_out_words = int(np.prod(self.get_folded_output_shape()[-2:-1]))

        in_shape = self.get_nodeattr("in_shape")
        in_width = self.get_nodeattr("inWidth") 
        in_els = in_width / self.get_input_datatype().bitwidth()
        num_in_words = int(np.prod(self.get_folded_input_shape()[-2:-1]))

        numReps = int(np.prod(self.get_folded_input_shape()[:2]))

        ratio = max(in_width,out_width) / min(in_width,out_width)
        words = max(num_in_words,num_out_words)
        min_words = min(num_in_words,num_out_words)
        
        exp_cycles = words + min_words
    
        return int(exp_cycles)
    
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
from finn.util.basic import Characteristic_Node

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter(HWCustomOp):
    """Abstraction layer for HW implementation of StreamingDataWidthConverter"""

    def get_nodeattr_types(self):
        my_attrs = {
            # shape of input/output tensors
            "shape": ("ints", True, []),
            # bit width of input and output streams
            "inWidth": ("i", True, 0),
            "outWidth": ("i", True, 0),
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
        ishape = self.get_nodeattr("shape")
        return ishape

    def get_normal_output_shape(self, ind=0):
        oshape = self.get_nodeattr("shape")
        return oshape

    def get_iowidth_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")
        return int(np.lcm(iwidth, owidth))

    def needs_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")
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
        dummy_t = np.random.randn(*oshape)
        obits = self.get_output_datatype().bitwidth()
        assert (
            owidth % obits == 0
        ), """DWC output width must be divisible by
        input element bitwidth"""
        oelems = int(owidth // obits)
        ochannels = oshape[-1]
        new_shape = []
        for i in oshape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ochannels // oelems))
        new_shape.append(oelems)
        dummy_t = dummy_t.reshape(new_shape)

        return dummy_t.shape

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
        exp_shape = self.get_normal_input_shape()
        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == tuple(exp_shape), "Input shape does not match expected shape."

        output = inp
        output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
        context[node.output[0]] = output

    def get_exp_cycles(self):
        return np.prod(self.get_folded_input_shape()) + np.prod(self.get_folded_output_shape())

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        inw = self.get_instream_width()
        outw = self.get_outstream_width()

        minw = min(inw, outw)
        maxw = max(inw, outw)

        # sometimes widths aren't directly divisible
        # this requires going up from input width to least common multiple
        # then down to output width
        intw = abs(maxw * minw) // math.gcd(maxw, minw)

        # we assume a shift-based implementation
        # even if we don't use LUTs explicitly, we make some unavailable
        # to other logic because they're tied into the DWC control sets

        cnt_luts = 0
        cset_luts = 0

        if inw != intw:
            cnt_luts += abs(math.ceil(math.log(inw / intw, 2)))
            cset_luts += intw
        if intw != outw:
            cnt_luts += abs(math.ceil(math.log(intw / outw, 2)))
            cset_luts += outw

        return int(cnt_luts + cset_luts)

    def prepare_kwargs_for_characteristic_fx(self):
        # function for the old DWC version
        # newer version will need a separate, much more
        # complicated function

        numInWords = int(np.prod(self.get_folded_input_shape()[-2:-1]))
        numOutWords = int(np.prod(self.get_folded_output_shape()[-2:-1]))
        numReps = int(np.prod(self.get_folded_input_shape()[:1]))

        # inWidth = self.get_nodeattr("inWidth")
        # outWidth = self.get_nodeattr("outWidth")

        read_inputs = Characteristic_Node("read all words", [(numInWords, [1, 0])], True)

        write_outputs = Characteristic_Node("write all words", [(numOutWords, [0, 1])], True)

        up_convert_word = Characteristic_Node(
            "up convert all words in a single transaction",
            [(1, read_inputs), (1, write_outputs)],
            False,
        )

        down_convert_word = Characteristic_Node(
            "down convert all words in a single transaction",
            [(1, read_inputs), (1, write_outputs)],
            False,
        )

        if numInWords > numOutWords:
            reps = Characteristic_Node(
                "compute a set of DWCs with up conversion", [(numReps, up_convert_word)], False
            )
        else:
            reps = Characteristic_Node(
                "compute a set of DWCs with down conversion", [(numReps, down_convert_word)], False
            )

        return reps

    # def prepare_kwargs_for_characteristic_fx_old(self):
    #     numInWords = int(np.prod(self.get_folded_input_shape()[-2:-1]))
    #     numOutWords = int(np.prod(self.get_folded_output_shape()[-2:-1]))
    #     numReps = int(np.prod(self.get_folded_input_shape()[:1]))

    #     inWidth = self.get_nodeattr("inWidth")
    #     outWidth = self.get_nodeattr("outWidth")

    #     kwargs = (numInWords, numOutWords, inWidth, outWidth, numReps)

    #     # assert True==False
    #     return kwargs

    # def characteristic_fx_input(self, txns, cycles, counter, kwargs):
    #     (numInWords, numOutWords, inWidth, outWidth, numReps) = kwargs

    #     # HYPER PARAMETERS WHICH MAY CHANGE OVER TIME
    #     windup_clocks_up_convert_input = 4

    #     windup_clocks_down_convert_input = 3

    #     windup_clocks_down_convert_output = 4
    #     windup_clocks_equal_convert_output = 3

    #     if numInWords < windup_clocks_up_convert_input:
    #         windup_clocks_up_convert_input = numInWords

    #     if numInWords < windup_clocks_down_convert_input:
    #         windup_clocks_down_convert_input = numInWords

    #     if numOutWords < windup_clocks_down_convert_output:
    #         windup_clocks_down_convert_output = numOutWords

    #     if numOutWords < windup_clocks_equal_convert_output:
    #         windup_clocks_equal_convert_output = numOutWords

    #     # first input period
    #     tracker = 0
    #     maximum = numReps * numInWords

    #     if numReps > 1:
    #         # loop windup
    #         for i in range(2):
    #             txns.append(counter)
    #             counter += 1
    #             cycles += 1
    #             tracker += 1

    #     for j in range(0, numReps):
    #         for i in range(0, numInWords):
    #             if tracker < maximum:
    #                 txns.append(counter)
    #                 counter += 1
    #                 cycles += 1
    #                 tracker += 1
    #         for i in range(0, 1):
    #             txns.append(counter)
    #             cycles += 1

    #     return txns, cycles, counter

    # def characteristic_fx_output(self, txns, cycles, counter, kwargs):
    #     (numInWords, numOutWords, inWidth, outWidth, numReps) = kwargs

    #     # HYPER PARAMETERS WHICH MAY CHANGE
    #     windup_clocks_up_convert_input = 3
    #     windup_clocks_down_convert_input = 2

    #     windup_clocks_down_convert_output = 3
    #     windup_clocks_equal_convert_output = 2

    #     if numInWords < windup_clocks_up_convert_input:
    #         windup_clocks_up_convert_input = numInWords

    #     if numInWords < windup_clocks_down_convert_input:
    #         windup_clocks_down_convert_input = numInWords

    #     if numOutWords < windup_clocks_down_convert_output:
    #         windup_clocks_down_convert_output = numOutWords

    #     if numOutWords < windup_clocks_equal_convert_output:
    #         windup_clocks_equal_convert_output = numOutWords

    #     # calculation to adjust for padding or cropping adding latency

    #     if outWidth > inWidth:
    #         higher = outWidth
    #         lower = inWidth
    #     else:
    #         higher = inWidth
    #         lower = outWidth

    #     if higher % lower != 0:
    #         if numInWords * inWidth > numOutWords * outWidth:
    #             pad = False
    #         else:
    #             pad = True

    #     else:
    #         pad = False

    #         # windup period
    #         if inWidth == outWidth:
    #             clock = windup_clocks_equal_convert_output
    #         else:
    #             clock = windup_clocks_up_convert_input
    #         for i in range(0, clock):
    #             txns.append(counter)
    #             cycles += 1
    #         # padding +=1

    #         # first input period

    #         remainder = 0

    #         for k in range(numReps):
    #             # windup
    #             txns.append(counter)
    #             cycles += 1

    #             for i in range(0, numOutWords):
    #                 for j in range(0, int(np.floor(outWidth / inWidth))):
    #                     if j != 0:
    #                         txns.append(counter)
    #                         cycles += 1
    #                     remainder += inWidth
    #                 #  padding +=1

    #                 if pad and remainder < outWidth:
    #                     print(remainder)
    #                     txns.append(counter)
    #                     remainder += inWidth
    #                     cycles += 1

    #                 txns.append(counter)
    #                 cycles += 1

    #                 counter += 1
    #                 remainder -= outWidth

    #     return txns, cycles, counter

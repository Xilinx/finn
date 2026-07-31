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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import Characteristic_Node


class GlobalAccPool(HWCustomOp):
    """Abstraction layer for HW implementation of GlobalAccPool"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

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

    def get_exp_cycles(self):
        # Channels/PE * batch size * idim * idim + Channels/PE
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        folds = int(ch / pe)
        return int(np.prod(self.get_folded_input_shape()[:-1]) + folds)

    def get_tree_model(self):
        """Read a whole frame, pause, then emit the accumulators.

        The HLS pool is two loops that do not overlap. The first reads the
        frame's ``N`` folded words back to back at one per cycle, accumulating
        each into the bank entry for its channel tile and writing nothing. The
        second walks the bank and writes its ``NumChannels / PE`` words, again
        one per cycle. So a frame is a wind-up, the read run, a gap while the
        last accumulation settles and the write loop is entered, the write run,
        and a tail before the next frame's first read.

        The wind-up and the tail are the fitted part, and step with the width of
        the accumulator word: more than eight accumulators side by side take two
        more cycles to reach and give one back at the end. The gap does not
        move.

        ``NumChannels / PE == 1`` is a different schedule -- with one word to
        emit there is no write loop for HLS to enter and the write lands in the
        read loop's epilogue -- and is not modelled here.

        The read run assumes the accumulate loop pipelines at II=1, as
        ``get_exp_cycles`` does. A design that schedules slower stretches every
        read by that factor and takes proportionally longer.

        Valid for PE 1..64 and NumChannels / PE 2..64.
        """
        pe = self.get_nodeattr("PE")
        n_in = int(np.prod(self.get_folded_input_shape()[:-1]))
        n_out = int(np.prod(self.get_folded_output_shape()[:-1]))
        if n_in < 1 or n_out < 2 or pe < 1:
            return None
        wind_up = 3 if pe <= 8 else 5
        tail = 1 if pe <= 8 else 0
        gap = 5
        frame = Characteristic_Node(
            "accumulate then emit",
            [(wind_up, [0, 0]), (n_in, [1, 0]), (gap, [0, 0]), (n_out, [0, 1]), (tail, [0, 0])],
            True,
        )
        return Characteristic_Node("GlobalAccPool frame", [(1, frame)], False)

    def execute_node(self, context, graph):
        # simulate behavior with Python functionality
        node = self.onnx_node
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        result = np.apply_over_axes(np.sum, inp_values, [1, 2])
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

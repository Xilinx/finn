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
import onnxruntime as rt
import warnings
from math import gcd
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import flat_characteristic_leaf

# One output word leaves every this many cycles. The upsampling loop itself is
# pipelined at II=1, but it is invoked once per output word from a top level that
# is not pipelined, and that call costs the same handful of cycles whatever the
# part or the clock target.
_UPSAMPLE_II = 5
# Where in the II slot the read of the replacement input word falls, relative to
# the output word that released it. Measured, not derived: upsample.hpp gates the
# read on the write pointer and the output on WP_DELAY, and the two together place
# the read here.
_UPSAMPLE_READ_LAG = 4
# Cycles the pipeline takes to fill before the first output word appears.
_UPSAMPLE_WIND_UP = 19
# Frames DeriveTokenAccessVectors simulates per characterization run
# (periods_to_simulate); one period is that share of the run, so it carries the
# same share of the one-off pipeline fill.
_CHRC_PERIODS = 5


class _Stepper:
    """One dimension's input index generator, as the upsampling loop nest runs it.

    A dimension that is not upsampled steps its input index every output step; one
    upsampled from a single input element holds that element for the whole output
    extent; any other ratio walks the input index with Bresenham's line algorithm.
    ``replays()`` answers whether the current input element is needed again, which
    is what decides when its buffer slot may be refilled.
    """

    def __init__(self, size_in, size_out):
        self.size_out = size_out
        self.pos = 0
        if size_in == size_out:
            self.kind = "copy"
        elif size_in == 1:
            self.kind = "hold"
        else:
            self.kind = "bresenham"
            common = gcd(size_in, size_out)
            self.num, self.den = size_in // common, size_out // common
            self.err = (3 * self.num - 2 * self.den) >> 1

    def replays(self):
        if self.kind == "copy":
            return False
        if self.kind == "hold":
            return self.pos != self.size_out - 1
        return self.err < 0

    def step(self):
        """Advance one output position; returns True when the dimension wraps."""
        if self.kind == "bresenham":
            self.err += self.num if self.err < 0 else self.num - self.den
        self.pos += 1
        wrapped = self.pos == self.size_out
        if wrapped:
            self.pos = 0
        return wrapped


class UpsampleNearestNeighbour(HWCustomOp):
    """Abstraction layer for HW implementation of UpsampleNearestNeighbour."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "SIMD": ("i", True, 0),
            # Height, width of the output feature map
            "HO": ("i", True, 0),
            "WO": ("i", True, 0),
            # Height, width of the input feature map
            "HI": ("i", True, 0),
            "WI": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "batchSize": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def release_mask(self, dims):
        """Which output words release the input word they were built from.

        ``dims`` lists ``(input extent, output extent)`` from the slowest- to the
        fastest-changing dimension of the output raster. An input word may be
        overwritten once no dimension will replay it, so the mask is one exactly at the
        output positions where every dimension is done with its current input index. It
        holds ``prod(input extents)`` ones, one per input word of a frame.
        """
        steppers = [_Stepper(size_in, size_out) for size_in, size_out in dims]
        num_out = int(np.prod([size_out for _, size_out in dims]))
        mask = np.zeros(num_out, dtype=np.int8)
        for pos in range(num_out):
            mask[pos] = 0 if any(s.replays() for s in steppers) else 1
            wrapped = True
            for stepper in reversed(steppers):
                if not wrapped:
                    break
                wrapped = stepper.step()
        return mask

    def get_tree_model(self):
        """Writes the output raster at a fixed rate and refills the buffer behind it.

        The layer holds an input frame in a buffer and walks the output raster over
        it, emitting one folded word every five cycles -- the loop nest is pipelined
        at one word per iteration, but each iteration is one call from a top level
        that is not pipelined. Nothing about that rate depends on the part or the
        clock, so the period is five times the output frame ``get_exp_cycles``
        counts.

        An input word is read once its buffer slot comes free, which happens on the
        output word that used it for the last time: the last output row that
        replicates its row and, within that row, the last output column that
        replicates its column. That gives exactly one read per input word of a
        frame, placed a fixed few cycles after the output word that released it.
        Reads therefore inherit the bunching of the replication pattern, which is
        what makes them worth stating rather than spreading evenly.

        The buffer is filled ahead once out of reset, and that head start is not
        modelled -- it shifts which input word a read carries, not when reads
        happen. The cycles the pipeline takes to fill before the first output word
        are modelled: a period is one frame of a ``_CHRC_PERIODS``-frame run, so
        that share of the one-off fill lands in it, and the same share is added
        here.

        Valid for the HLS implementation, for any output extent at least as large
        as its input extent, any SIMD dividing NumChannels, and any batch size.
        """
        if not self.onnx_node.op_type.endswith("_hls"):
            return None
        simd = self.get_nodeattr("SIMD")
        num_channels = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("batchSize")
        hi, wi = self.get_nodeattr("HI"), self.get_nodeattr("WI")
        ho, wo = self.get_nodeattr("HO"), self.get_nodeattr("WO")
        if simd < 1 or num_channels % simd != 0:
            return None
        folds = num_channels // simd
        if min(hi, wi, ho, wo, folds, batch) < 1 or ho < hi or wo < wi:
            return None
        # the loop nest walks (row, column, channel fold), and the channel folds of
        # one pixel are neither replicated nor reordered
        released = np.tile(self.release_mask([(hi, ho), (wi, wo), (folds, folds)]), batch)
        num_out = released.size
        period = _UPSAMPLE_II * num_out + _UPSAMPLE_WIND_UP // _CHRC_PERIODS
        wr = np.zeros(period, dtype=np.int8)
        wr[_UPSAMPLE_II * np.arange(num_out)] = 1
        rd = np.zeros(period, dtype=np.int8)
        rd[_UPSAMPLE_II * np.flatnonzero(released) + _UPSAMPLE_READ_LAG] = 1
        return flat_characteristic_leaf(rd, wr, "Upsample raster")

    def get_normal_input_shape(self, ind=0):
        batch = self.get_nodeattr("batchSize")
        HI = self.get_nodeattr("HI")
        WI = self.get_nodeattr("WI")
        num_ch = self.get_nodeattr("NumChannels")
        ishape = (batch, HI, WI, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        batch = self.get_nodeattr("batchSize")
        HO = self.get_nodeattr("HO")
        WO = self.get_nodeattr("WO")
        num_ch = self.get_nodeattr("NumChannels")
        oshape = (batch, HO, WO, num_ch)
        return oshape

    def get_folded_input_shape(self, ind=0):
        spatial_shape = list(self.get_normal_input_shape())[:-1]
        simd = self.get_nodeattr("SIMD")
        folds = self.get_nodeattr("NumChannels") // simd
        return tuple(spatial_shape + [folds, simd])

    def get_folded_output_shape(self, ind=0):
        spatial_shape = list(self.get_normal_output_shape())[:-1]
        simd = self.get_nodeattr("SIMD")
        folds = self.get_nodeattr("NumChannels") // simd
        return tuple(spatial_shape + [folds, simd])

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

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def execute_node(self, context, graph):
        # create a standard resize node to help calculate the result
        node = self.onnx_node
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        HO = self.get_nodeattr("HO")
        WO = self.get_nodeattr("WO")
        HI = self.get_nodeattr("HI")
        WI = self.get_nodeattr("WI")
        scales_val = [1, int(round(HO / HI)), int(round(WO / WI)), 1]
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_resize = helper.make_node(
            "Resize",
            inputs=[node.input[0], "", "scales"],
            outputs=[node.output[0]],
            mode="nearest",
        )
        graph_resize = helper.make_graph(
            nodes=[node_resize],
            name="single-resize-exec",
            inputs=[inp, scales],
            outputs=[outp],
        )

        opset_imports = [helper.make_opsetid("", 13)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_resize = qonnx_make_model(graph_resize, **onnx_kwargs)
        idict = {node.input[0]: inp_values, "scales": scales_val}
        sess = rt.InferenceSession(model_resize.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

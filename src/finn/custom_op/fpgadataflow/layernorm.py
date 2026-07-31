###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

import numpy as np
import torch
import torch.nn.functional as F
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import passthrough_characteristic


class LayerNorm(HWCustomOp):
    """Abstraction layer for HW implementation of the LayerNorm layer."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "SIMD": ("i", True, 0),
                "ifm_dim": ("ints", True, []),
                "epsilon": ("f", True, 1e-5),
                # FINN DataTypes for inputs, outputs
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", True, ""),
            }
        )
        return my_attrs

    def execute_node(self, context, graph):
        node = self.onnx_node
        # Get tensor values
        in_values = context[node.input[0]]
        out_values = context[node.output[0]]
        # Get any shape info that needs reuse
        ishape = in_values.shape
        oshape = out_values.shape
        # Functionally verify with PyTorch implementation, since weight & bias are removed
        in_act = torch.from_numpy(in_values)
        out_act = F.layer_norm(in_act, [ishape[-1]], eps=self.get_nodeattr("epsilon"))
        context[node.output[0]] = np.asarray(out_act, dtype=np.float32).reshape(oshape)

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_exp_cycles(self):
        """One beat per cycle, so a frame costs its number of beats."""
        return int(np.prod(self.get_folded_input_shape()[:-1]))

    def get_tree_model(self):
        """One beat in and one out per cycle.

        The layer streams: the statistics stage runs a beat behind the data and
        the normalization follows it, so in steady state a beat is taken and a
        beat is emitted every cycle and there is nothing else to describe.
        Consecutive frames run into each other, so the two rows are one solid
        run and the period carries no wind-up.

        The schedule assumes the pipeline is scheduled at II=1, as
        ``get_exp_cycles`` does. The RTL backend is II=1 by construction. The
        HLS backend with a floating-point input is not: both of its statistics
        stages accumulate, and in floating point that accumulation is a
        loop-carried dependency the tool does not close in one cycle, so every
        beat is stretched by the same factor and the node runs several times
        longer than this says.

        TODO: that case returns None for now, which costs one rtlsim per node to
        measure the interval the tool actually reached. The shape of the
        schedule is right, only its rate is not, so a model that knew the II
        would cover it -- but the II is a synthesis outcome and nothing at this
        level can predict it. Reinstate the model here if the II ever becomes
        available, e.g. read back from the synthesis report.

        The HLS backend has one regime this shape does not cover. Its second
        statistics stage feeds a reciprocal square root whose result the
        normalization waits on, and with an integer input -- where nothing else
        holds the pipeline back -- a vector shorter than that path's latency
        cannot hide it. The stream then becomes a burst of ``N / SIMD`` beats
        per vector followed by a wait, which is not a stretch of this schedule
        but a different one, so that case returns None.

        Valid for SIMD 1..8, and on the HLS backend for an integer input with
        ``N / SIMD`` of 40 upwards.
        """
        # the latency of the reciprocal square root the second statistics
        # stage feeds, in cycles a vector has to be at least as long as
        RSQRT_PATH = 40
        dim = int(np.prod(self.get_folded_input_shape()[:-1]))
        simd = self.get_nodeattr("SIMD")
        if dim < 1 or simd < 1:
            return None
        if self.onnx_node.op_type.endswith("_hls"):
            if not self.get_input_datatype().is_integer():
                return None  # see the TODO above: II > 1, and by how much is
                # a synthesis outcome
            if self.get_normal_input_shape()[-1] // simd < RSQRT_PATH:
                return None
        return passthrough_characteristic(dim, "normalize a beat")

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

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
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        in_width = i_bits * self.get_nodeattr("SIMD")
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("SIMD")
        return out_width

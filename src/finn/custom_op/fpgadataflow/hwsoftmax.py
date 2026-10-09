############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import warnings
from qonnx.core.datatype import DataType
from scipy.special import softmax

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import Characteristic_Node


class HWSoftmax(HWCustomOp):
    """Abstraction layer for HW implementation of SoftMax layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "ifm_dim": ("ints", True, []),
            "SIMD": ("i", False, 1),
            # FINN DataTypes for inputs, weights, outputs
            "input_data_type": ("s", True, ""),
            "NumChannels": ("i", False, 128),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_tree_model(self):
        """One beat in and one out per cycle, with a short stall between vectors.

        The HLS design is three dataflow stages -- maximum, exponentiate and
        accumulate, divide -- with a FIFO between each. Inside a vector the beats
        stream straight through: one beat is read and one written every cycle.
        The stall is at the vector boundary, where the middle stage spends a
        cycle taking the next vector's maximum before it will touch data again;
        from SIMD 4 upwards it spends one more, the reduction tree over the SIMD
        exponentials having to settle before the vector's sum can be handed on.
        So a vector costs ``N / SIMD + stall`` cycles and a frame is one vector
        per position of the feature map.

        The period carries no pipeline fill. The stages buffer whole vectors
        between them and consecutive frames run into each other, so a frame
        boundary is just another inter-vector stall: what the schedule has to
        get right is the rate, and the recorded period is longer than a frame by
        the node's share of the fill rather than by anything periodic.

        Within a vector the stages are assumed to run at II=1, as
        ``get_exp_cycles`` does. A design that schedules slower stretches every
        beat by that factor and takes proportionally longer.

        The RTL design has no per-vector stall and is not modelled here; it falls
        back to rtlsim.

        Valid for SIMD 1..16 and N / SIMD 4..96.
        """
        if not self.onnx_node.op_type.endswith("_hls"):
            return None
        n_beats = int(np.prod(self.get_folded_input_shape()[:-1]))
        n = self.get_normal_input_shape()[-1]
        simd = self.get_nodeattr("SIMD")
        beats_per_vec = max(1, n // simd)
        if n_beats < 1 or n_beats % beats_per_vec:
            return None
        stall = 2 if simd < 4 else 3
        stream = Characteristic_Node("stream a vector", [(beats_per_vec, [1, 1])], True)
        gap = Characteristic_Node("wait on the next maximum and sum", [(stall, [0, 0])], True)
        vector = Characteristic_Node("softmax one vector", [(1, stream), (1, gap)], False)
        return Characteristic_Node("softmax frame", [(n_beats // beats_per_vec, vector)], False)

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        output_data = softmax(input_data, axis=-1)
        context[node.output[0]] = output_data

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        data_type = DataType[self.get_nodeattr("input_data_type")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert data_type.allowed(0), "DataType must support zero"
        return data_type

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "input_data_type changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("input_data_type", idt.name)

        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType["FLOAT32"]

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

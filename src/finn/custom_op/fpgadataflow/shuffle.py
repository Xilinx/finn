############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import warnings
from onnx import helper
from operator import itemgetter
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Shuffle(HWCustomOp):
    """Abstraction layer for Shuffle (rearrange and transpose) layers.
    This operator is later transformed into InnerShuffle and OuterShuffle operations."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """
        The attributes for the Shuffle node capture the
        optional reshapes either side of the transpose.
        Below is a diagram indicating what tensors the
        attribute names are referring to.

              │ in_shape
              │
              │
        ┌─────▼──────┐
        │            │
        │ Reshape    │
        │            │
        └─────┬──────┘
              │
              │ transpose_in_shape
        ┌─────▼──────┐
        │            │
        │  Transpose │
        │            │
        └─────┬──────┘
              │  transpose_out_shape
        ┌─────▼──────┐
        │            │
        │  Reshape   │
        │            │
        └─────┬──────┘
              │
              │  out_shape
              ▼
        """
        my_attrs = {
            "data_type": ("s", True, ""),
            "transpose_in_shape": ("ints", True, []),
            "in_shape": ("ints", True, []),
            "transpose_out_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "perm": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "NumChannels": ("i", False, 128),
            "original_node_name": ("s", False, ""),  # Track original shuffle name for SIMD config
            "original_simd": ("i", False, 1),  # Track original shuffle SIMD for config export
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_nodeattr("out_shape")

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.get_nodeattr("transpose_in_shape"))
        transposed = np.transpose(input_reshaped, axes=self.get_nodeattr("perm"))
        output_reshaped = transposed.reshape(self.get_nodeattr("out_shape"))
        context[node.output[0]] = output_reshaped

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = (
                f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "SIMD must divide into the innermost output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into the innermost input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_exp_cycles(self):
        """Estimate cycles by decomposing into Inner/OuterShuffle stages.

        Decomposes the transpose into a sequence of hardware-constrained
        operations (inner_shuffle / outer_shuffle), creates temporary nodes
        for each stage, and returns the MAX of their cycle estimates
        (stages are pipelined, so throughput is limited by the slowest).
        """
        from finn.transformation.fpgadataflow.transpose_decomposition import (
            _is_inner_shuffle,
            decompose_transpose_with_constraints,
            shuffle_perfect_loopnest_coeffs,
        )

        transpose_in_shape = list(self.get_nodeattr("transpose_in_shape"))
        perm = list(self.get_nodeattr("perm"))
        simd = self.get_nodeattr("SIMD")
        data_type = self.get_nodeattr("data_type")

        P_list, operation_types = decompose_transpose_with_constraints(
            perm, transpose_in_shape, simd
        )

        if len(P_list) == 0:
            return 0

        stage_cycles = []
        current_shape = list(transpose_in_shape)

        for step_idx, (P, op_type) in enumerate(zip(P_list, operation_types)):
            out_shape = list(itemgetter(*P)(current_shape))

            if _is_inner_shuffle(P, current_shape):
                # InnerShuffle: in_shape = current_shape
                tmp_node = helper.make_node(
                    "InnerShuffle",
                    ["tmp_in"],
                    ["tmp_out"],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=current_shape,
                    data_type=data_type,
                    SIMD=simd,
                    name=f"tmp_inner_{step_idx}",
                )
            else:
                # OuterShuffle
                loop_coeffs = shuffle_perfect_loopnest_coeffs(shape=current_shape, perm=P)
                tmp_node = helper.make_node(
                    "OuterShuffle",
                    ["tmp_in"],
                    ["tmp_out"],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=current_shape,
                    transpose_in_shape=current_shape,
                    perm=P,
                    out_shape=out_shape,
                    transpose_out_shape=out_shape,
                    data_type=data_type,
                    loop_coeffs=loop_coeffs,
                    SIMD=simd,
                    NumChannels=current_shape[-1],
                    name=f"tmp_outer_{step_idx}",
                )

            inst = getCustomOp(tmp_node)
            stage_cycles.append(inst.get_exp_cycles())
            current_shape = out_shape

        return max(stage_cycles)

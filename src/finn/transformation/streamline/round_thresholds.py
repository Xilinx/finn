# Copyright (c) 2020-2022, Xilinx
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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
from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes

def widened_datatype(dtype: DataType) -> DataType:
    """The type of dtype extended by one value above its maximum (max + step)."""
    if dtype.is_integer():
        max_val = dtype.max() + 1
        if not dtype.signed():
            return DataType.get_smallest_possible(max_val)
        return DataType.get_smallest_possible(-(max_val) - 1)
    # smallest underlying integer representation of max + step
    max_val = dtype.max() / dtype.scale_factor() + 1
    tdt_int = DataType.get_smallest_possible(-max_val - 1)
    return DataType[f"FIXED<{tdt_int.bitwidth()},{tdt_int.bitwidth() - dtype.frac_bits()}>"]


def drop_constant_thresholds(
    thresholds: np.ndarray, always: float, never: float, name: str
) -> tuple[np.ndarray, int]:
    """Sorts the thresholds of each channel and removes the threshold columns
    that are constant for all channels: those never met (>= never) and those
    always met (<= always). Returns the remaining thresholds and the number of
    removed always-met columns, which the caller adds to the output bias.

    Counting thresholds is invariant under their order, so sorting changes no
    result. If every threshold is always or never met, the output of each
    channel does not depend on the input, which a thresholding node cannot
    implement: raises a ValueError.
    """
    if np.all((thresholds <= always) | (thresholds >= never)):
        # every channel's output is a constant, even if columns are mixed
        raise ValueError(
            f"{name}: every threshold is always or never met over the input range, "
            "so the output is constant"
        )
    thresholds = np.sort(thresholds, axis=-1)
    num = thresholds.shape[-1]
    num_always = int(np.all(thresholds <= always, axis=0).sum())
    num_never = int(np.all(thresholds >= never, axis=0).sum())
    return thresholds[..., num_always : num - num_never], num_always


def shift_output_bias(node: NodeProto, num_always: int) -> None:
    """Adds the count of removed always-met thresholds to the output bias."""
    inst = getCustomOp(node)
    if node.op_type == "MultiThreshold":
        bias = inst.get_nodeattr("out_bias") + num_always * inst.get_nodeattr("out_scale")
        inst.set_nodeattr("out_bias", float(bias))
    else:
        inst.set_nodeattr("ActVal", inst.get_nodeattr("ActVal") + num_always)


class RoundAndClipThresholds(Transformation):
    """For MultiThreshold, Thresholding, MVAU, and VVAU nodes operating on integer inp/accumulators,
    round up (ceil) threshold values to the nearest integer and clip to valid range.
    Type-casts thresholds (back) to the float32 container type (this is separate from the
    quantization annotation). Runs InferDataTypes() afterward to propagate any changes to the
    quantization data types."""

    def apply(self, model: ModelWrapper):  # noqa
        graph = model.graph
        graph_modified = False
        for index, node in enumerate(graph.node):
            op_type = node.op_type
            if op_type == "MultiThreshold" or op_type.startswith("Thresholding"):
                thresholds = model.get_initializer(node.input[1])
                if thresholds is None:
                    continue
                dtype = model.get_tensor_datatype(node.input[0])
                # This transformation only applies to thresholding operations
                # operating on integer inputs
                if not (dtype.is_integer() or dtype.is_fixed_point()):
                    continue
                # Round thresholds up to the nearest value of the input type and
                # clip them to [min, max + step]: a threshold at or below min is
                # always met, one above max never.
                #   Note: This might promote the thresholds to float64 and
                #   introduce extra inaccuracies due to large integers not being
                #   exactly representable in floating-point representation.
                #   See for example: np.ceil(np.float32(16777217)) == 16777216
                step = 1 if dtype.is_integer() else dtype.scale_factor()
                never = dtype.max() + step
                new_thresholds = np.clip(np.ceil(thresholds / step) * step, dtype.min(), never)
                runtime = op_type.startswith("Thresholding") and (
                    getCustomOp(node).get_nodeattr("runtime_writeable_weights")
                    or getCustomOp(node).get_nodeattr("mlo_max_iter")
                )
                if not runtime:
                    new_thresholds, num_always = drop_constant_thresholds(
                        new_thresholds, dtype.min(), never, node.name or node.op_type
                    )
                    if num_always > 0:
                        shift_output_bias(node, num_always)
                    if op_type.startswith("Thresholding"):
                        getCustomOp(node).set_nodeattr("numSteps", new_thresholds.shape[-1])
                # Convert back to the preferred float32 container type
                new_thresholds = new_thresholds.astype(np.float32)
                model.set_initializer(node.input[1], new_thresholds)
                # The thresholds fit the input type unless a channel keeps a
                # threshold that is never met; it needs one more value, max + step.
                # Thresholds written at runtime may take any value.
                if runtime or np.any(new_thresholds == never):
                    tdt = widened_datatype(dtype)
                else:
                    tdt = dtype

                model.set_tensor_datatype(node.input[1], tdt)
                # If hw op we need to set the weight data type attribute as well
                if op_type.startswith("Thresholding"):
                    inst = getCustomOp(node)
                    inst.set_nodeattr("weightDataType", tdt.name)
                # ones
                if new_thresholds.shape != thresholds.shape or np.any(new_thresholds != thresholds):
                    # Track the graph has been modified to inform the transform
                    # container to exhaustively repeat this transformation until
                    # no changes are possible
                    graph_modified = True
                    # Immediately exit here to propagate the data type changes
                    # before considering the next node
                    break

            # Handle MVAU and VVAU nodes with thresholds (noActivation=0)
            elif op_type.startswith("MVAU") or op_type.startswith("VVAU"):
                inst = getCustomOp(node)
                # Only process if node has thresholds (noActivation=0)
                if inst.get_nodeattr("noActivation") == 0 and len(node.input) > 2:
                    thresholds = model.get_initializer(node.input[2])
                    if thresholds is None:
                        continue

                    # Get accumulator datatype (should be set by MinimizeAccumulatorWidth)
                    acc_dt = DataType[inst.get_nodeattr("accDataType")]

                    # This transformation only applies to integer accumulators
                    if not acc_dt.is_integer():
                        continue

                    # Round thresholds up to nearest integer and clip to accumulator range
                    new_thresholds = np.clip(np.ceil(thresholds), acc_dt.min(), acc_dt.max() + 1)
                    # Convert back to the preferred float32 container type
                    new_thresholds = new_thresholds.astype(np.float32)
                    # Insert the rounded and clipped thresholds back into the model
                    model.set_initializer(node.input[2], new_thresholds)
                    # The rounded and clipped thresholds now fit into a data type
                    # that is one bit bigger than the accumulator datatype
                    # Determine new max_value
                    max_val = acc_dt.max() + 1
                    if not acc_dt.signed():
                        tdt = DataType.get_smallest_possible(max_val)
                    else:
                        tdt = DataType.get_smallest_possible(-(max_val) - 1)

                    model.set_tensor_datatype(node.input[2], tdt)

                    if np.any(new_thresholds != thresholds):
                        # Track the graph has been modified to inform the transform
                        # container to exhaustively repeat this transformation until
                        # no changes are possible
                        graph_modified = True
                        # Immediately exit here to propagate the data type changes
                        # before considering the next node
                        break

        model = model.transform(InferDataTypes())
        return model, graph_modified

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
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes


class RoundAndClipThresholds(Transformation):
    """For MultiThreshold nodes operating on integer inputs, round up
    thresholds values to the nearest integer. Additionally, if the input
    is unsigned, sets negative thresholds to zero. Type-casts thresholds (back)
    to the float32 container type (this is separate from the quantization
    annotation). Runs InferDataTypes() afterward to propagate any changes to the
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
                if not dtype.is_integer():
                    continue
                # Round thresholds up to nearest integer and clip thresholds
                # outside the input range
                #   Note: This might promote the thresholds to float64 and
                #   introduce extra inaccuracies due to large integers not being
                #   exactly representable in floating-point representation.
                #   See for example: np.ceil(np.float32(16777217)) == 16777216
                new_thresholds = np.clip(np.ceil(thresholds), dtype.min(), dtype.max() + 1)
                # Convert back to the preferred float32 container type
                new_thresholds = new_thresholds.astype(np.float32)
                # Insert the rounded and clipped thresholds back into the model
                model.set_initializer(node.input[1], new_thresholds)
                # The rounded and clipped thresholds now fit into a data type
                # that is one bit bigger than the input datatype
                # Determine new max_value
                max_val = dtype.max() + 1
                if not dtype.signed():
                    tdt = DataType.get_smallest_possible(max_val)
                else:
                    tdt = DataType.get_smallest_possible(-(max_val) - 1)
                model.set_tensor_datatype(node.input[1], tdt)
                # If hw op we need to set the weight data type attribute as well
                if op_type.startswith("Thresholding"):
                    inst = getCustomOp(node)
                    inst.set_nodeattr("weightDataType", tdt.name)
                # ones
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

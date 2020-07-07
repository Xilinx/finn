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

import numpy as np

from onnx import TensorProto
from onnx import helper as oh

from finn.transformation import Transformation
from finn.core.datatype import DataType


class InsertTopK(Transformation):
    """Add TopK node at the network output and replace the graph output with
    the TopK indices."""

    def __init__(self, k=5, axis=-1, largest=1, sorted=1):
        super().__init__()
        self.k = k
        self.axis = axis
        self.largest = largest
        self.sorted = sorted

    def is_scalar_linear(self, model, node):
        # if is linear
        test = (node.op_type == "Mul") or (node.op_type == "Add")
        if test:
            init = model.get_initializer(node.input[1])
            test = test and (init is not None) and all(x == 1 for x in init.shape)
        return test

    def apply(self, model):
        # get name of output tensor
        graph_out_name = model.graph.output[0].name
        # find final node
        final_node = model.find_producer(graph_out_name)
        # if a top-select op is already present, do nothing
        if final_node.op_type == "TopK":
            return (model, False)
        else:
            # remove any scalar linear transformations at graph output
            # because TopK is invariant to them
            while self.is_scalar_linear(model, final_node):
                # remove the predecessor
                final_node_input = model.get_tensor_valueinfo(final_node.input[0])
                model.graph.output.insert(0, final_node_input)
                model.graph.output.pop(1)
                model.graph.node.remove(final_node)
                graph_out_name = model.graph.output[0].name
                final_node = model.find_producer(graph_out_name)

            out_shape = model.get_tensor_shape(graph_out_name)
            out_dtype = model.get_tensor_datatype(graph_out_name)
            # adjust shape
            out_shape[self.axis] = self.k
            # make new buffer
            k_tensor = np.array([self.k]).astype(np.int64)
            k_value = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.INT64, [1]
            )
            topk_values = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
            )
            topk_indices = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.INT64, out_shape
            )
            model.graph.value_info.append(k_value)
            model.set_tensor_datatype(k_value.name, out_dtype)  # TODO set to int64
            model.graph.value_info.append(topk_values)
            model.set_tensor_datatype(topk_values.name, out_dtype)
            # create and append topk node
            model.set_initializer(k_value.name, k_tensor)
            topk_node = oh.make_node(
                "TopK",
                inputs=[graph_out_name, k_value.name],
                outputs=[topk_values.name, topk_indices.name],
                axis=self.axis,
                largest=self.largest,
                sorted=self.sorted,
            )
            model.graph.node.append(topk_node)
            # replace the existing output definition with topk indices
            model.graph.output.insert(0, topk_indices)
            model.graph.output.pop(1)
            # set quantization annotation for indices
            # minimal output dtype for TopK indices dependens on num. classes
            # assuming UINT32 is large enough for now (FINN has currently no
            # DataType.INT64)
            model.set_tensor_datatype(topk_indices.name, DataType.UINT32)
            return (model, True)

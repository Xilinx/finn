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
from finn.transformation.infer_shapes import InferShapes


class BatchNormToAffine(Transformation):
    """Replaces any test-time BatchNorm layers with Mul-Add layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "BatchNormalization":
                graph_modified = True
                bn_input = n.input[0]
                bn_output = n.output[0]
                # extract batchnorm parameters as numpy arrays
                scale = model.get_initializer(n.input[1])
                bias = model.get_initializer(n.input[2])
                mean = model.get_initializer(n.input[3])
                variance = model.get_initializer(n.input[4])
                epsilon = 1e-5
                # find A and B to compute batchnorm as affine transpose Ax+B
                # TODO is a division by moving avg factor needed for variance?
                A = scale / np.sqrt(epsilon + variance)
                B = bias - (A * mean)
                # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
                producer = model.find_producer(bn_input)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        bn_input = producer.input[0]
                consumer = model.find_consumer(bn_output)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        bn_output = consumer.output[0]
                data_shape = model.get_tensor_shape(bn_input)
                # create value_info and initializers for Mul and Add constants
                mul_const = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape
                )
                graph.value_info.append(mul_const)
                model.set_initializer(mul_const.name, A)
                mul_output = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, data_shape
                )
                graph.value_info.append(mul_output)
                add_const = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape
                )
                graph.value_info.append(add_const)
                model.set_initializer(add_const.name, B)
                # create Mul and Add nodes to replace the batchnorm
                mul_node = oh.make_node(
                    "Mul", [bn_input, mul_const.name], [mul_output.name]
                )
                add_node = oh.make_node(
                    "Add", [mul_output.name, add_const.name], [bn_output]
                )
                # insert where the batchnorm is to preserve topological ordering
                graph.node.insert(node_ind, mul_node)
                graph.node.insert(node_ind + 1, add_node)
                # remove old nodes
                graph.node.remove(n)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        graph.node.remove(consumer)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        graph.node.remove(producer)
        model = model.transform(InferShapes())
        return (model, graph_modified)

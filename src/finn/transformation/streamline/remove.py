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


from finn.transformation.base import Transformation
from finn.transformation.infer_shapes import InferShapes
import numpy as np


def _remove_node_and_rewire(model, node):
    producer = model.find_producer(node.input[0])
    if producer is not None:
        # wire output tensor to
        # output of producer node
        producer.output[0] = node.output[0]
    else:
        # node is first in graph
        consumer = model.find_consumer(node.output[0])
        assert consumer is not None, "Whole graph is identity"
        assert consumer.input[0] == node.output[0]
        # rewire consumer's input directly to graph input
        consumer.input[0] = node.input[0]
    # remove node
    model.graph.node.remove(node)


class RemoveIdentityOps(Transformation):
    """Remove identity ops like Add/Sub with zero or Mul/Div with one. A tolerance
    value (defaults to 1e-05) can be specified during init for the comparison
    to zero/one."""

    def __init__(self, atol=1e-05):
        super().__init__()
        self.atol = atol

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type in ["Add", "Sub"]
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                A = model.get_initializer(n.input[1])
                if (
                    A is not None
                    and np.isclose(A, np.zeros_like(A), atol=self.atol).all()
                ):
                    _remove_node_and_rewire(model, n)

            elif (
                n.op_type in ["Mul", "Div"]
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                A = model.get_initializer(n.input[1])
                if (
                    A is not None
                    and np.isclose(A, np.ones_like(A), atol=self.atol).all()
                ):
                    _remove_node_and_rewire(model, n)
        model = model.transform(InferShapes())
        return (model, graph_modified)

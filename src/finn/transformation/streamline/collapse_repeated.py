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

from onnx import helper as oh

from finn.core.datatype import DataType
from finn.transformation.base import Transformation
from finn.transformation.infer_shapes import InferShapes


class CollapseRepeatedOp(Transformation):
    """Collapse repeated consecutive operations with constant parameters into
    a single operation. make_collapsed_param_fxn must take two tensors and
    return a tensor which gives the equivalent result using a single op."""

    def __init__(self, op_name, make_collapsed_param_fxn):
        super().__init__()
        self.op_name = op_name
        self.make_collapsed_param_fxn = make_collapsed_param_fxn

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == self.op_name
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == self.op_name
                    and not model.is_join_node(consumer)
                ):
                    op0_param_name = n.input[1]
                    op1_param_name = consumer.input[1]
                    op0_param = model.get_initializer(op0_param_name)
                    op1_param = model.get_initializer(op1_param_name)
                    assert (
                        op0_param is not None
                    ), """Initializer for parameters for
                    op0 is not set."""
                    assert (
                        op1_param is not None
                    ), """Initializer for parameters for
                    op1 is not set."""
                    start_name = n.input[0]
                    end_name = consumer.output[0]
                    # compute the new parameter
                    new_param = self.make_collapsed_param_fxn(op0_param, op1_param)
                    # make and insert new node
                    new_node_param_name = op0_param_name
                    new_node = oh.make_node(
                        self.op_name, [start_name, new_node_param_name], [end_name]
                    )
                    graph.node.insert(node_ind, new_node)
                    # replace parameter value
                    model.set_initializer(new_node_param_name, new_param)
                    # be conservative with param/output DataTypes
                    model.set_tensor_datatype(new_node_param_name, DataType["FLOAT32"])
                    model.set_tensor_datatype(end_name, DataType["FLOAT32"])
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class CollapseRepeatedAdd(CollapseRepeatedOp):
    """Collapse repeated adder node into a single operation."""

    def __init__(self):
        super().__init__("Add", lambda x, y: y + x)


class CollapseRepeatedMul(CollapseRepeatedOp):
    """Collapse repeated multiplier node into a single operation."""

    def __init__(self):
        super().__init__("Mul", lambda x, y: y * x)

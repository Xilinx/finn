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

# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX arbitrary precision data types
from qonnx.core.datatype import DataType

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_shapes import InferShapes

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name


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


# Collapses repeated transpose operations into a single transpose operation
# having the same effect
class CollapseRepeatedTranspose(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "Transpose":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Transpose is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Successor must be a Transpose to be collapsed
                if successor.op_type != "Transpose":
                    # Softly skip this node
                    continue
                # Get the (optional) permutation indices of the first transpose
                # in case it is a multi-axis transpose
                perm1 = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers
                perm1 = perm1.ints if perm1 is not None else None

                # Get the (optional) permutation indices of the second transpose
                # in case it is a multi-axis transpose
                perm2 = get_by_name(successor.attribute, "perm")
                # Convert permutation indices to list of integers
                perm2 = perm2.ints if perm2 is not None else None

                # Get the shape of the input tensor
                shape = model.get_tensor_shape(
                    # fmt: off
                    node.input[0], fix_missing_init_shape=True
                    # fmt: on
                )
                # List of dimension indices in order
                dims = range(len(shape))

                # Substitute the permutation indices by the reversed index list
                # if they are not given: This is default behavior, see the docs:
                #   https://onnx.ai/onnx/operators/onnx__Transpose.html
                perm1 = list(reversed(dims)) if perm1 is None else perm1
                perm2 = list(reversed(dims)) if perm2 is None else perm2

                # Combined permutation permutes the first permutation of the
                # dimensions according to the second permutation
                perm = [perm1[i] for i in perm2]

                # Create a new Transpose operator replacing the other two
                transpose = oh.make_node(
                    # Name of the operator type
                    "Transpose",
                    # Connect to the inputs to the first transpose
                    inputs=node.input,
                    # Connect to the outputs of the second transpose
                    outputs=successor.output,
                    # Insert the new permutation indices
                    perm=perm,
                )
                # Insert the collapsed transpose operator
                graph.node.insert(index + 2, transpose)
                # Remove the two original transpose operators
                graph.node.remove(node)
                graph.node.remove(successor)
                # Track whether the graph has been modified, never resets to
                # False
                graph_modified = True
                # Break the loop after adding and removing nodes to start over
                # with a clean index
                break
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

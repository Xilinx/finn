# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_shapes import InferShapes

# Reuse node removal and rewiring from qonnx
from qonnx.transformation.remove import remove_node_and_rewire

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name


# Removes identity reshape operations, i.e., Reshape where input shape is the
# same as the target shape
class RemoveIdentityReshape(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Reshape operation types
            if node.op_type == "Reshape":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Second input to the reshape operation is the target shape
                shape = model.get_initializer(node.input[1])
                # If the initializer is present, this is a constant shape
                # reshape which can be removed if it does not reshape
                if shape is not None:
                    # Get the shape of the input to the reshape
                    inp = model.get_tensor_shape(node.input[0])
                    # If input and target shape are the same, this is an
                    # identity operation
                    if len(shape) == len(inp) and (shape == inp).all():  # noqa
                        # Remove and rewire this node
                        remove_node_and_rewire(model, node)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Removes identity transpose operations, i.e., Transpose where input order is
# the same as the target permutation
class RemoveIdentityTranspose(Transformation):
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
                if model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # If the permutation indices are given, we can check whether
                # they are in order making this an identity transpose
                # Note: Without perm attribute, this is implicitly reversing the
                # axes, i.e., not an identity transpose
                if perm is not None:
                    # Convert permutation indices to list of integers
                    perm = perm.ints
                    # Get the shape of the input tensor
                    shape = model.get_tensor_shape(
                        # fmt: off
                        node.input[0], fix_missing_init_shape=True
                        # fmt: on
                    )
                    # If the permutation indices cover the input shape in order,
                    # this transpose does nothing
                    if perm == [i for i in range(len(shape))]:
                        # Remove and rewire this node
                        remove_node_and_rewire(model, node)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

# QONNX wrapper of ONNX model graphs
# For array handling
import numpy as np

# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name, remove_by_name


# Squeezes, i.e., removes, dimensions of size 1
# Note: Use this transformation with great care, it currently serves only the
# purpose of turning the not well-supported 3d data layouts encountered in
# transformer models with batch dimension of size 1 into 2d data layouts where
# the sequence dimension is treated as a batch dimension. Everything else is
# not tested, it might break the model or simply lack support for certain node
# op-types.
class Squeeze(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # There should not be any squeeze or unsqueeze operations in the
            # graph as these would interfere with this transformation
            assert node.op_type not in {
                "Squeeze",
                "Unsqueeze",
            }, f"Squeezing graph containing {node.op_type}"

            # Validate slice not slicing along squeezed dimension
            if node.op_type == "Slice":
                # Axes to slice along is supplied as the 4th input to the node
                axes = model.get_initializer(node.input[3])
                # If this is an initializer, there are constant axes to slice
                if axes is not None:
                    # Get the shape of the input, assuming the input from
                    # upstream to be the 1st input
                    shape = model.get_tensor_shape(node.input[0])
                    # Slice might operate on multiple axes
                    for axis in axes:
                        # Axis must not refer to a dimension of size 1
                        # fmt: off
                        assert shape[axis] > 1, \
                            f"Slice along dimension to be squeezed: {node.name}"
                        # fmt: on

            # Need to adapt reshape operations to drop dimensions of size 1
            if node.op_type == "Reshape":
                # Second input to the reshape operation is the target shape
                shape = model.get_initializer(node.input[1])
                # If the initializer is present, this is a constant shape
                # reshape which can be replaced by the squeezed shape
                if shape is not None:
                    # Squeeze the shape by removing all dimensions with size 1
                    # fmt: off
                    new_shape = np.asarray([
                        size for size in shape if size != 1
                    ])
                    # fmt: on
                    # Reassign the squeezed tensor
                    model.set_initializer(node.input[1], new_shape)
                    # Track whether the shape actually changed
                    if len(new_shape) != len(shape):
                        # Is never reset back to False during iteration
                        graph_modified = True

            # Need to drop dimensions of size 1 from transpose permutation list
            if node.op_type == "Transpose":
                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # If the permutation indices are given, we need to remove all
                # dimension of size 1 from these
                if perm is not None:
                    # Convert permutation indices to list of integers
                    perm = perm.ints
                    # Get the shape of the input tensor to seek for input
                    # dimensions of size 1
                    shape = model.get_tensor_shape(
                        # fmt: off
                        node.input[0], fix_missing_init_shape=True
                        # fmt: on
                    )
                    # Keep track of new axis enumeration, skipping dimensions of
                    # size 1
                    mapping, new_axis = {}, 0
                    # Enumerate the sizes per axis
                    for axis, size in enumerate(shape):
                        # Insert mapping from old to new axis
                        mapping[axis] = new_axis
                        # Only advance the new axis index for dimensions not to
                        # be squeezed
                        new_axis += size > 1
                    # Filter and remap the axis enumeration of the permutation
                    new_perm = [
                        # fmt: off
                        mapping[axis] for axis in perm if shape[axis] > 1
                        # fmt: on
                    ]
                    # Track whether the permutations actually changed
                    if len(new_perm) != len(perm) or new_perm != perm:
                        # Is never reset back to False during iteration
                        graph_modified = True
                    # Remove the permutation attribute before setting the new
                    # permutation
                    remove_by_name(node.attribute, "perm")
                    # Insert new permutation attribute
                    node.attribute.append(oh.make_attribute("perm", new_perm))

            # Need to squeeze the number of inputs to multi-head splitting
            if node.op_type == "SplitMultiHeads":
                # Get number of input feature maps to the merging operation
                num_inputs = get_by_name(node.attribute, "num_inputs")  # noqa
                # Squeeze all dimensions of size 1
                new_num_inputs = [size for size in num_inputs.ints if size != 1]
                # Update the attribute by removing and reinserting
                remove_by_name(node.attribute, "num_inputs")
                node.attribute.append(
                    # fmt: off
                    oh.make_attribute("num_inputs", new_num_inputs)
                    # fmt: on
                )
                # Track whether the number of inputs actually changed
                if len(new_num_inputs) != len(num_inputs.ints):
                    # Is never reset back to False during iteration
                    graph_modified = True

            # Need to set the squeezed output mode of multi-head merging
            if node.op_type == "MergeMultiHeads":
                # Remove the squeezed attribute
                remove_by_name(node.attribute, "squeezed")
                # Set squeezed mode attribute
                node.attribute.append(oh.make_attribute("squeezed", True))
                # Get number of input feature maps to the merging operation
                num_inputs = get_by_name(node.attribute, "num_inputs")  # noqa
                # Squeeze all dimensions of size 1
                new_num_inputs = [size for size in num_inputs.ints if size != 1]
                # Update the attribute by removing and reinserting
                remove_by_name(node.attribute, "num_inputs")
                node.attribute.append(
                    # fmt: off
                    oh.make_attribute("num_inputs", new_num_inputs)
                    # fmt: on
                )
                # Track whether the number of inputs actually changed
                if len(new_num_inputs) != len(num_inputs.ints):
                    # Is never reset back to False during iteration
                    graph_modified = True

        # Iterate all tensors in the graph keeping track of the index
        for index, name in enumerate(model.get_all_tensor_names()):
            # Query the shape of the tensor adding annotations for initializers
            # if missing
            shape = model.get_tensor_shape(name, fix_missing_init_shape=True)
            # Skip squeezing 0d or 1d tensors
            if len(shape) <= 1:
                continue
            # Squeeze the shape by removing all dimensions with size 1
            new_shape = [size for size in shape if size != 1]
            # Try to get the initializer of the tensor
            initializer = model.get_initializer(name)
            # If an initializer is present replace by the squeezed tensor
            if initializer is not None:
                # Reassign the squeezed tensor
                model.set_initializer(name, initializer.squeeze())
            # Set new shape annotation
            model.set_tensor_shape(name, new_shape)
            # Track whether the shape actually changed
            if len(new_shape) != len(shape):
                # Is never reset back to False during iteration
                graph_modified = True
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

# QONNX wrapper of ONNX model graphs
# For array handling
import numpy as np

# Python warning subsystem
import warnings

# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# Transformations running qonnx datatype inference
from qonnx.transformation.infer_datatypes import InferDataTypes

# Transformation running onnx shape inference
from qonnx.transformation.infer_shapes import InferShapes

# Reuse node removal and rewiring from qonnx
from qonnx.transformation.remove import remove_node_and_rewire

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name, remove_by_name

# Small utility functions for graph transformations
from .util import is_threshold


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
        # # Keep track of whether the graph has been modified
        # graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # There should not be any squeeze or unsqueeze operations in the
            # graph as these would interfere with this transformation
            if node.op_type in {"Squeeze", "Unsqueeze"}:
                # Issue a warning to make the user aware of this potential issue
                # fmt: off
                warnings.warn(
                    f"Squeezing graph containing {node.op_type}: {node.name}"
                )
                # fmt: on

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
                        # graph_modified = True
                        pass

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
                        # # Is never reset back to False during iteration
                        # graph_modified = True
                        pass
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
                    # # Is never reset back to False during iteration
                    # graph_modified = True
                    pass

            # Need to adjust the index of the split axis by the amount of
            # squeezed axes before
            if node.op_type == "Split":
                # Get the axis attribute from the Split operator
                axis = get_by_name(node.attribute, "axis")
                # Convert to integer or substitute default 0 according to ONNX
                # reference
                axis = axis.i if axis is not None else 0
                # Get the shape of the input tensor to the split operation
                shape = model.get_tensor_shape(node.input[0])
                # Subtract the number of squeezed, i.e, size=1, axes before axis
                axis = axis - sum(size == 1 for size in shape[:axis])
                # Update the attribute by removing and reinserting
                remove_by_name(node.attribute, "axis")
                node.attribute.append(oh.make_attribute("axis", axis))

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
                    # # Is never reset back to False during iteration
                    # graph_modified = True
                    pass

            # Need to patch the Im2Col operator when squeezing as this cannot
            # operate on other data layouts than 4-dimensional layouts
            if node.op_type == "Im2Col":
                # Do not squeeze the same operation twice
                if get_by_name(node.attribute, "squeezed"):
                    continue
                # Add a new marker attribute to not squeeze this node again
                node.attribute.append(oh.make_attribute("squeezed", True))
                # Get the shape of the input tensor to seek for input
                # dimensions of size 1
                shape = model.get_tensor_shape(
                    # fmt: off
                    node.input[0], fix_missing_init_shape=True
                    # fmt: on
                )
                # Skip if there is no shape
                if shape is None:
                    continue
                # Get the axes to be squeezed, i.e., dimensions of size 1
                axes = [dim for dim, size in enumerate(shape) if size == 1]
                # To be compatible with ONNX opset >= 13, the axes to
                # unsqueeze/squeeze need to be provided as an input
                axes_input = model.make_new_valueinfo_name()
                # Set the axes as an initializer list
                model.set_initializer(axes_input, np.asarray(axes))
                # Instantiate an unsqueeze operation adapting from the squeezed
                # layout back to the 4-dimensional layout
                unsqueeze = oh.make_node(
                    # Unsqueeze ONNX operators
                    "Unsqueeze",
                    # Inherit the inputs from the Im2Col operation
                    inputs=[node.input[0], axes_input],
                    # Create a new output tensor
                    outputs=[model.make_new_valueinfo_name()],
                    # Specify the axes to unsqueeze
                    axes=axes,
                )
                # Instantiate a squeeze operator adapting from unsqueezed
                # 4-dimensional layout back to the squeezed layout
                squeeze = oh.make_node(
                    # Squeeze ONNX operators
                    "Squeeze",
                    # Create a new input tensor
                    inputs=[model.make_new_valueinfo_name(), axes_input],
                    # Inherit the output tensor from the Im2Col operation
                    outputs=node.output,
                    # Specify the axes to squeeze
                    axes=axes,
                )
                # Rewire the input/output to/from the Im2Col operator to connect
                # the Unsqueeze/Squeeze wrapper
                node.input[0] = unsqueeze.output[0]
                node.output[0] = squeeze.input[0]
                # Insert the new nodes
                graph.node.insert(index, unsqueeze)
                graph.node.insert(index, squeeze)
                # # The graph has now been modified. This is never reset back to
                # # False during iteration
                # graph_modified = True

        # Iterate the graph once again to get rid of existing Squeeze/Unsqueeze
        # Note: This needs to be done after all other operations to not mess
        # with the shape annotations
        for index, node in enumerate(graph.node):
            # Squeeze and Unsqueeze can be handled the same
            if node.op_type in {"Squeeze", "Unsqueeze"}:
                # Do not touch the Unsqueeze/Squeeze surrounding the Im2Col
                # operation
                if "Im2Col" not in [
                    n.op_type
                    for n in [
                        *model.find_direct_predecessors(node),
                        *model.find_direct_successors(node),
                    ]
                ]:
                    # Remove existing Squeeze/Unsqueeze from the graph as these
                    # will not have any effect anymore
                    remove_node_and_rewire(model, node)

        # Get the names of all global input tensors to insert a Squeeze
        # operation in front
        global_inputs = [inp.name for inp in model.graph.input]
        # Insert Squeeze operators at each global input
        for inp in global_inputs:
            # Get the shape of the tensor to seek for dimensions of size 1
            shape = model.get_tensor_shape(  # noqa: Duplicate
                inp, fix_missing_init_shape=True
            )
            # Skip if there is no shape and skip squeezing 0d or 1d tensors
            if shape is None or len(shape) <= 1:
                continue
            # Get the axes to be squeezed, i.e., dimensions of size 1
            axes = [dim for dim, size in enumerate(shape) if size == 1]
            # Te be compatible with ONNX opset >= 13, the axes to
            # unsqueeze/squeeze need to be provided as an input
            axes_input = model.make_new_valueinfo_name()
            # Set the axes as an initializer list
            model.set_initializer(axes_input, np.asarray(axes))
            # Instantiate the squeeze operator
            squeeze = oh.make_node(
                # Squeeze ONNX operators
                "Squeeze",
                # Inherit the input from the global input and add axes to be
                # squeezed to the input list
                inputs=[inp, axes_input],
                # Create a new output connecting to the graph
                outputs=[model.make_new_valueinfo_name()],
                # Specify the axes to squeeze
                axes=axes,
            )
            # Connect the new squeeze operator to all consumers of this
            # global input
            for consumer in model.find_consumers(inp):
                # Find the inputs of the consumer which are the global input
                for i, c_inp in enumerate(consumer.input):
                    # Note: This might happen multiple times?
                    if c_inp == inp:
                        # Rewire consumer's input directly to the output of
                        # the squeeze operation
                        consumer.input[i] = squeeze.output[0]
            # Insert the squeeze operator into the model graph
            model.graph.node.insert(0, squeeze)

        # Get the names of all global output tensors to insert an Unsqueeze
        # operation afterward
        global_outputs = [out.name for out in model.graph.output]
        # Insert Unsqueeze operators at each global output
        for out in global_outputs:
            # Get the shape of the tensor to seek for dimensions of size 1
            shape = model.get_tensor_shape(  # noqa: Duplicate
                out, fix_missing_init_shape=True
            )
            # Skip if there is no shape and skip squeezing 0d or 1d tensors
            if shape is None or len(shape) <= 1:
                continue
            # Get the axes to be squeezed, i.e., dimensions of size 1
            axes = [dim for dim, size in enumerate(shape) if size == 1]
            # Te be compatible with ONNX opset >= 13, the axes to
            # unsqueeze/squeeze need to be provided as an input
            axes_input = model.make_new_valueinfo_name()
            # Set the axes as an initializer list
            model.set_initializer(axes_input, np.asarray(axes))
            # Instantiate the unsqueeze operator
            unsqueeze = oh.make_node(
                # Unsqueeze ONNX operators
                "Unsqueeze",
                # Connect to a new intermediate tensor
                inputs=[model.make_new_valueinfo_name(), axes_input],
                # Connect tho the global output
                outputs=[out],
                # Specify the axes to unsqueeze
                axes=axes,
            )
            # Connect the new unsqueeze operator to the producer of this global
            # output
            producer = model.find_producer(out)
            # Find the output of the producer which is the global output
            for i, p_out in enumerate(producer.output):
                # Note: This might happen multiple times?
                if p_out == out:
                    # Rewire producer's output directly to the input of
                    # the unsqueeze operation
                    producer.output[i] = unsqueeze.input[0]
            # Insert the unsqueeze operator into the model graph
            model.graph.node.insert(0, unsqueeze)

        # Iterate all tensors in the graph keeping track of the index
        for index, name in enumerate(model.get_all_tensor_names()):
            # Skip the global inputs and outputs
            if name in [*global_inputs, *global_outputs]:
                # Skip without warning, these are handled by explicit
                # Squeeze/Unsqueeze operations
                continue
            # Skip initializer tensors: Shape inference should actually restore
            # these shapes, but for some reason it does not work...
            if (init := model.get_initializer(name)) is not None:
                # If any of the consumers of this initializer is a
                # multi-threshold function, it should not be squeezed as the
                # thresholding is quite sensitive to data layouts and does not
                # handle broadcasting.
                # Note: Not sue whether there can actually be cases wih multiple
                # consumers of a threshold tensor, but this should be perfectly
                # legal according to standard ONNX.
                if any(is_threshold(op) for op in model.find_consumers(name)):
                    # Skip without warning
                    continue
                # First squeeze the actual data of the initializer tensors
                model.set_initializer(name, np.squeeze(init))
                # Now also annotate the squeezed shape, otherwise the following
                # shape inference might fail or break the graph
                # Note: Deleting the annotation is not sufficient here, it is
                # not recovered properly from the tensor data for some reason...
                model.set_tensor_shape(name, np.squeeze(init).shape)
                # Continue with the next tensor, skipping the default case below
                continue
            # Just delete all existing shape annotations to redo them later
            model.set_tensor_shape(name, None)
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether this transformation
        # needs to be repeated
        # Note: Never repeat this transformation as it might break when
        # inserting multiple Squeeze operators
        return model, False

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Make copies and deep copies of python objects
import copy

# Need numpy for modifying the onnx graph tensors, which are numpy style arrays
import numpy as np

# Output warning messages
import warnings

# Utility for handling ONNX nodes and tensors
from onnx import NodeProto
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import GiveUniqueParameterTensors

# Transformation running qonnx datatype inference
from qonnx.transformation.infer_datatypes import InferDataTypes

# Transformation running onnx shape inference
from qonnx.transformation.infer_shapes import InferShapes

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name, remove_by_name

# Utility function for transforming ONNX graphs
from finn.transformation.util import (
    is_reshape_transpose,
    is_transpose_reshape,
    op_types,
)


# Infers reshaping of attention heads, i.e., converts the Reshape and transpose
# patterns to the SplitMultiHeads and MergeMultiHeads hardware custom operators.
class InferMultiHeads(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Head-slicing reshaping is triggered by detecting a reshape
            # operation followed by a transpose
            if is_reshape_transpose(node, model):
                # Get the single successor node
                transpose = model.find_direct_successors(node)[0]

                # Get the input and output tensor names to the pattern
                inp = node.input[0]
                mid = node.output[0]
                end = transpose.output[0]

                # Get the shape of the input tensor for inferring the number of
                # heads and correctly propagating shapes
                shape = model.get_tensor_shape(inp)
                # Determine the rank of the input tensor to support batched and
                # non-batched inputs
                rank = len(shape)

                # The input shape determines the sequence length
                seq, _, dim = shape if (rank == 3) else (shape[0], 1, shape[1])

                # The intermediate shape must be the same as specified as the
                # second input to the reshape operation
                assert (model.get_tensor_shape(mid)  # noqa
                        == model.get_initializer(node.input[1])).all()  # noqa
                # Expected layout after reshape is "head last"
                _, heads, _ = model.get_tensor_shape(mid)

                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(transpose.attribute, "perm")
                # Convert permutation indices to list of integers if it is
                # given
                perm = perm.ints if perm is not None else None

                # Transpose must either keep or flip the sequence and embedding
                # dimensions
                if perm not in [[1, 0, 2], [1, 2, 0]]:
                    # Issue a warning of near match of the supported head
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported permutation near {transpose.name}: {perm}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Check whether the transpose only permutes to head first or
                # additionally transposes sequence and embedding dimension as
                # well
                keep_transpose = (perm == [1, 2, 0])

                # Start assuming there is no middle node, as the transpose is
                # removed
                maybe_mid = end

                # Insert a new transpose node if the sequence and embedding
                # dimensions are flipped
                if keep_transpose:
                    # Construct a new intermediate tensor using the current one
                    # as template
                    maybe_mid = mid
                    # Construct a new Transpose with attributes inferred from
                    # the detected graph patter
                    new_transpose = oh.make_node(**{
                        "op_type": "Transpose",
                        # Named inputs extracted from the graph pattern
                        "inputs": [maybe_mid],
                        # Named outputs extracted from the graph pattern
                        "outputs": [end],
                        # Give node name derived from the operator type and the
                        # name of the triggering node to be removed
                        "name": f"MultiHeads_Transpose_{node.name}",
                        # Permute the last two dimensions
                        "perm": [0, 2, 1]
                    })
                    # Insert the new node into the graph
                    graph.node.insert(index + 1, new_transpose)
                    # Change the shape of the intermediate tensor to reflect
                    # partial reshaping
                    model.set_tensor_shape(
                        maybe_mid, (heads, seq, dim // heads)
                    )

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "SplitMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    "domain": "finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs extracted from the graph pattern
                    "inputs": [inp],
                    # Named outputs extracted from the graph pattern
                    "outputs": [maybe_mid],
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"SplitMultiHeads_{node.name}",
                    # Number of attention heads inferred
                    "heads": heads,
                    # Inferred multi-heads produce packed tensors
                    "packed": True,
                    # Datatype of inputs and outputs
                    "dtype": model.get_tensor_datatype(node.input[0]).name,
                    # Number of input elements, i.e., embedding dimension
                    "num_elems": dim,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    "num_inputs": [seq, 1] if (rank == 3) else [seq]
                }

                # Create a new custom node replacing the multi head reshape
                heads = oh.make_node(**kwargs)
                # Insert the new node into the graph
                graph.node.insert(index, heads)
                # Collect all nodes comprising the original pattern
                nodes = [node, transpose]
                # Remove all nodes of the original pattern
                for n in nodes:
                    # Do not try to remove non-existing nodes
                    if n is not None:
                        graph.node.remove(n)
                # The graph has been modified
                graph_modified = True

            # Head-merging reshaping is triggered by detecting a transpose
            # operation followed by a reshape
            if is_transpose_reshape(node, model):
                # Get the single successor node
                reshape = model.find_direct_successors(node)[0]

                # Get the input and output tensor names to the pattern
                inp = node.input[0]
                end = reshape.output[0]

                # The input shape determines the heads, sequence length and
                # embedding dimension
                heads, seq, dim = model.get_tensor_shape(inp)

                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers if it is given
                perm = perm.ints if perm is not None else None

                # Transpose must flip the heads and sequence dimensions
                if perm not in [[1, 0, 2]]:
                    # Issue a warning of near match of the supported head
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported permutation near {node.name}: {perm}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Shape of the final output of the operator pattern
                out_shape = model.get_tensor_shape(end)

                # The output of the reshape must be the same as specified as the
                # second input to the reshape operation
                assert (out_shape  # noqa
                        == model.get_initializer(reshape.input[1])).all()

                # The final output shape must match the expectation of
                # reintegrating the heads back into the embeddings
                if out_shape not in [[seq, heads * dim], [seq, 1, heads * dim]]:
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Output shape mismatch near: {reshape.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "MergeMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    "domain": "finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs extracted from the graph pattern
                    "inputs": [inp],
                    # Named outputs extracted from the graph pattern
                    "outputs": [end],
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"MergeMultiHeads_{node.name}",
                    # Number of attention heads inferred
                    "heads": heads,
                    # Remember, whether the output needs to be squeezed
                    "squeezed": out_shape == [seq, heads * dim],
                    # Inferred multi-heads produce packed tensors
                    "packed": True,
                    # Datatype of inputs and outputs
                    "dtype": model.get_tensor_datatype(node.input[0]).name,
                    # Number of input elements, i.e., embedding dimension
                    "num_elems": dim,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    "num_inputs": [heads, seq],
                }

                # Create a new custom node replacing the multi head reshape
                heads = oh.make_node(**kwargs)
                # Insert the new node into the graph
                graph.node.insert(index, heads)
                # Collect all nodes comprising the original pattern
                nodes = [node, reshape]
                # Remove all nodes of the original pattern
                for n in nodes:
                    # Do not try to remove non-existing nodes
                    if n is not None:
                        graph.node.remove(n)
                # The graph has been modified
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Move SplitMultiHeads operation past MultiThreshold operation. This is required
# as a precondition for later unrolling the attention heads, as there may not be
# any other operations between splitting and merging the attention heads,
# besides the actual attention operator.
class MoveSplitMultiHeadsPastMultiThreshold(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Transformation applies to SplitMultiHeads operation (not Merge)
            if node.op_type == "SplitMultiHeads":
                # Slicing should not fork or join
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Slicing may not join or fork: {node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue
                # Now we know there is only one consumer operation following the
                # slice node
                thresholds_node = model.find_direct_successors(node)[0]  # noqa
                # Successor must actually be a MultiThresholds for this
                # transform to apply
                if not thresholds_node.op_type == "MultiThreshold":
                    # Skip transforming this instance, probably no need to warn
                    continue

                # Thresholds should not fork or join either
                if (model.is_fork_node(thresholds_node)
                        or model.is_join_node(thresholds_node)):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"MultiThreshold may not join or fork:"
                        f" {thresholds_node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Get the thresholds tensor, which must be an initializer at
                # the second input
                thresholds = model.get_initializer(thresholds_node.input[1])
                # This is indeed an error, no way to recover from this, so
                # assertion is fine
                assert thresholds is not None, \
                    f"Missing threshold tensor for {thresholds_node.name}"

                # The slice node should have an attribute specifying the number
                # of heads
                heads = get_by_name(node.attribute, "heads")
                # Heads must be present, otherwise this is an errr
                assert heads is not None, \
                    f"Missing number of heads for {node.name}"
                # Convert heads attribute proto to integer
                heads = heads.i

                # Repeat the thresholds for each head along the channel
                # dimension
                thresholds = np.concatenate(heads * [thresholds])
                # Update the thresholds tensor to simply repurpose the existing
                # node
                model.set_initializer(thresholds_node.input[1], thresholds)

                # Get names of all tensors involved in connecting the nodes
                inp = node.input[0]
                mid = node.output[0]
                out = thresholds_node.output[0]

                # The middle tensor is now produced by the multi-threshold,
                # which does not change the shape. Propagate the shape of the
                # input tensor
                model.set_tensor_shape(mid, model.get_tensor_shape(inp))
                # As the middle tensor is now produced by the multi-threshold,
                # the datatype needs to be taken from the output tensor
                model.set_tensor_datatype(mid, model.get_tensor_datatype(out))
                # Remove the datatype attribute before setting the new
                # datatype
                remove_by_name(node.attribute, "dtype")
                # Insert new datatype attribute
                node.attribute.append(
                    oh.make_attribute(
                        "dtype", model.get_tensor_datatype(out).name
                    )
                )

                # Rewire the nodes locally switching order. Reuses all the
                # exising tensors.
                thresholds_node.input[0] = inp
                thresholds_node.output[0] = mid
                node.input[0] = mid
                node.output[0] = out

                # Graph has been modified, required additional transformations
                # to be run
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Move MergeMultiHeads operation past MultiThreshold operation to avoid merging
# excessively large streams and maybe even allow absorbing the thresholds into
# the attention operator.
class MoveMergeMultiHeadsPastMultiThreshold(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Transformation applies to MergeMultiHeads operation
            if node.op_type == "MergeMultiHeads":
                # Merging should not fork, but it may join
                if model.is_fork_node(node):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Slicing may not fork: {node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue
                # Now we know there is only one consumer operation following the
                # slice node
                thresholds_node = model.find_direct_successors(node)[0]  # noqa
                # Successor must actually be a MultiThresholds for this
                # transform to apply
                if not thresholds_node.op_type == "MultiThreshold":
                    # Skip transforming this instance, probably no need to warn
                    continue

                # Thresholds must not fork or join either
                if (model.is_fork_node(thresholds_node)
                        or model.is_join_node(thresholds_node)):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"MultiThreshold may not join or fork:"
                        f" {thresholds_node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Get the thresholds tensor, which must be an initializer at
                # the second input
                thresholds = model.get_initializer(thresholds_node.input[1])
                # This is indeed an error, no way to recover from this, so
                # assertion is fine
                assert thresholds is not None, \
                    f"Missing threshold tensor for {thresholds_node.name}"

                # The merge node should have an attribute specifying the number
                # of heads
                heads = get_by_name(node.attribute, "heads")
                # Heads must be present, otherwise this is an errr
                assert heads is not None, \
                    f"Missing number of heads for {node.name}"
                # Convert heads attribute proto to integer
                heads = heads.i

                # Split the thresholds for each head along the channel dimension
                # Note: This is a list of thresholds per head now
                thresholds = np.split(thresholds, heads)

                # Need to insert a new thresholding operation at each input of
                # the multi-head merging
                for i, inp in enumerate(node.input):
                    # Start by making a full copy of the original thresholds
                    # node
                    new_thresholds = copy.deepcopy(thresholds_node)
                    # The input to the original merging node becomes the first
                    # input to the new thresholds node
                    new_thresholds.input[0] = inp
                    # Create a new input tensor name for the thresholds
                    new_thresholds.input[1] = model.make_new_valueinfo_name()
                    # Annotate the new thresholds input with the new shape of
                    # the split thresholds
                    model.set_tensor_shape(
                        new_thresholds.input[1], thresholds[i].shape
                    )
                    # Set the initializer input to the split thresholds
                    model.set_initializer(
                        new_thresholds.input[1], thresholds[i]
                    )
                    # Create a new output tensor name
                    new_thresholds.output[0] = model.make_new_valueinfo_name()
                    # Annotate the new output with the shape of the input
                    model.set_tensor_shape(
                        new_thresholds.output[0], model.get_tensor_shape(inp)
                    )
                    # Connect the new output tensor to the corresponding input
                    # of the merge node
                    node.input[i] = new_thresholds.output[0]
                    # Connect the output of the merging node to successor of the
                    # original thresholding node
                    node.output[0] = thresholds_node.output[0]
                    # Insert the thresholding node into the graph
                    graph.node.insert(index + i - 1, new_thresholds)
                # Remove the original thresholds node
                graph.node.remove(thresholds_node)
                # Graph has been modified, required additional transformations
                # to be run
                graph_modified = True
                # Break the loop after adding and removing nodes to start over
                # with a clean index
                break
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Re-do the datatype annotations after inserting new tensors without and
        # moving tensors with existing annotations
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Detects multi-head attention pattern, i.e., scaled dot-product attention
# between head splitting and merging
def is_multi_head_attention(node: NodeProto, model: ModelWrapper):  # noqa
    # The anchor node must be scaled dot product attention
    if node.op_type == "ScaledDotProductAttention":
        # Get the nodes feeding the attention operation
        predecessors = model.find_direct_predecessors(node)
        # There must be exactly three predecessors of type head-splitting
        # Note: there must be nothing in between splitting and the attention
        # itself
        if op_types(predecessors) == 3 * ["SplitMultiHeads"]:
            # Get the node fed by the attention operation
            successors = model.find_direct_successors(node)
            # There must be exactly onde successor of type head-merging
            # Note: there must be nothing in between attention and the merging
            if op_types(successors) == 1 * ["MergeMultiHeads"]:
                # Get the shape of the input tensor for inferring the number of
                # heads and correctly propagating shapes
                shape = model.get_tensor_shape(node.input[0])
                # Determine the rank of the input tensor to support batched and
                # non-batched inputs
                rank = len(shape)
                # The input shape determines the sequence length
                heads, _, _ = shape if (rank == 3) else (1, shape[0], shape[1])
                # Pattern detected, if there are actually multiple heads
                return heads > 1
    # Pattern not detected
    return False


# Unrolls multiple attention heads in the onnx graph to be implemented in
# parallel
class UnrollMultiHeadAttention(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Apply transformation to nodes which match the multi-head attention
            # pattern
            if is_multi_head_attention(node, model):
                # Get the splitting nodes fed by the attention operation
                split0, split1, split2 = model.find_direct_predecessors(node)
                # Get the single merging node
                merge0, = model.find_direct_successors(node)
                # Get the number of heads produced by an arbitrary splitters
                heads = get_by_name(split0.attribute, "heads").i
                # Get the number of input elements to the heads splitting
                # Note: Embedding dims might actually differ per input stream,
                #   e.g., for cross-attention
                dim0 = get_by_name(split0.attribute, "num_elems").i
                dim1 = get_by_name(split1.attribute, "num_elems").i
                dim2 = get_by_name(split2.attribute, "num_elems").i
                # get the number of input features per splitting
                # Note: Feature map sizes might actually differ per input
                #   stream, e.g., for cross-attention
                ins0 = get_by_name(split0.attribute, "num_inputs").ints
                ins1 = get_by_name(split1.attribute, "num_inputs").ints
                ins2 = get_by_name(split2.attribute, "num_inputs").ints
                # Validate the number of heads matches between all slice and
                # merge nodes
                for n in [split0, split1, split2, merge0]:
                    # All heads must match, otherwise this is a failure from
                    # which we cannot recover
                    assert get_by_name(n.attribute, "heads").i == heads, \
                        f"Differing number of heads at {node.name} and {n.name}"
                    # Remove the original node from the graph
                    graph.node.remove(n)

                # TODO: Clean up the following code

                # Create replicas of the splitting nodes with expanded output
                # list
                split0 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SplitMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    backend="fpgadataflow",
                    # Connect to the same input as the original
                    inputs=split0.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False,
                    # Datatype of inputs and outputs
                    dtype=get_by_name(split1.attribute, "dtype").s,
                    # Number of input elements, i.e., embedding dimension
                    num_elems=dim0,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    num_inputs=[*ins0]
                )
                split1 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SplitMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    backend="fpgadataflow",
                    # Connect to the same input as the original
                    inputs=split1.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False,
                    # Datatype of inputs and outputs
                    dtype=get_by_name(split1.attribute, "dtype").s,
                    # Number of input elements, i.e., embedding dimension
                    num_elems=dim1,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    num_inputs=[*ins1]
                )
                split2 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SplitMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    backend="fpgadataflow",
                    # Connect to the same input as the original
                    inputs=split2.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False,
                    # Datatype of inputs and outputs
                    dtype=get_by_name(split2.attribute, "dtype").s,
                    # Number of input elements, i.e., embedding dimension
                    num_elems=dim2,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    num_inputs=[*ins2]
                )
                # Create replica of the merging node with expanded input list
                merge0 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="MergeMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    backend="fpgadataflow",
                    # Generate new input tensor names for each head
                    inputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Connect to the same input as the original
                    outputs=merge0.output,
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Attribute specifying whether the output needs to be
                    # squeezed
                    squeezed=get_by_name(merge0.attribute, "squeezed").i,
                    # Unrolled heads do not produce packed tensors
                    packed=False,
                    # Datatype of inputs and outputs
                    dtype=get_by_name(merge0.attribute, "dtype").s,
                    # Number of input elements, i.e., embedding dimension
                    num_elems=get_by_name(merge0.attribute, "num_elems").i,
                    # Number of embeddings in the whole input sequence/feature
                    # map
                    # Note: Drop head-first head dimension of previously packed
                    # input
                    num_inputs=get_by_name(
                        merge0.attribute, "num_inputs").ints[1:]
                )

                # Replicate the attention operator for each head
                for i in range(heads):
                    # Start by making a full copy of the original node
                    attention = copy.deepcopy(node)
                    # Get the original shape of each input to remove the head
                    # number
                    _, seq, dim = model.get_tensor_shape(attention.input[0])
                    model.set_tensor_shape(split0.output[i], (1, seq, dim))
                    _, seq, dim = model.get_tensor_shape(attention.input[1])
                    model.set_tensor_shape(split1.output[i], (1, seq, dim))
                    _, seq, dim = model.get_tensor_shape(attention.input[2])
                    model.set_tensor_shape(split2.output[i], (1, seq, dim))

                    # Propagate the original datatype to each of the head inputs
                    dtype = model.get_tensor_datatype(attention.input[0])
                    model.set_tensor_datatype(split0.output[i], dtype)
                    dtype = model.get_tensor_datatype(attention.input[1])
                    model.set_tensor_datatype(split1.output[i], dtype)
                    dtype = model.get_tensor_datatype(attention.input[2])
                    model.set_tensor_datatype(split2.output[i], dtype)

                    # Connect the inputs of the replica to the output of each
                    # of the new slice operators
                    attention.input[0] = split0.output[i]
                    attention.input[1] = split1.output[i]
                    attention.input[2] = split2.output[i]

                    # Get the original shape the output to remove the head
                    # number
                    _, seq, dim = model.get_tensor_shape(attention.output[0])
                    model.set_tensor_shape(merge0.input[i], (1, seq, dim))

                    # Propagate the original datatype to each of the head
                    # outputs
                    dtype = model.get_tensor_datatype(attention.output[0])
                    model.set_tensor_datatype(merge0.input[i], dtype)

                    # Connect the output of the attention replica to the input
                    # of the new merge operator
                    attention.output[0] = merge0.input[i]
                    # Insert the new node into the graph
                    graph.node.insert(index + i + 1, attention)
                # Insert the new slice and merge nodes into the graph
                for i, n in enumerate([split0, split1, split2, merge0]):
                    # Insert the new node into the graph at index offset by
                    # number of heads
                    graph.node.insert(index + heads + i + 1, n)
                # Remove the original attention operator from the graph
                graph.node.remove(node)
                # The graph has been modified, needs to be reported back to the
                # caller
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows model
        # By replicating the attention operator, multiple instances refer to the
        # same initializer, replace these by a unique one for each head
        model = model.transform(GiveUniqueParameterTensors())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

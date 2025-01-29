# ADAPTED FROM Christoph's radioml-transformer repository, specifically these files:
# build_steps.py
# custom/apply_config.py
# custom/composed_transformation.py
# custom/streamline.py

# Python warning messages
import warnings
# Copies of python objects
from copy import deepcopy
# Copies (deep-copies) python objects
import copy
# Numpy for loading and comparing the verification input/output
import numpy as np
# YAML for loading experiment configurations
import yaml

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Range information structure for seeding the range analysis for converting
# quantized activations to MultiThreshold
from qonnx.util.range_analysis import RangeInfo

# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    Transformation,
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# If we have a convolution with a bias tensors input, QONNX and later FINN
# expect the bias to be expressed as a standalone Add node following the Conv
# node.
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
# Converts Gemm operation to MatMul with extracted standalone bias op
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
# Converts Conv to Im2Col and MatMul with extracted standalone bias op
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
# Transposes the initializer tensors of a Quant node instead of having a
# standalone Transpose following
from qonnx.transformation.quant_constant_folding import (
    FoldTransposeIntoQuantInit
)
# Collapses chains of constants into a single constant operation or even
# initializer tensors.
from qonnx.transformation.fold_constants import FoldConstants
# Folds quantizers into weight tensor initializers, needed for lowering
# convolutions to MatMuls
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
# FINN streamlining transformations reordering the graph
from finn.transformation.streamline.reorder import (
    MoveTransposePastFork,
    MoveTransposePastEltwise,
    MoveTransposePastJoinMul,
    MoveTransposePastJoinAdd,
    MoveTransposePastSplit,
    MoveTransposePastJoinConcat,
    MoveSqueezePastMultiThreshold,
    MoveSqueezePastMatMul
)
# FINN streamlining transformations absorbing tensors/nodes into others
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)
# FINN streamlining transformations fusing/collapsing operations of the same
# kind
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedTranspose
)
# FINN streamlining transformations removing nodes without real effect from the
# graph
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape,
    RemoveIdentityOps
)
# Cleanup transformation getting rid of 3d data layout
from finn.transformation.squeeze import Squeeze
# Detects the attention pattern and converts to hardware custom op
from finn.transformation.fpgadataflow.attention import (
    InferScaledDotProductAttention,
    AbsorbMultiThresholdIntoScaledDotProductAttention
)
# Mult-Head Attention support
from finn.transformation.fpgadataflow.attention_heads import (
    InferMultiHeads,
    UnrollMultiHeadAttention,
    MoveSplitMultiHeadsPastMultiThreshold,
    MoveMergeMultiHeadsPastMultiThreshold
)
# Converts (infers) ONNX and QONNX nodes to FINN hardware CustomOps
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferSqueeze,
    InferUnsqueeze,
    InferElementwiseBinaryOperation,
    InferSplitLayer,
    InferConcatLayer,
    InferLookupLayer,
    InferVectorVectorActivation
)
# Converts fork-nodes to ReplicateStream hardware operator
from finn.transformation.fpgadataflow.replicate_stream import (
    InferReplicateStream
)
# Standard QONNX to FINN conversion function
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
# QONNX quantization data types
from qonnx.core.datatype import DataType
# Converts ONNX graph nodes to QONNX custom-ops if possible
from qonnx.custom_op.registry import getCustomOp
# Inserts data-width converter and FIFO nodes into the model graph
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
# Splitting and removing of FIFOs from the model graph
from finn.transformation.fpgadataflow.set_fifo_depths import (
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
)
# Specializes each layer's implementation style: HLS or RTL implementation
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
# FINN dataflow builder configuration
from finn.builder.build_dataflow_config import (
    VerificationStepType, DataflowBuildConfig
)
# Graph transformation setting the folding, i.e., parallelization configuration
from finn.transformation.fpgadataflow.set_folding import SetFolding
# FINN verification after build/graph transformation steps
from finn.builder.build_dataflow_steps import verify_step

# Transformations preparing the operators for synthesis and simulation
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# Execute onnx model graphs from the dataflow parent for verification
from finn.util.test import execute_parent

# Base class for all QONNX graph transformations and some basic cleanup
# transformations
from qonnx.transformation.general import (
    Transformation,
    ConvertDivToMul,
    ConvertSubToAdd,
)

# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine

# Groups node inputs by dynamic vs. initializer category
from finn.transformation.streamline.absorb import group_inputs_by_category

# FINN streamlining transformations converting and rounding values
from finn.transformation.streamline import (
    ConvertSignToThres,
    RoundAndClipThresholds
)
# FINN streamlining transformations reordering the graph
from finn.transformation.streamline.reorder import (
    MoveMulPastFork,
    MoveTransposePastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants,
    MoveTransposePastEltwise,
    MoveMulPastMaxPool,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastMatMul,
    MoveScalarMulPastConv,
    MoveTransposePastJoinMul,
    MoveTransposePastJoinAdd,
    MoveMulPastJoinAdd,
    MoveAddPastJoinAdd,
    MoveScalarLinearPastSplit,
    MoveAffinePastJoinConcat,
    MoveMulPastJoinConcat,
    MoveAddPastJoinConcat,
    MoveTransposePastSplit,
    MoveTransposePastJoinConcat,
    MoveSqueezePastMultiThreshold,
    is_scalar
)
# FINN streamlining transformations absorbing tensors/nodes into others
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbMulIntoMultiThreshold,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbTransposeIntoMultiThreshold
)
# FINN streamlining transformations fusing/collapsing operations of the same
# kind
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedTranspose,
    CollapseRepeatedAdd
)
# FINN streamlining transformations removing nodes without real effect from the
# graph
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
)

# Composes graph transformations such that each individual transformation as
# well as the whole sequence is applied exhaustively
class ComposedTransformation(Transformation):
    # Initializes the transformation given a list of transformations
    def __init__(self, transformations: list[Transformation]):
        # Initialize the transformation base class
        super().__init__()
        # Register the list of transformations to be applied in apply()
        self.transformations = transformations

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all transformations to be applied
        for transformation in self.transformations:
            # Start each transformation on a deep copy of the model to mimic the
            # behavior of ModelWrapper.transform()
            model = copy.deepcopy(model)
            # Exhaustively apply the transformation until it no longer modifies
            # the graph
            while True:
                # Apply the transformation once, reporting back whether any node
                # or pattern has been modified
                model, _graph_modified = transformation.apply(model)
                # Keep track whether the graph has been modified at least once
                graph_modified = graph_modified or _graph_modified
                # Break the loop if this transformation did not change anything
                if not _graph_modified:
                    break
            # Apply the cleanup transformations of the ModelWrapper
            model.cleanup()
            # Apply some further cleanup transformations to the model graph
            # removing some clutter and keeping all names readable and ordered
            # at any time
            model = model.transform(RemoveIdentityOps())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed by at least one transformation so the whole
        # sequence of transformations will be reapplied
        return model, graph_modified

# # Custom conversion from Quant to MultiThreshold
# TODO: Enable once fixed...
# from custom.quant_activation_to_multithreshold import (
#     QuantActivationToMultiThreshold
# )

# Moves scale factor, i.e., scalar Mul and Div, past Im2Col (and Col2Im): These
# cannot be handled by MoveScalarLinearPastInvariants as potential padding makes
# Add-Im2Col not commute to Im2Col-Add
class MoveScalesPastIm2Col(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type in {"Mul", "Div"}:
                # Cannot handle fork- or join-multiplications
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # The first input must be dynamically received from upstream
                if model.get_initializer(node.input[0]) is not None:
                    # Softly skip this node
                    continue
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(node.input[1])):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If this is the final operation in the graph, there might be no
                # successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Handle both, Im2Col and the inverse Col2Im, as well as padding
                if successor.op_type in {"Im2Col", "Col2Im", "Pad"}:
                    # Get names of all tensors involved in connecting the
                    # nodes
                    inp = node.input[0]  # noqa: Duplicate
                    mid = node.output[0]
                    out = successor.output[0]
                    # Rewire the graph to feed original input into the
                    # Add node first
                    successor.input[0] = inp
                    # Repurpose the middle tensor for the output of the Add
                    successor.output[0] = mid
                    # The Mul operator now gets the middle tensor as its
                    # input
                    node.input[0] = mid
                    # Mul now produces the original output tensor
                    node.output[0] = out
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later
                    model.set_tensor_shape(mid, None)
                    model.set_tensor_shape(out, None)
                    # Track whether the graph has been modified, never
                    # resets to False
                    graph_modified = True
                    # Break the loop after deleting shape annotations to
                    # immediately re-do these before changing the next
                    # operator
                    break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified

# Moves scalar linear elementwise operations past fork nodes, applies to Add,
# Mul, Sub, Div, etc.
class MoveScalarLinearPastFork(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul-like and Add-like operation types
            if node.op_type in {"Add", "Sub", "Mul", "Div"}:
                # Only handles non-joining forks for now
                if not model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(node.input[1])):
                    # Softly skip this node
                    continue
                # We need to insert a replica of this operation in front of each
                # consumer node
                for consumer in model.find_direct_successors(node):
                    # Create an exact replica of this operator
                    copy = deepcopy(node)
                    # Insert a new unique tensor connecting the output of the
                    # copy to the consumer
                    copy.output[0] = model.make_new_valueinfo_name()
                    # The original node might be connecting to multiple inputs
                    # of the consumer...
                    for idx, inp in enumerate(consumer.input):
                        # Find each instance of connection from original node
                        if inp == node.output[0]:
                            # Rewire to connect to the replica
                            consumer.input[idx] = copy.output[0]
                    # Insert the new replica node into the graph
                    graph.node.insert(index + 1, copy)
                # Remove the original node from the graph
                graph.node.remove(node)
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified

# Moves constant elementwise multiplication past another joining multiplication
class MoveConstMulPastJoinMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to Multiplications
                if successor.op_type in {"Mul"}:
                    # Applies only if the second multiplication is a join-node
                    if model.is_join_node(successor):
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Need to match the correct input of the joining second
                        # multiplication
                        for i, name in enumerate(successor.input):
                            # If the successors input currently matches the
                            # intermediate tensors, this input needs to be
                            # rewired
                            if name == mid:
                                # Rewire the graph to feed original into the
                                # second Mul node first
                                successor.input[i] = inp
                                # Note: Do not break here as it is perfectly
                                # legal to connect the same tensor multiple
                                # times to different inputs
                        # Repurpose the middle tensor for the output of the
                        # second Mul
                        successor.output[0] = mid
                        # The first Mul operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # The first Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified
    
# Moves elementwise additions past MatMul operations: Applicable if each
# operation has one initializer input
class MoveAddPastMatMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Add operations
            if node.op_type == "Add":
                # If the add is a join operation, we do not have a constant
                # added to the input
                if model.is_join_node(node):
                    # Skip transforming this
                    continue
                # If the Add is a fork operation we should first distribute the
                # Add into the branches
                if model.is_fork_node(node):
                    # Issue a warning to make the use aware of this potential
                    # transformation if the fork is moved first
                    warnings.warn(
                        f"{self.__class__.__name__}:"
                        f" Skipping near match: {node.name} is a fork-node,"
                        f" try MoveLinearPastFork first"
                    )
                    # Skip transforming this node as moving this would lead
                    # to messed up or detached graph
                    continue
                # Decompose the inputs into the dynamic and the constant
                # initializer input
                (x_name,), (c_name,) = group_inputs_by_category(node, model)
                # Now check the successor node which must be a MatMul
                consumer = model.find_direct_successors(node)
                # If there is no consumer, this Add seems to be last node of the
                # graph
                if not consumer:
                    # Skip transforming this
                    continue
                # There must be exactly one consumer now
                consumer = consumer[0]
                # This transformation only applies to Add in front of MatMul
                if not consumer.op_type == "MatMul":
                    # Skip this if not MatMul
                    continue
                # MatMul may not be a join operation to apply this
                # transformation
                if model.is_join_node(consumer):
                    # Skip transforming without warning (there is nothing we can
                    # do about this)
                    continue
                # Decompose the inputs to the MatMul to get the weight tensor
                # name (the other input is the output of the Add)
                _, (w_name,) = group_inputs_by_category(consumer, model)
                # Read the weights and the constant addition tensor
                w = model.get_initializer(w_name)
                c = model.get_initializer(c_name)
                # Determine whether the weights are the left or right input to
                # the MatMul
                left = w_name == consumer.input[0]
                # Apply the weights to the constant tensor
                c = np.matmul(w, c) if left else np.matmul(c, w)
                # Insert the transformed tensor back into the mode as an
                # initializer
                model.set_initializer(c_name, c)
                # The connecting tensors of this pattern
                inp = x_name
                mid = node.output[0]
                out = consumer.output[0]
                # Rewire the graph pattern connecting the input to the MatMul
                # and the MatMul output to the Add node
                consumer.input[1 if left else 0] = inp
                # The Add now produces the original MatMul output
                node.output[0] = out
                # The middel tensor connects to the Add input
                node.input[0 if node.input[0] == x_name else 1] = mid
                # The MatMul feeds the middle tensors
                consumer.output[0] = mid
                # Delete the shape annotation of the connecting tensors
                # to be re-done later
                model.set_tensor_shape(mid, None)
                model.set_tensor_shape(out, None)
                # Delete the type annotations of the connecting tensors
                # to be re-done later
                # model.set_tensor_datatype(mid, None)
                # model.set_tensor_datatype(out, None)
                # Track whether the graph has been modified, never
                # resets to False
                graph_modified = True
                # Break the loop after deleting shape annotations to
                # immediately re-do these before changing the next
                # operator
                break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves elementwise multiplication past elementwise addition if one input to
# each of the operators is a known constant
# Note: Reverse of MoveAddPastMul
class MoveMulPastAdd(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to additions
                if successor.op_type in {"Add"}:
                    # The addition may not join as we need to know the second
                    # input
                    if not model.is_join_node(successor):
                        # Get the constant initializer tensors for both
                        # operations: y = s * x + b
                        _, s_name = group_inputs_by_category(node, model)
                        _, b_name = group_inputs_by_category(successor, model)
                        # Skip if either node has no constant initializer
                        if not s_name or not b_name:
                            # Skip without warning ok?
                            continue
                        # There must be exactly one constant per operations
                        assert len(s_name) == 1, \
                            f"To many constant inputs for {node}"
                        assert len(b_name) == 1, \
                            f"To many constant inputs for {successor}"
                        # Now read the initializer tensors
                        s = model.get_initializer(*s_name)
                        b = model.get_initializer(*b_name)
                        # Update the addition initializer according to the
                        # distributive law
                        model.set_initializer(*b_name, b / s)
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Rewire the graph to feed original input into the
                        # Add node first
                        successor.input[0] = inp
                        # Repurpose the middle tensor for the output of the Add
                        successor.output[0] = mid
                        # The Mul operator now gets the middle tensor as its
                        # input
                        node.input[0] = mid
                        # Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified

# Define a set of custom streamlining transformations: These are applied once
# during the actual streamlining step and once after converting attention to
# hardware (the associated cleanup afterward might enable some Streamlining
# transformations once again)
def Streamline():  # noqa: Uppercase
    # Return a set of exhaustively applies transformations
    return ComposedTransformation([
        # On skip-connections: prefer pushing scalar multiplication forward
        # before MoveAddPastMul
        MoveMulPastFork(),
        # The "standard" set of FINN streamlining transformations or at least
        # inspired by them but applied exhaustively until none of them changes
        # the graph anymore.
        # Note: Covers most parts of non-branching linear topologies
        ComposedTransformation([
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
        ]),
        # Streamlining scales and biases forward through residual topologies
        # Note: This mostly covers forking and joining operations
        ComposedTransformation([
            # Note: This is probably the most common way of joining skip
            # connections, i.e., this corresponds to the original residual
            # addition, i.e., y = f(x) + x
            MoveLinearPastEltwiseAdd(),
            MoveScalarLinearPastFork(),
            MoveScalarLinearPastInvariants(),
            MoveMulPastFork(),
            MoveMulPastJoinAdd(),
            MoveAddPastJoinAdd(),
            # Note: This brings constant Muls (i.e., quantizer scales to be
            # removed) forward through joining Muls (i.e., those ending up
            # as actual hardware operators).
            MoveConstMulPastJoinMul()
        ]),
        # Streamlining scales and biases forward through shape/layout changing
        # operations, i.e., mostly transposes
        ComposedTransformation([
            # Convolution inputs and padding
            MoveScalesPastIm2Col(),
            # Streamlining for Split and Concat operations
            MoveScalarLinearPastSplit(),
            MoveAffinePastJoinConcat(),
            MoveMulPastJoinConcat(),
            MoveAddPastJoinConcat(),
            # Move transposes around to some place where they could be removed
            # later, i.e., where they collapse into identities
            MoveTransposePastFork(),
            MoveTransposePastSplit(),
            MoveTransposePastJoinConcat(),
            MoveTransposePastEltwise(),
            MoveTransposePastJoinMul(),
            MoveTransposePastJoinAdd(),
            CollapseRepeatedTranspose(),
            # Remove identity shape/layout transformations
            RemoveIdentityTranspose(),
            RemoveIdentityReshape(),
            # Squeeze operators can be moved past the thresholding
            MoveSqueezePastMultiThreshold(),
            # A certain type of 4d-layout transpose can be absorbed (actually
            # moved past) MultiThreshold operations
            AbsorbTransposeIntoMultiThreshold(),
        ]),
        # Only round and clip after all streamlining transformations have
        # been applied exhaustively.
        # Note: Might still enable another round of streamlining.
        RoundAndClipThresholds(),
    ])


# Prepares the graph to be consumed by FINN:
# 1. Some graph cleanup removing unused tensors, nodes without effect and
#  folding constants, i.e., collapsing chains of operations on constant tensors
# 2. Lowers some "more complex" operations: converts Conv and Gemm to MatMul and
#  BatchNorm to Mul and Add operations followed by some necessary cleanup
# 3. Converts all QONNX Quant nodes to MultiThreshold operations which can
#  absorb scales and biases during streamlining
def prepare_graph(range_info: RangeInfo):
    # Wrap the actual transformation/build step function
    def step_prepare_graph(model: ModelWrapper, cfg: DataflowBuildConfig):
        # Exhaustively apply the set of cleanup transformations
        model = model.transform(ComposedTransformation([
            # Adds shape and datatype annotations to all tensors in this graph
            InferDataTypes(),
            InferShapes(),
            # Cleanup the graph by removing redundant, unnecessary and constant
            # nodes and tensors and give unique names to everything remaining
            GiveUniqueNodeNames(),
            GiveReadableTensorNames(),
            RemoveStaticGraphInputs(),
            RemoveUnusedTensors(),
            GiveUniqueParameterTensors(),
            FoldConstants(),
            # Remove unnecessary shape and layout transformations
            RemoveIdentityReshape(),
            RemoveIdentityTranspose(),
            # Redo shape and datatype annotations after removing nodes and
            # tensors
            InferShapes(),
            InferDataTypes(),
        ]))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.TIDY_UP_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "tidied_up_python", need_parent=False
            )
        # Exhaustively apply the lowering transformations
        model = model.transform(ComposedTransformation([
            # Moves the bias input to the Conv operator as a separate Add node
            # behind the Conv node
            ExtractBiasFromConv(),
            # Converts Gemm nodes to MatMul (+ bias)
            GemmToMatMul(),
            # Need to do some constant and weight folding first
            FoldConstants(),
            FoldTransposeIntoQuantInit(),
            FoldQuantWeights(),
            # Annotate the graph with shape and data type information
            InferShapes(),
            InferDataTypes(),
            # Converts Conv layers to MatMul
            LowerConvsToMatMul(),
            # Converts BatchNorm to affine scale and bias
            BatchNormToAffine(),
            # Annotate the graph with shape and data type information
            InferShapes(),
            InferDataTypes(),
        ]))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "lowered_python", need_parent=False
            )
        # Apply the quantizer to MultiThreshold conversion
        # Note: This is exhaustive as well as single .transform reapplies as
        # long as possible.
        # TODO: Enable once fixed...
        # model = model.transform(QuantActivationToMultiThreshold(range_info))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "quant_to_thresholds_ra_python", need_parent=False
            )
        # Apply the standard QONNX to FINN conversion step to convert the
        # remaining quantizers not yet covered by the new range analysis based
        # method
        model = model.transform(ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=cfg.max_multithreshold_bit_width
            )
        ))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "prepared_graph_python", need_parent=False
            )
        # Return the transformed model
        return model

    # Return the wrapped transformation step function
    return step_prepare_graph


# Applies the custom set of exhaustive streamlining transformations, also taking
# special topology like attention, residuals, splits and transposes into account
def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # These should not be applied exhaustively with the other streamlining
    # transformations to not end up in cycles.
    # Note: This is essential to allow some Add operations to be
    # absorbed by the next round's AbsorbSignBiasIntoMultiThreshold
    model = model.transform(MoveMulPastAdd())
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    # Exhaustively apply the following set of transformations to streamline the
    # graph with the overall goal of collecting scales and biases in front of
    # MultiThreshold operations or, alternatively, at the end of the graph.
    # Note: Contains some sets of nested exhaustive transformations meant for
    # particular architectural patterns, e.g., residual topologies.
    model = model.transform(Streamline())
    # If configured, run a verification of the transformed model on some
    # sample inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_python", need_parent=False
        )
    # Return the transformed model
    return model


# Converts scaled dot-product attention operations to FINN hardware operations
# Note: This includes some necessary cleanup after converting the pattern, in
# particular squeezing the data layouts throughout the graph
def step_convert_attention_to_hw(model: ModelWrapper, _: DataflowBuildConfig):
    # Try to infer reshaping of attention heads
    model = model.transform(InferMultiHeads())  # noqa: Duplicate
    # Try to mode the mult-head splitting past the multi thresholds
    model = model.transform(MoveSplitMultiHeadsPastMultiThreshold())
    # Moving multi-head splitting past multi thresholds might enable absorbing
    # adds into thresholds once again
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # Try to infer a ScaledDotProductAttention custom op
    model = model.transform(InferScaledDotProductAttention())
    # Parallelize attention head in the onnx graph
    model = model.transform(UnrollMultiHeadAttention())
    # Swap the order of merging the multi heads and applying thresholds
    model = model.transform(MoveMergeMultiHeadsPastMultiThreshold())
    # If applicable, absorb the final thresholds into the attention operator
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())
    # Squeeze (i.e., remove dimensions of size 1) the data layouts throughout
    # the graph to treat the time dimension as the batch dimension for all MVU
    # and Threshold operators
    model = model.transform(Squeeze())
    # Squeezing might have turned further transpose and reshape operations into
    # identities (those which just swapped around the dimensions of size 1)
    model = model.transform(ComposedTransformation([
        # Move transposes around to some place where they could be removed
        # later, i.e., where they collapse into identities
        MoveTransposePastFork(),
        MoveTransposePastSplit(),
        MoveTransposePastJoinConcat(),
        MoveTransposePastEltwise(),
        MoveTransposePastJoinMul(),
        MoveTransposePastJoinAdd(),
        CollapseRepeatedTranspose(),
        # Remove identity shape/layout transformations
        RemoveIdentityTranspose(),
        RemoveIdentityReshape(),
        # Squeeze operators can be moved past MatMuls and thresholding
        MoveSqueezePastMatMul(),
        MoveSqueezePastMultiThreshold(),
    ]))
    # Squeezing might enable absorbing adds into thresholds once again
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # If applicable, absorb the final thresholds into the attention operator
    #   Note: Might be applicable again after squeezing a transpose away
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())
    # We should do another round of streamlining to be sure and support more
    # general architectural patterns, we are not aware of yet...
    model = model.transform(Streamline())
    # Convert Squeeze and Unsqueeze operators to hardware operations
    model = model.transform(InferSqueeze())
    model = model.transform(InferUnsqueeze())
    # Return the model with attention and multi-heads mapped to hardware
    # operators
    return model


# Function running the transformations to convert elementwise binary operations
# to their hardware implementations
def step_convert_elementwise_binary_to_hw(model: ModelWrapper, _):
    # Convert elementwise operations to hardware operators
    #   Note: Do not convert the final Mul operator at the output
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_output_dequant
    ))


# Converts Split and Concat operations to hardware custom operators
def step_convert_split_concat_to_hw(model: ModelWrapper, _):
    return model.transform(InferSplitLayer()).transform(InferConcatLayer())


# Function running the transformations to convert Gather, i.e., index lookup,
# nodes to their hardware implementations
def step_convert_lookup_to_hw(model: ModelWrapper, _):
    # Iterate all nodes in the graph keeping track of the index
    for index, node in enumerate(model.graph.node):
        # If this is a Gather node, force the input (index) type annotation
        if node.op_type == "Gather":
            # Force to unsigned 64-bit integer for now
            model.set_tensor_datatype(node.input[1], DataType["UINT64"])
            # Get the value info for the input tensor to have access to the ONNX
            # datatype of the tensor
            value_info = model.get_tensor_valueinfo(node.input[1])
            # Force the container datatype of the input to be a float
            value_info.type.tensor_type.elem_type = 1
    # Convert Gather to Lookup layers
    return model.transform(InferLookupLayer())


# Converts depth-wise convolution to hardware operator calling the
# InferVectorVectorActivation transformation
def step_convert_depth_wise_to_hw(model: ModelWrapper, _: DataflowBuildConfig):
    return model.transform(InferVectorVectorActivation())


# Function running the InferReplicateStream transformation
def step_replicate_streams(model: ModelWrapper, _):
    # Properly replicate the stream feeding the query, key and value projections
    return model.transform(InferReplicateStream())


# Custom step for setting the parallelism to meet the target of T^2 cycles per
# sequence
def set_target_parallelization(seq_len: int,
                               emb_dim: int):  # noqa: emb_dim
    # The wrapping function is a generator and this is the actual build step
    # function taking the model and build configuration
    def step_set_target_parallelization(
            model: ModelWrapper, cfg: DataflowBuildConfig
    ):
        # Run over all nodes in the model graph to look for attention operators,
        # which are currently not handled by the SetFolding transformation
        for index, node in enumerate(model.graph.node):
            # Only handle attention operations here
            if node.op_type == "ScaledDotProductAttention_hls":
                # Convert this to the custom-op instance for easy access to node
                # attributes
                inst = getCustomOp(node)
                # Set the sequence and embedding dimension folding to meet the
                # T^2 cycles target, i.e., fully parallel along the embedding
                # dimension and fully sequential along the sequence dimension
                inst.set_nodeattr("EmbFold", 1)
                inst.set_nodeattr("SeqFold", seq_len)
        # Apply the built-in folding configuration transformation with the
        # T^2 target cycles
        model = model.transform(SetFolding(
            seq_len ** 2, cfg.mvau_wwidth_max, cfg.folding_two_pass_relaxation
        ))
        # TODO: Extract the folding configuration
        # Return the model with configured parallelization
        return model

    # Return the wrapped build step function
    return step_set_target_parallelization


# Applies configuration dictionary to the model graph
class ApplyConfig(Transformation):
    # Initializes the transformation with the configuration dictionary
    def __init__(self, config):
        # Initialize the transformation base class
        super().__init__()
        # Register the configuration dictionary to be used in apply()
        self.config = config

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # A node should not be named "defaults"...
            assert node.name != "defaults", \
                "Node has reserved name 'defaults'"
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Apply the per operator type default configurations to the node
            if node.op_type in self.config["defaults"]:
                # Run over all default options to be applied to this node
                for key, value in self.config["defaults"][node.op_type].items():
                    # Set the nodes attribute to the default option value
                    inst.set_nodeattr(key, value)
            # If there is an individual, node-specific configuration apply
            # this next, potentially overriding the defaults set above
            if node.name in self.config:
                # Run over all node-specific options to be applied to this
                # node
                for key, value in self.config[node.name].items():
                    # Set the nodes attribute to the option value
                    inst.set_nodeattr(key, value)
        # Return model with configuration applied
        # Note: Do not consider this as modifying the graph. This does not have
        # to be reapplied multiple times.
        return model, False


# Custom build step trying to set appropriate FIFO sizes for the transformer
def set_fifo_depths(
        seq_len: int, emb_dim: int, uram_threshold: int = 32  # noqa: emb_dim
):
    # The wrapping function is a generator and this is the actual build step
    # function taking the model and build configuration
    def step_set_fifo_depths(model: ModelWrapper, cfg: DataflowBuildConfig):
        # Run over all nodes in the model graph
        for index, node in enumerate(model.graph.node):
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Extract the FIFO depths configuration of the node
            in_depths = inst.get_nodeattr("inFIFODepths")
            out_depths = inst.get_nodeattr("outFIFODepths")

            # Number of inputs and outputs to/from the node
            num_inputs = len(node.input)
            num_outputs = len(node.output)

            # If the input/output has only default configurations, fill with as
            # many shallow FIFOs as there are inputs, to avoid later problems
            # with to few FIFO depths specified
            if in_depths == [2] and num_inputs > 1:
                in_depths = num_inputs * [2]
            if out_depths == [2] and num_outputs > 1:
                out_depths = num_outputs * [2]

            # Special case: Attention needs properly sized input FIFOs
            if node.op_type == "ScaledDotProductAttention_hls":
                # Each folded input stream needs to be buffered completely
                # TODO: Not exactly sure whether this is always correct or just
                #  the worst-case
                in_depths = [
                    inst.get_number_input_values(i) for i in range(num_inputs)
                ]
                # Note: No special treatment of the output FIFO
                # out_depths = ...

            # Special case: Adding residual branches needs to buffer the inputs
            # to avoid deadlocks if one branch is running faster/slower
            if node.op_type == "ElementwiseAdd_hls":
                # Only relevant if for join-node operations, i.e., node actually
                # consumes two branches, potentially operating at a different
                # rate
                if model.is_join_node(node):
                    # Set both inputs to buffer as many cycles as we target for
                    # the attention operations, i.e., the T^2 cycles per
                    # sequence target
                    # TODO: Not exactly sure whether this is always correct or
                    #  just the worst-case
                    # TODO: Currently we do not really have a reliable way of
                    #  figuring out which of the two is the longer/deeper branch
                    #  in terms of cycles to set a corresponding buffer only to
                    #  the shorter branch.
                    in_depths = [seq_len ** 2, seq_len ** 2]
                    # Note: No special treatment of the output FIFO
                    # out_depths = ...

            # Set the updated FIFO depths attributes
            inst.set_nodeattr("inFIFODepths", in_depths)
            inst.set_nodeattr("outFIFODepths", out_depths)

        # The following partially mirrors (or even copies from) the build-in
        # step_set_fifo_depths using only manual FIFO depths and our YAML-based
        # folding configuration.

        # Insert data-width converters
        model = model.transform(InsertDWC())
        # Insert FIFOs between all operators (inserts shallow, depths 2 FIFOs if
        # no other depth is specified)
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        # Specialize the implementation variant of the (newly added FIFO) layers
        model = model.transform(
            SpecializeLayers(cfg._resolve_fpga_part())  # noqa: Access _ method
        )
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # Only applies if a configuration file is given
        if cfg.folding_config_file is not None:
            # Load the configuration dictionary form YAML file
            with (open(cfg.folding_config_file, "r") as file):
                # Load YAML string
                config = yaml.safe_load(file)
                # Assign unique names to the nodes which can be matched by
                # individual per-node configuration options
                model = model.transform(GiveUniqueNodeNames())
                # Apply the configuration dictionary to the model graph
                model = model.transform(ApplyConfig(config))

        # Run over all nodes in the model graph once again to modify the
        # inserted FIFOs
        # Note: This overwrites the folding configuration...
        # TODO: Find a better way to handle this
        for index, node in enumerate(model.graph.node):
            # Modify all RTL FIFO operators
            if node.op_type == "StreamingFIFO_rtl":
                # Convert this to the custom-op instance for easy access to node
                # attributes
                inst = getCustomOp(node)
                # Check the depth of the FIFO: If this is not a shallow FIFO,
                # implement this via the vivado strategy in URAM
                if inst.get_nodeattr("depth") >= uram_threshold:
                    # Change the implementation style to vivado
                    inst.set_nodeattr("impl_style", "vivado")
                    # Set the resource type for the memory to URAM
                    inst.set_nodeattr("ram_style", "ultra")

        # Hardware attributes to be extracted from each node
        hw_attrs = {
            "PE",
            "SIMD",
            "parallel_window",
            "ram_style",
            "ram_style_thresholds",
            "ram_style_mask",
            "depth",
            "impl_style",
            "resType",
            "mac_resource",
            "mem_mode",
            "runtime_writeable_weights",
            "inFIFODepths",
            "outFIFODepths",
            "depth_trigger_uram",
            "depth_trigger_bram",
        }

        # Start collecting the configuration from the model graph as a
        # dictionary
        config = {"defaults": {}}
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(model.graph.node):
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Prepare the node-specific configuration entry for this node
            config[node.name] = {}
            # Collect attribute values for all specified hardware attributes
            for key in hw_attrs:
                # Some hardware attributes may not be present for all nodes or
                # op-types, this will be signaled via exception
                try:
                    # Try extracting the configuration value from the node
                    # custom-op instance
                    config[node.name][key] = inst.get_nodeattr(key)
                # Missing attributes are signaled va AttributeError
                except AttributeError:
                    # Can be safely ignored here
                    pass
            # Cleanup: If no attribute is present for this node, there is no
            # need to keep this in the configuration dictionary as there is
            # nothing to be restored later
            if not config[node.name]:
                # Remove the entry form the configuration dictionary
                del config[node.name]

        # Create/Open a YAML file to store the configuration for later reuse
        with open(cfg.output_dir + "/final_hw_config.yaml", "w") as file:
            # Store the configuration dictionary as YAML code
            yaml.safe_dump(config, file)

        # Perform FIFO splitting and shallow FIFO removal only after the final
        # config file has been written. Otherwise, since these transforms may
        # add/remove FIFOs, we get name mismatch problems when trying to reuse
        # the final config.
        if cfg.split_large_fifos:
            model = model.transform(SplitLargeFIFOs())
        model = model.transform(RemoveShallowFIFOs())

        # After FIFOs are ready to go, call PrepareIP and HLSSynthIP again
        # this will only run for the new nodes (e.g. FIFOs and DWCs)
        model = model.transform(
            PrepareIP(
                cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()  # noqa
            )
        )
        model = model.transform(HLSSynthIP())

        # Return the model with configured parallelization
        return model

    # Return the wrapped build step function
    return step_set_fifo_depths


# Custom step applying our custom format of folding configuration to the graph
def step_apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Only applies if a configuration file is given
    if cfg.folding_config_file is not None:
        # Load the configuration dictionary form YAML file
        with (open(cfg.folding_config_file, "r") as file):
            # Load YAML string
            config = yaml.safe_load(file)
            # Assign unique names to the nodes which can be matched by
            # individual per-node configuration options
            model = model.transform(GiveUniqueNodeNames())
            # Apply the configuration dictionary to the model graph
            model = model.transform(ApplyConfig(config))
    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.FOLDED_HLS_CPPSIM in
            cfg._resolve_verification_steps()):  # noqa
        # Prepare C++ Simulation for verification
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        # Execute a verification step of the model with inputs specified in
        # build configuration
        verify_step(model, cfg, "folded_hls_cppsim", need_parent=True)

    # Return model with configuration applied
    return model


# Runs a node-by-node C++ simulation of the model saving the fill execution
# context
def node_by_node_cppsim(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Save the original model
    original = model
    # Copy the model
    model = copy.deepcopy(model)
    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Load the verification input/output pair
    inp = np.load(cfg.verify_input_npy)  # noqa
    out = np.load(cfg.verify_expected_output_npy)

    # Path to the parent model wrapping the streaming dataflow partition and the
    # wrapped child model, i.e., the inside of the streaming dataflow partition
    parent = f"{cfg.output_dir}/intermediate_models/dataflow_parent.onnx"
    child = f"{cfg.output_dir}/intermediate_models/verify_cppsim.onnx"
    # Save the child model prepared for C++ simulation
    model.save(child)
    # Load the parent model to pass to verification execution
    parent_model = ModelWrapper(parent)

    # Reshape the input/output to match the model
    inp = inp.reshape(parent_model.get_tensor_shape(model.graph.input[0].name))
    out = out.reshape(parent_model.get_tensor_shape(model.graph.output[0].name))

    # Execute the onnx model to collect the result
    # context = execute_onnx(model, context, return_full_exec_context=True)
    context = execute_parent(parent, child, inp, return_full_ctx=True)
    # Extract the output tensor from the execution context
    model_out = context[parent_model.graph.output[0].name]
    # Compare input to output
    result = {True: "SUCCESS", False: "FAIL"}[np.allclose(out, model_out)]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_cppsim_{result}.npz", **context)
    # Return the original, unmodified model
    return original


# Runs a node-by-node RTL simulation of the model saving the fill execution
# context
def node_by_node_rtlsim(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Save the original model
    original = model
    # Copy the model
    model = copy.deepcopy(model)
    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(
        cfg._resolve_fpga_part(), cfg.synth_clk_period_ns)  # noqa
    )
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Load the verification input/output pair
    inp = np.load(cfg.verify_input_npy)  # noqa
    out = np.load(cfg.verify_expected_output_npy)

    # Path to the parent model wrapping the streaming dataflow partition and the
    # wrapped child model, i.e., the inside of the streaming dataflow partition
    parent = f"{cfg.output_dir}/intermediate_models/dataflow_parent.onnx"
    child = f"{cfg.output_dir}/intermediate_models/verify_rtlsim.onnx"
    # Save the child model prepared for RTL simulation
    model.save(child)
    # Load the parent model to pass to verification execution
    parent_model = ModelWrapper(parent)

    # Reshape the input/output to match the model
    inp = inp.reshape(parent_model.get_tensor_shape(model.graph.input[0].name))
    out = out.reshape(parent_model.get_tensor_shape(model.graph.output[0].name))

    # Execute the onnx model to collect the result
    # context = execute_onnx(model, context, return_full_exec_context=True)
    context = execute_parent(parent, child, inp, return_full_ctx=True)
    # Extract the output tensor from the execution context
    model_out = context[parent_model.graph.output[0].name]
    # Compare input to output
    result = {True: "SUCCESS", False: "FAIL"}[np.allclose(out, model_out)]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_rtlsim_{result}.npz", **context)
    # Return the original, unmodified model
    return original

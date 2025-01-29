# ADAPTED FROM Christoph's attention-dummy build_steps.py

# Copies (deep-copies) python objects
import copy
# Numpy for loading and comparing the verification input/output
import numpy as np
# YAML for loading experiment configurations
import yaml
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX quantization data types
from qonnx.core.datatype import DataType
# Converts ONNX graph nodes to QONNX custom-ops if possible
from qonnx.custom_op.registry import getCustomOp
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    Transformation,
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveUniqueParameterTensors,
    ConvertDivToMul,
    ConvertSubToAdd
)
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
# QONNX cleanup transformations
from qonnx.transformation.remove import RemoveIdentityOps
# Precompute constant output nodes
from qonnx.transformation.fold_constants import FoldConstants
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import (
    ConvertSignToThres, RoundAndClipThresholds
)
# Fuse/Absorb operations
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbMulIntoMultiThreshold,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv
)
# Reorder operations
from finn.transformation.streamline.reorder import (
    MoveMulPastFork,
    MoveLinearPastFork,
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
)
# Collapse consecutive operations of the same type
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedTranspose,
    CollapseRepeatedAdd
)
# FINN transformation converting ONNX nodes to hardware custom operators
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
    InferLookupLayer
)
# Remove some operations without real effect
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
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
    MoveSplitMultiHeadsPastMultiThreshold,
    UnrollMultiHeadAttention,
    MoveMergeMultiHeadsPastMultiThreshold
)
# Stream replication for outputs with multiple consumers
from finn.transformation.fpgadataflow.replicate_stream import (
    InferReplicateStream
)
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
            model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed by at least one transformation so the whole
        # sequence of transformations will be reapplied
        return model, graph_modified


# Custom Streamlining transformation: Similar to the built-in transformations
# but exhaustively reapplied until none of the transformations can be applied
# anymore.
def Streamline():  # noqa: Uppercase
    return ComposedTransformation([
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
        RoundAndClipThresholds(),
    ])


# Function running transformations necessary to clean up models containing
# attention operators
def step_tidy_up_pre_attention(model: ModelWrapper, _):
    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())  # noqa Duplicate
    model = model.transform(InferShapes())

    # Cleanup the graph by removing redundant, unnecessary and constant nodes
    # and tensors and give unique names to everything remaining
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(FoldConstants())

    # Remove unnecessary shape and layout transformations
    model = model.transform(RemoveIdentityReshape())
    model = model.transform(RemoveIdentityTranspose())
    # Insert tensor layout annotations for Quant to MultiThreshold transform
    # to determine the correct output channel dimension
    model = model.transform(InferDataLayouts())
    # Return the tidied up model
    return model


# Variant of streamlining transformations adapted to attention operators
def step_streamline_attention(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Exhaustively apply the pattern of streamlining and moving past fork-nodes
    model = model.transform(ComposedTransformation([
        # Apply the set of standard streamlining transformations from finn to
        # the model
        Streamline(),
        # We need a custom streamlining step to enable streamlining through
        # certain fork-nodes Note: This transform is part of finn, but not
        # included in the standard streamlining transformations
        MoveLinearPastFork(),
        # Streamline again there should be more transformations enabled after
        # moving some nodes past forks
        Streamline(),
    ]))

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_attention_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Streamlining transformations to be applied to residual branches
def step_streamline_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Exhaustively apply the pattern for streamlining residual branches. This
    # ensures streamlining to work for arbitrary many consecutive residual
    # blocks, where one "round" of these transformations is required per block.
    model = model.transform(ComposedTransformation([
        # Streamline the residual connections by moving scale factors past
        # elementwise add nodes
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
        MoveScalarLinearPastInvariants(),
        # Do the normal streamlining flow once again
        Streamline(),
    ]))

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_residual_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Streamlining transformation to be applied to the normalization layers
def step_streamline_norms(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Exhaustively apply the pattern for streamlining norms. This ensures
    # streamlining to work for arbitrary many consecutive blocks, where one
    # round of these transformations is required per block.
    model = model.transform(ComposedTransformation([
        # Streamline transposed batch normalization (move transposes past the
        # scale-bias operator, so they can be collapsed afterward)
        MoveTransposePastEltwise(),
        # There should now be transposes next to each other which can be
        # collapsed
        CollapseRepeatedTranspose(),
        # The transposes around the batch normalization should be collapsed by
        # now and cancel each other out
        RemoveIdentityTranspose(),
        # Nested, exhaustive compositions of transformations
        ComposedTransformation([
            # We now might have transpose operations accumulating in front of
            # fork nodes
            MoveTransposePastFork(),
            MoveTransposePastEltwise(),
            CollapseRepeatedTranspose(),
            RemoveIdentityTranspose(),
        ]),
        # This might have caused the normalization scale and bias to accumulate
        # in front of transpose or fork node
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
        MoveScalarLinearPastInvariants(),
        # This might have enabled more streamlining transformations
        Streamline(),
        # We need a custom streamlining step to enable streamlining through
        # certain fork-nodes Note: This transform is part of finn, but not
        # included in the standard streamlining transformations
        MoveLinearPastFork(),
        # This might have enabled more streamlining transformations
        Streamline(),
    ]))

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(model, cfg, "streamlined_norms_python", need_parent=False)

    # Return the streamlined model
    return model


# Streamlining transformation to be applied to the positional encoding layer
def step_streamline_positional(model: ModelWrapper, cfg: DataflowBuildConfig):
    # There is probably a division in front of the quantized positional
    # encoding, which is exactly the inverse of the multiplication in front of
    # that: The are the matching scale factors of the shared input quantizer of
    # input and positional encoding. Convert the division to multiplication, so
    # these two can be merged.
    model = model.transform(ConvertDivToMul())
    # Merge the quantization scales of shared input quantizers
    model = model.transform(CollapseRepeatedMul())
    # Push scalar multiplications, probably scale factors of quantizers, into
    # the branches of a fork
    model = model.transform(MoveMulPastFork())

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_positional_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Function running the InferScaledDotProductAttention transformation
def step_convert_attention_to_hw(model: ModelWrapper, _):
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


# Function running the InferReplicateStream transformation
def step_replicate_streams(model: ModelWrapper, _):
    # Properly replicate the stream feeding the query, key and value projections
    return model.transform(InferReplicateStream())


# Post-processing tidy-up squeezing dimensions and identity operators left over
# from mapping the attention operators
def step_tidy_up_post_attention(model: ModelWrapper, _):
    # Remove dimensions of size 1 (single batch tensors)
    model = model.transform(Squeeze())
    model = model.transform(RemoveIdentityTranspose())

    # Squeezing might enable absorbing adds into thresholds once again
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # If applicable, absorb the final thresholds into the attention operator
    #   Note: Might be applicable again after squeezing a transpose away
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())

    # Squeezing might enable some more streamlining transformations once again
    model = model.transform(ComposedTransformation([
        # Streamline the residual connections by moving scale factors past
        # elementwise add nodes
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
        MoveScalarLinearPastInvariants(),
        # Do the normal streamlining flow once again
        Streamline(),
    ]))

    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # Return the tidied up model
    return model


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


# Runs a node-by-node Python simulation of the model saving the fill execution
# context
# Note: Assumes no execution mode to be set
def node_by_node_python(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Save the original model
    original = model
    # Copy the model
    model = copy.deepcopy(model)

    # Load the verification input/output pair
    inp = np.load(cfg.verify_input_npy)  # noqa
    out = np.load(cfg.verify_expected_output_npy)

    # Path to the parent model wrapping the streaming dataflow partition and the
    # wrapped child model, i.e., the inside of the streaming dataflow partition
    parent = f"{cfg.output_dir}/intermediate_models/dataflow_parent.onnx"
    child = f"{cfg.output_dir}/intermediate_models/verify_cppsim.onnx"
    # Save the child model prepared for python simulation
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
    result = {True: "SUCCESS", False: "FAIL"}[
        np.allclose(out, model_out, atol=1e-3)
    ]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_python_{result}.npz", **context)
    # Return the original, unmodified model
    return original


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
    result = {True: "SUCCESS", False: "FAIL"}[
        np.allclose(out, model_out, atol=1e-3)
    ]
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
    result = {True: "SUCCESS", False: "FAIL"}[
        np.allclose(out, model_out, atol=1e-3)
    ]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_rtlsim_{result}.npz", **context)
    # Return the original, unmodified model
    return original

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from Brainsmith:
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Custom build steps for the BERT benchmark.

These steps are ported from the Brainsmith BERT flow
(``examples/bert_training`` + ``brainsmith/steps``) into self-contained FINN
``DataflowBuildConfig`` steps. The Brainsmith ``@step`` / ``@transform``
decorators and the ``get_transform(...)`` / ``apply_transforms(...)`` plugin
indirection have been removed; the concrete QONNX/FINN transforms are imported
directly.
"""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    SortCommutativeInputsInitializerLast,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedOp
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def step_bert_cleanup(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Basic graph cleanup / preparation for BERT models.

    Ported from Brainsmith ``bert_cleanup`` step. Sorts commutative inputs so
    initializers come last and removes identity ops.
    """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model


def step_remove_head(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Remove all nodes up to the first LayerNormalization and rewire the input.

    Ported from the Brainsmith example ``remove_head`` step. After the head is
    removed, the QONNX->FINN pre-processing transforms from Brainsmith's
    ``qonnx_to_finn`` step that the stock ``step_qonnx_to_finn`` does not run
    (ExtractNormScaleBias, FoldConstants, ConvertDivToMul) are applied here.
    They run *after* the head removal because ExtractNormScaleBias appends
    Mul/Add nodes onto LayerNorm outputs; the stock ``step_qonnx_to_finn``
    (ConvertQONNXtoFINN) runs in a later step.
    """
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"

    to_remove = []
    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected a linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes (including the first LayerNorm)
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i, ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    # Clean up after head removal
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())

    # QONNX->FINN pre-processing (ported from Brainsmith qonnx_to_finn step).
    model = model.transform(ExtractNormScaleBias())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul())

    return model


def step_bert_streamlining(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """BERT-specific streamlining.

    Ported from Brainsmith ``bert_streamlining``. Handles the Mul nodes left
    over from the SoftMax transformations: it moves the Mul down the graph so it
    can be merged into a MultiThreshold node.

    Channelwise Add/Mul nodes are not absorbed into per-tensor MultiThresholds
    (``preserve_thresh_shape``), so the thresholds keep their (1, 255) shape and
    those Add/Mul nodes become standalone ElementwiseAdd/ElementwiseMul layers.
    """
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(AbsorbAddIntoMultiThreshold(preserve_thresh_shape=True))
    model = model.transform(AbsorbMulIntoMultiThreshold(preserve_thresh_shape=True))
    model = model.transform(RoundAndClipThresholds())

    model = model.transform(MoveOpPastFork(["Mul"]))

    model = model.transform(MoveScalarMulPastMatMul())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(AbsorbAddIntoMultiThreshold(preserve_thresh_shape=True))
    model = model.transform(AbsorbMulIntoMultiThreshold(preserve_thresh_shape=True))
    model = model.transform(RoundAndClipThresholds())

    model = model.transform(CollapseRepeatedOp("Mul", lambda x, y: y * x))

    # Final cleanup
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())

    return model

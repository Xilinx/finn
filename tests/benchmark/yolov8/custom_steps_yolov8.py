# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Ported from the YOLOv8 custom steps implemented by Michal Danilowicz.

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MakeScaleResizeNHWC,
    MoveAddPastJoinConcat,
    MoveLinearPastFork,
    MoveMulPastJoinAdd,
    MoveMulPastJoinConcat,
    MoveMulPastMaxPool,
    MoveScalarLinearPastSplit,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
    MoveTransposePastJoinConcat,
    MoveTransposePastSplit,
)


def step_yolov8_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # YOLOv8-specific streamline: push the affine scales past the C2f
    # split/concat and residual fork/join-Add structure (and through the SPPF
    # MaxPool chain) so they absorb into the MultiThresholds, then lower the convs
    # to matmuls and detangle the resulting NCHW<->NHWC transposes around the
    # split/fork/join/resize/maxpool nodes.
    model = model.transform(Streamline())
    model = model.transform(MoveAddPastJoinConcat())
    streamline_transformations = [
        # Affine ops
        MoveScalarLinearPastSplit(),
        MoveLinearPastFork(),
        MoveMulPastJoinAdd(),
        MoveMulPastJoinConcat(),
        Streamline(),
        # Affine ops in SPPF
        MoveLinearPastFork(),
        MoveMulPastMaxPool(),
        MoveLinearPastFork(),
        MoveMulPastMaxPool(),
        MoveMulPastJoinConcat(),
        Streamline(),
        # Transposes
        LowerConvsToMatMul(),
        AbsorbTransposeIntoMultiThreshold(),
        AbsorbConsecutiveTransposes(),
        MakeScaleResizeNHWC(),
        MoveTransposePastSplit(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        MoveTransposePastJoinConcat(),
        AbsorbConsecutiveTransposes(),
        # Transposes in SPPF
        MakeMaxPoolNHWC(),
        MoveTransposePastJoinConcat(),
        AbsorbConsecutiveTransposes(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model

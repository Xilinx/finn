# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.util.cleanup import cleanup_model

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbTransposeIntoFlatten,
    AbsorbTransposeIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.streamline.reorder import (
    MoveAddPastJoinAdd,
    MoveMulPastJoinAdd,
    MoveOpPastFork,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
)
from finn.util.basic import make_build_dir
from finn.util.pytorch import ToTensor


def step_resnet18_attach_preproc(model: ModelWrapper, cfg: DataflowBuildConfig):
    # The CIFAR-10 ResNet-18 exports take a float image without an input quantizer.
    # Prepend torchvision's ToTensor (raw uint8 / 255) so the accelerator runs on
    # raw uint8 pixels, annotate the global input as UINT8, and append a TopK(k=1)
    # so the graph ends in a top-1 class index (-> LabelSelect).
    ishape = model.get_tensor_shape(model.get_first_global_in())
    preproc_file = os.path.join(make_build_dir("resnet18_preproc_"), "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), preproc_file, opset_version=13)
    qonnx_cleanup(preproc_file, out_file=preproc_file)
    pre_model = ModelWrapper(preproc_file)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes())
    pre_model = pre_model.transform(FoldConstants())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def step_resnet18_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # ResNet-18-specific streamline: push the scalar scales past the residual
    # forks/join-Adds and absorb them into the MultiThresholds, then lower the
    # convs to matmuls and detangle the resulting NCHW<->NHWC transposes around
    # the residual forks/joins.
    streamline_transformations = [
        MoveOpPastFork(["Mul"]),
        MoveAddPastJoinAdd(),
        MoveMulPastJoinAdd(),
        ConvertDivToMul(),
        BatchNormToAffine(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoConv(),
        RemoveIdentityOps(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
    model = cleanup_model(model)

    lower_transformations = [
        LowerConvsToMatMul(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        AbsorbTransposeIntoFlatten(),
    ]
    for trn in lower_transformations:
        model = model.transform(trn)
    model = cleanup_model(model)

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model

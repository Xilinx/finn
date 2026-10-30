# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.subpixel_to_deconv import SubPixelToDeconvolution

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import (
    InferPixelPaddingDeconv,
)
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import MoveScalarMulPastConvTranspose
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import make_build_dir
from finn.util.pytorch import ToTensor


def step_espcn_subpixel_to_deconv(model: ModelWrapper, cfg: DataflowBuildConfig):
    # ESPCN upsamples with a sub-pixel convolution (Conv -> DepthToSpace). Rewrite
    # it as an equivalent ConvTranspose, which FINN implements as
    # FMPadding_Pixel + sliding window + MVAU (see step_espcn_lower_deconv).
    model = model.transform(InferShapes())
    model = model.transform(SubPixelToDeconvolution())
    model = model.transform(InferShapes())
    return model


def step_espcn_attach_preproc(model: ModelWrapper, cfg: DataflowBuildConfig):
    # The BSD300 ESPCN export takes a [0, 1] float image without an input
    # quantizer. Prepend torchvision's ToTensor (raw uint8 / 255) so the
    # accelerator runs on raw uint8 pixels, and annotate the global input as UINT8.
    ishape = model.get_tensor_shape(model.get_first_global_in())
    preproc_file = os.path.join(make_build_dir("espcn_preproc_"), "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), preproc_file, opset_version=11)
    pre_model = ModelWrapper(preproc_file)
    pre_model.set_tensor_datatype(pre_model.get_first_global_in(), DataType["UINT8"])
    pre_model = pre_model.transform(InferShapes())
    pre_model = pre_model.transform(FoldConstants())
    pre_model = pre_model.transform(GiveUniqueNodeNames())
    pre_model = pre_model.transform(GiveUniqueParameterTensors())
    pre_model = pre_model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(MergeONNXModels(pre_model))
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    return model


def step_espcn_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # ESPCN-specific streamline: additionally move the scalar scale past the
    # ConvTranspose so it absorbs into the following MultiThreshold, then lower the
    # convs to matmuls and absorb the resulting NCHW<->NHWC transposes.
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(MoveScalarMulPastConvTranspose())
    model = model.transform(AbsorbMulIntoMultiThreshold())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(AbsorbTransposeIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_espcn_lower_deconv(model: ModelWrapper, cfg: DataflowBuildConfig):
    # The generic HW conversion does not handle ConvTranspose: lower it to
    # FMPadding_Pixel + Im2Col + MatMul (NHWC, surrounded by transposes) here, so
    # phase_convert_to_hardware picks up the Im2Col/MatMul like any other conv.
    model = model.transform(InferPixelPaddingDeconv())
    model = model.transform(AbsorbTransposeIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    return model

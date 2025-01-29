# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from qonnx.core.modelwrapper import ModelWrapper

from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.general import (
    ConvertSubToAdd,
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    SortGraph,
    RemoveUnusedTensors,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    ApplyConfig,
)

from finn.transformation.streamline.absorb import (
    AbsorbScalarMulAddIntoTopK,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoMultiThreshold,
)

from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)

from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveScalarMulPastMatMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveMaxPoolPastMultiThreshold,
)

from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine

# just for not linear
from finn.transformation.streamline.reorder import (
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
)

from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.core.datatype import DataType

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.insert_topk import InsertTopK
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
)

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten


def step_resnet50_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def step_resnet50_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        AbsorbScalarMulAddIntoTopK(),  # before MoveAddPastMul to avoid int->float
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    for iter_id in range(4):
        model = step_resnet50_streamline_linear(model, cfg)
        model = step_resnet50_streamline_nonlinear(model, cfg)

        # big loop tidy up
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    return model


def step_resnet50_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hw_transformations = [
        to_hw.InferAddStreamsLayer,
        LowerConvsToMatMul,
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        AbsorbTransposeIntoMultiThreshold,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
        to_hw.InferLabelSelectLayer,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model


def step_resnet50_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        # previously, we would always ran the finn experimental partitioner on ResNet-50
        # this is now changed and a fixed floorplan is applied
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.floorplan_path))
        print("Fixed SLR floorplanning applied")

        # if you would like to try out the experimental partitioner
        # please uncomment the lines (that are not marked as comment) below.

        # import numpy as np
        # from finnexperimental.analysis.partitioning import partition

        # comment: apply partitioning of the model, restricting the first and last layer to SLR0
        # default_slr = 0
        # abs_anchors = [(0, [default_slr]), (-1, [default_slr])]

        # comment: increase resource limits to make partitioning feasible, except for SLR0
        # comment: which also has DDR subsystem
        # limits = np.array(
        #    [
        #        [0.75, 0.5, 0.7, 0.6, 0.6],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #    ]
        # )
        # floorplan = partition(
        #    model,
        #    cfg.synth_clk_period_ns,
        #    cfg.board,
        #    abs_anchors=abs_anchors,
        #    multivariant=False,
        #    linear_cuts=True,
        #    limits=limits,
        # )[0]

        # comment: apply floorplan to model
        # model = model.transform(ApplyConfig(floorplan))
        # print("SLR floorplanning applied from partitioner")
    return model
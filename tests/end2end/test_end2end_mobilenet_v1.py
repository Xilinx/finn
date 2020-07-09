# Copyright (c) 2020, Xilinx
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

import os
import numpy as np
import brevitas.onnx as bo

from finn.custom_op.registry import getCustomOp
from finn.util.pytorch import NormalizePreProc
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType

from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.transformation.insert_topk import InsertTopK
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline import Streamline
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.streamline.remove import RemoveIdentityOps
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)


build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]


def test_end2end_mobilenet_preprocess_export():
    preproc_onnx = build_dir + "/end2end_mobilenet_preproc.onnx"
    mean = [0.485, 0.456, 0.406]
    std = 0.226
    ch = 3
    preproc = NormalizePreProc(mean, std, ch)
    bo.export_finn_onnx(preproc, (1, 3, 224, 224), preproc_onnx)
    preproc_model = ModelWrapper(preproc_onnx)
    # set input finn datatype to UINT8
    preproc_model.set_tensor_datatype(preproc_model.graph.input[0].name, DataType.UINT8)
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())
    preproc_model.save(build_dir + "/end2end_mobilenet_preproc.onnx")


def test_end2end_mobilenet_export():
    finn_onnx = build_dir + "/end2end_mobilenet_export.onnx"
    mobilenet = get_test_model_trained("mobilenet", 4, 4)
    bo.export_finn_onnx(mobilenet, (1, 3, 224, 224), finn_onnx)
    model = ModelWrapper(finn_onnx)
    model.save(build_dir + "/end2end_mobilenet_export.onnx")


def test_end2end_mobilenet_tidy_and_merge_with_preproc():
    preproc_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_preproc.onnx"
    )
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_export.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InsertTopK())
    # get initializer from Mul that will be absorbed into topk
    a0 = model.get_initializer(model.graph.node[-2].input[1])
    np.save(build_dir + "/end2end_mobilenet_topk_scale.npy", a0)
    model = model.transform(absorb.AbsorbScalarMulIntoTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(MergeONNXModels(preproc_model))
    model.save(build_dir + "/end2end_mobilenet_tidy.onnx")


def test_end2end_mobilenet_streamline():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_tidy.onnx")
    model = model.transform(Streamline())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(reorder.MoveMulPastDWConv())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(ChangeDataLayoutQuantAvgPool2d())
    model = model.transform(InferDataLayouts())
    model = model.transform(reorder.MoveTransposePastScalarMul())
    model = model.transform(absorb.AbsorbTransposeIntoFlatten())
    model = model.transform(reorder.MoveFlattenPastAffine())
    model = model.transform(reorder.MoveFlattenPastTopK())
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    model = model.transform(CollapseRepeatedMul())
    model = model.transform(RemoveIdentityOps())
    model.save(build_dir + "/end2end_mobilenet_streamlined.onnx")


def test_end2end_mobilenet_lowering():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_streamlined.onnx"
    )
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model.save(build_dir + "/end2end_mobilenet_lowered.onnx")


def test_end2end_mobilenet_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_lowered.onnx")
    model = model.transform(to_hls.InferPool_Batch())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferVVAU())
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer())
    model = model.transform(to_hls.InferChannelwiseLinearLayer())
    model.save(build_dir + "/end2end_mobilenet_hls_layers.onnx")


def test_end2end_mobilenet_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_mobilenet_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_mobilenet_dataflow_model.onnx")

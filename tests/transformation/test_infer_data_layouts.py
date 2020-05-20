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

import brevitas.onnx as bo
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.util.test import get_test_model_trained
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.infer_data_layouts import InferDataLayouts
import finn.core.data_layout as DataLayout

export_onnx_path_cnv = "test_output_cnv.onnx"


def test_infer_data_layouts():
    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path_cnv)
    model = ModelWrapper(export_onnx_path_cnv)
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())

    assert model.get_tensor_layout("global_in") == DataLayout.NCHW
    assert model.get_tensor_layout("Conv_0_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("MaxPool_0_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("MultiThreshold_6_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("Reshape_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("MatMul_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("global_out") == DataLayout.NC

    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())

    assert model.get_tensor_layout("global_in") == DataLayout.NCHW
    assert model.get_tensor_layout("Transpose_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("Im2Col_0_out0") == DataLayout.NHWC
    # note: im2col output isn't really NHWC or any other common layout
    # since the concept of channels changes with lowering... but it is
    # conceptually close to NHWC since the innermost dim gets multiplied
    assert model.get_tensor_layout("MatMul_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("Transpose_1_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("Transpose_2_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("MaxPoolNHWC_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("Reshape_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("global_out") == DataLayout.NC

    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())

    assert model.get_tensor_layout("global_in") == DataLayout.NCHW
    assert model.get_tensor_layout("Transpose_0_out0") == DataLayout.NHWC
    # note: im2col output isn't really NHWC or any other common layout
    # since the concept of channels changes with lowering... but it is
    # conceptually close to NHWC since the innermost dim gets multiplied
    assert (
        model.get_tensor_layout("ConvolutionInputGenerator_0_out0") == DataLayout.NHWC
    )
    assert model.get_tensor_layout("StreamingFCLayer_Batch_3_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("Reshape_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("StreamingFCLayer_Batch_6_out0") == DataLayout.NC
    assert model.get_tensor_layout("global_out") == DataLayout.NC

    os.remove(export_onnx_path_cnv)

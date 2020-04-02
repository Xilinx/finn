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

# from pkgutil import get_data

# import pytest

# import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA

# import onnx.numpy_helper as nph

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.move_reshape import MoveReshape
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.streamline import Streamline
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls

from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained


build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 5
mem_mode = "const"


def test_end2end_cnv_w1a1_export():
    import brevitas.onnx as bo

    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(
        cnv, (1, 3, 32, 32), build_dir + "/end2end_cnv_w1a1_export.onnx"
    )


def test_end2end_cnv_w1a1_import_and_tidy():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_export.onnx")
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_cnv_w1a1_tidy.onnx")


def test_end2end_cnv_w1a1_streamline():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_tidy.onnx")
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")


def test_end2end_cnv_w1a1_convert_to_hls_layers():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(MoveReshape())
    model.save(build_dir + "/end2end_cnv_w1a1_hls_layers.onnx")

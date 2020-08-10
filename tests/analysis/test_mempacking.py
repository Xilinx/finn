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

from pkgutil import get_data
from finn.analysis.mempacking import pack_memory, packDefaultConfig
from finn.analysis.shapes import weight_shapes
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames
from finn.core.datatype import DataType


def test_weight_shapes():

    # test with tfcW1A1
    raw_m = get_data(
        "finn", "data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    wshapes = model.analysis(weight_shapes)

    pe = wshapes["StreamingFCLayer_Batch_0"]["PE"]
    simd = wshapes["StreamingFCLayer_Batch_0"]["SIMD"]
    wmem = wshapes["StreamingFCLayer_Batch_0"]["WMEM"]
    dt = wshapes["StreamingFCLayer_Batch_0"]["DataType"]

    assert pe == 16 and simd == 16 and wmem == 196 and dt == DataType.BINARY

    pe = wshapes["StreamingFCLayer_Batch_1"]["PE"]
    simd = wshapes["StreamingFCLayer_Batch_1"]["SIMD"]
    wmem = wshapes["StreamingFCLayer_Batch_1"]["WMEM"]
    dt = wshapes["StreamingFCLayer_Batch_1"]["DataType"]

    assert pe == 16 and simd == 16 and wmem == 16 and dt == DataType.BINARY

    pe = wshapes["StreamingFCLayer_Batch_2"]["PE"]
    simd = wshapes["StreamingFCLayer_Batch_2"]["SIMD"]
    wmem = wshapes["StreamingFCLayer_Batch_2"]["WMEM"]
    dt = wshapes["StreamingFCLayer_Batch_2"]["DataType"]

    assert pe == 16 and simd == 16 and wmem == 16 and dt == DataType.BINARY

    pe = wshapes["StreamingFCLayer_Batch_3"]["PE"]
    simd = wshapes["StreamingFCLayer_Batch_3"]["SIMD"]
    wmem = wshapes["StreamingFCLayer_Batch_3"]["WMEM"]
    dt = wshapes["StreamingFCLayer_Batch_3"]["DataType"]

    assert pe == 10 and simd == 16 and wmem == 4 and dt == DataType.BINARY


def test_weight_packing():

    # test with tfcW1A1
    raw_m = get_data(
        "finn", "data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())

    config = packDefaultConfig()
    config.thresh_min = 0

    packed_shapes = model.analysis(pack_memory, args=config)

    # check that we have all PEs in the result
    pe = [0, 0, 0, 0]
    for bin in packed_shapes:
        for i in range(4):
            pe[i] += packed_shapes[bin].count("StreamingFCLayer_Batch_" + str(i))

    shapes = model.analysis(weight_shapes)
    assert pe[0] == shapes["StreamingFCLayer_Batch_0"]["PE"]
    assert pe[1] == shapes["StreamingFCLayer_Batch_1"]["PE"]
    assert pe[2] == shapes["StreamingFCLayer_Batch_2"]["PE"]
    assert pe[3] == shapes["StreamingFCLayer_Batch_3"]["PE"]

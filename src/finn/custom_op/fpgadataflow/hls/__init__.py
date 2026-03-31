# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

# flake8: noqa
# Disable linting from here, as all import will be flagged E402 and maybe F401

from finn.custom_op.fpgadataflow.hls.checksum_hls import CheckSum_hls
from finn.custom_op.fpgadataflow.hls.concat_hls import StreamingConcat_hls
from finn.custom_op.fpgadataflow.hls.crop_hls import Crop_hls
from finn.custom_op.fpgadataflow.hls.duplicatestreams_hls import DuplicateStreams_hls

# Also import ElementwiseBinary variants
from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
    ElementwiseAbsDiff_hls,
    ElementwiseAdd_hls,
    ElementwiseAnd_hls,
    ElementwiseBinaryOperation_hls,
    ElementwiseBitwiseAnd_hls,
    ElementwiseBitwiseOr_hls,
    ElementwiseBitwiseXor_hls,
    ElementwiseDiv_hls,
    ElementwiseEqual_hls,
    ElementwiseGreater_hls,
    ElementwiseGreaterOrEqual_hls,
    ElementwiseLess_hls,
    ElementwiseLessOrEqual_hls,
    ElementwiseMul_hls,
    ElementwiseOr_hls,
    ElementwiseSub_hls,
    ElementwiseXor_hls,
)
from finn.custom_op.fpgadataflow.hls.fmpadding_pixel_hls import FMPadding_Pixel_hls
from finn.custom_op.fpgadataflow.hls.globalaccpool_hls import GlobalAccPool_hls
from finn.custom_op.fpgadataflow.hls.hwsoftmax_hls import HWSoftmax_hls
from finn.custom_op.fpgadataflow.hls.iodma_hls import IODMA_hls
from finn.custom_op.fpgadataflow.hls.labelselect_hls import LabelSelect_hls
from finn.custom_op.fpgadataflow.hls.layernorm_hls import LayerNorm_hls
from finn.custom_op.fpgadataflow.hls.lookup_hls import Lookup_hls
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.hls.outer_shuffle_hls import OuterShuffle_hls
from finn.custom_op.fpgadataflow.hls.pool_hls import Pool_hls
from finn.custom_op.fpgadataflow.hls.split_hls import StreamingSplit_hls
from finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import (
    StreamingDataWidthConverter_hls,
)
from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
from finn.custom_op.fpgadataflow.hls.tlastmarker_hls import TLastMarker_hls
from finn.custom_op.fpgadataflow.hls.upsampler_hls import UpsampleNearestNeighbour_hls
from finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VVAU_hls

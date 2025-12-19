# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

# Import all custom ops - they will be discovered automatically via namespace
from finn.custom_op.fpgadataflow.addstreams import AddStreams
from finn.custom_op.fpgadataflow.channelwise_op import ChannelwiseOp
from finn.custom_op.fpgadataflow.concat import StreamingConcat
from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.crop import Crop
from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams

# Also import ElementwiseBinary variants
from finn.custom_op.fpgadataflow.elementwise_binary import (
    ElementwiseAdd,
    ElementwiseAnd,
    ElementwiseBinaryOperation,
    ElementwiseBitwiseAnd,
    ElementwiseBitwiseOr,
    ElementwiseBitwiseXor,
    ElementwiseDiv,
    ElementwiseEqual,
    ElementwiseGreater,
    ElementwiseGreaterOrEqual,
    ElementwiseLess,
    ElementwiseLessOrEqual,
    ElementwiseMul,
    ElementwiseOr,
    ElementwiseSub,
    ElementwiseXor,
)
from finn.custom_op.fpgadataflow.fmpadding import FMPadding
from finn.custom_op.fpgadataflow.fmpadding_pixel import FMPadding_Pixel
from finn.custom_op.fpgadataflow.globalaccpool import GlobalAccPool
from finn.custom_op.fpgadataflow.hwsoftmax import HWSoftmax
from finn.custom_op.fpgadataflow.inner_shuffle import InnerShuffle
from finn.custom_op.fpgadataflow.labelselect import LabelSelect
from finn.custom_op.fpgadataflow.layernorm import LayerNorm
from finn.custom_op.fpgadataflow.lookup import Lookup
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.outer_shuffle import OuterShuffle
from finn.custom_op.fpgadataflow.pool import Pool
from finn.custom_op.fpgadataflow.shuffle import Shuffle
from finn.custom_op.fpgadataflow.split import StreamingSplit
from finn.custom_op.fpgadataflow.streamingdataflowpartition import (
    StreamingDataflowPartition,
)
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)
from finn.custom_op.fpgadataflow.streamingeltwise import StreamingEltwise
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.upsampler import UpsampleNearestNeighbour
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU

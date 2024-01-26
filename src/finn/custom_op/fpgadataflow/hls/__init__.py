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

from finn.custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
from finn.custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls
from finn.custom_op.fpgadataflow.hls.checksum_hls import CheckSum_hls
from finn.custom_op.fpgadataflow.hls.concat_hls import StreamingConcat_hls
from finn.custom_op.fpgadataflow.hls.convolutioninputgenerator_hls import (
    ConvolutionInputGenerator_hls,
)
from finn.custom_op.fpgadataflow.hls.downsampler_hls import DownSampler_hls
from finn.custom_op.fpgadataflow.hls.duplicatestreams_hls import DuplicateStreams_hls
from finn.custom_op.fpgadataflow.hls.fmpadding_hls import FMPadding_hls
from finn.custom_op.fpgadataflow.hls.fmpadding_pixel_hls import FMPadding_Pixel_hls
from finn.custom_op.fpgadataflow.hls.globalaccpool_hls import GlobalAccPool_hls
from finn.custom_op.fpgadataflow.hls.iodma_hls import IODMA_hls
from finn.custom_op.fpgadataflow.hls.labelselect_hls import LabelSelect_hls
from finn.custom_op.fpgadataflow.hls.lookup_hls import Lookup_hls
from finn.custom_op.fpgadataflow.hls.pool_hls import Pool_hls
from finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import (
    StreamingDataWidthConverter_hls,
)
from finn.custom_op.fpgadataflow.hls.streamingeltwise_hls import StreamingEltwise_hls
from finn.custom_op.fpgadataflow.hls.streamingmaxpool_hls import StreamingMaxPool_hls
from finn.custom_op.fpgadataflow.hls.tlastmarker_hls import TLastMarker_hls
from finn.custom_op.fpgadataflow.hls.upsampler_hls import UpsampleNearestNeighbour_hls

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["AddStreams_hls"] = AddStreams_hls
custom_op["ChannelwiseOp_hls"] = ChannelwiseOp_hls
custom_op["CheckSum_hls"] = CheckSum_hls
custom_op["ConvolutionInputGenerator_hls"] = ConvolutionInputGenerator_hls
custom_op["DownSampler_hls"] = DownSampler_hls
custom_op["DuplicateStreams_hls"] = DuplicateStreams_hls
custom_op["FMPadding_hls"] = FMPadding_hls
custom_op["FMPadding_Pixel_hls"] = FMPadding_Pixel_hls
custom_op["GlobalAccPool_hls"] = GlobalAccPool_hls
custom_op["IODMA_hls"] = IODMA_hls
custom_op["LabelSelect_hls"] = LabelSelect_hls
custom_op["Lookup_hls"] = Lookup_hls
custom_op["Pool_hls"] = Pool_hls
custom_op["StreamingConcat_hls"] = StreamingConcat_hls
custom_op["StreamingEltwise_hls"] = StreamingEltwise_hls
custom_op["StreamingDataWidthConverter_hls"] = StreamingDataWidthConverter_hls
custom_op["StreamingMaxPool_hls"] = StreamingMaxPool_hls
custom_op["TLastMarker_hls"] = TLastMarker_hls
custom_op["UpsampleNearestNeighbour_hls"] = UpsampleNearestNeighbour_hls

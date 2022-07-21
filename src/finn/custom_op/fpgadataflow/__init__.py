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

from finn.custom_op.fpgadataflow.addstreams_batch import AddStreams_Batch
from finn.custom_op.fpgadataflow.channelwise_op_batch import ChannelwiseOp_Batch
from finn.custom_op.fpgadataflow.checksum import CheckSum
from finn.custom_op.fpgadataflow.concat import StreamingConcat
from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.convolutioninputgenerator1d import (
    ConvolutionInputGenerator1D,
)
from finn.custom_op.fpgadataflow.downsampler import DownSampler
from finn.custom_op.fpgadataflow.duplicatestreams_batch import DuplicateStreams_Batch
from finn.custom_op.fpgadataflow.eltwise import StreamingEltwise
from finn.custom_op.fpgadataflow.fmpadding_batch import FMPadding_Batch
from finn.custom_op.fpgadataflow.globalaccpool_batch import GlobalAccPool_Batch
from finn.custom_op.fpgadataflow.iodma import IODMA
from finn.custom_op.fpgadataflow.labelselect_batch import LabelSelect_Batch
from finn.custom_op.fpgadataflow.lookup import Lookup
from finn.custom_op.fpgadataflow.matrixvectoractivation import MatrixVectorActivation
from finn.custom_op.fpgadataflow.pool_batch import Pool_Batch
from finn.custom_op.fpgadataflow.streamingdataflowpartition import (
    StreamingDataflowPartition,
)
from finn.custom_op.fpgadataflow.streamingdatawidthconverter_batch import (
    StreamingDataWidthConverter_Batch,
)
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from finn.custom_op.fpgadataflow.streamingmaxpool_batch import StreamingMaxPool_Batch
from finn.custom_op.fpgadataflow.thresholding_batch import Thresholding_Batch
from finn.custom_op.fpgadataflow.tlastmarker import TLastMarker
from finn.custom_op.fpgadataflow.upsampler import UpsampleNearestNeighbour_Batch
from finn.custom_op.fpgadataflow.vectorvectoractivation import VectorVectorActivation

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["DownSampler"] = DownSampler
custom_op["StreamingMaxPool_Batch"] = StreamingMaxPool_Batch
custom_op["MatrixVectorActivation"] = MatrixVectorActivation
custom_op["ConvolutionInputGenerator"] = ConvolutionInputGenerator
custom_op["ConvolutionInputGenerator1D"] = ConvolutionInputGenerator1D
custom_op["TLastMarker"] = TLastMarker
custom_op["StreamingDataWidthConverter_Batch"] = StreamingDataWidthConverter_Batch
custom_op["StreamingFIFO"] = StreamingFIFO
custom_op["GlobalAccPool_Batch"] = GlobalAccPool_Batch
custom_op["Pool_Batch"] = Pool_Batch
custom_op["FMPadding_Batch"] = FMPadding_Batch
custom_op["Thresholding_Batch"] = Thresholding_Batch
custom_op["AddStreams_Batch"] = AddStreams_Batch
custom_op["LabelSelect_Batch"] = LabelSelect_Batch
custom_op["DuplicateStreams_Batch"] = DuplicateStreams_Batch
custom_op["VectorVectorActivation"] = VectorVectorActivation
custom_op["ChannelwiseOp_Batch"] = ChannelwiseOp_Batch
custom_op["IODMA"] = IODMA
custom_op["StreamingDataflowPartition"] = StreamingDataflowPartition
custom_op["UpsampleNearestNeighbour_Batch"] = UpsampleNearestNeighbour_Batch
custom_op["Lookup"] = Lookup
custom_op["StreamingConcat"] = StreamingConcat
custom_op["CheckSum"] = CheckSum
custom_op["StreamingEltwise"] = StreamingEltwise

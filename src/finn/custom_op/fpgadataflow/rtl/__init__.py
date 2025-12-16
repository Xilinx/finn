# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl import (
    ConvolutionInputGenerator_rtl,
)
from finn.custom_op.fpgadataflow.rtl.fmpadding_rtl import FMPadding_rtl
from finn.custom_op.fpgadataflow.rtl.inner_shuffle_rtl import InnerShuffle_rtl
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
from finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import (
    StreamingDataWidthConverter_rtl,
)
from finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFO_rtl
from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
from finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl import VVAU_rtl

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["ConvolutionInputGenerator_rtl"] = ConvolutionInputGenerator_rtl
custom_op["FMPadding_rtl"] = FMPadding_rtl
custom_op["StreamingDataWidthConverter_rtl"] = StreamingDataWidthConverter_rtl
custom_op["StreamingFIFO_rtl"] = StreamingFIFO_rtl
custom_op["MVAU_rtl"] = MVAU_rtl
custom_op["VVAU_rtl"] = VVAU_rtl
custom_op["Thresholding_rtl"] = Thresholding_rtl
custom_op["InnerShuffle_rtl"] = InnerShuffle_rtl

# Copyright (c) 2021, Xilinx
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

from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

from finn.transformation.base import Transformation
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.gemm_to_matmul import GemmToMatMul
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
)


class ConvertQONNXtoFINN(Transformation):
    """Converts QONNX dialect to FINN ONNX dialect.
    First the weights are converted using the FoldQuantWeights transformation,
    then the ConvertQuantActToMultiThreshold transformation is used to convert
    the activations.
    If incompatibilities are found a ValueError or RuntimeError is raised.
    """

    def apply(self, model):
        # Gemm operations are not supported by FINN, so we convert them to MatMul
        model = model.transform(GemmToMatMul())
        model = model.transform(FoldTransposeIntoQuantInit())
        # Make sure the datatypes exist, these are required for folding the weights
        model = model.transform(InferDataTypes())
        # Fold weights
        model = model.transform(FoldQuantWeights())
        # Convert activations
        model = model.transform(ConvertQuantActToMultiThreshold())
        # Recompute datatypes
        model = model.transform(InferDataTypes())

        return (model, False)

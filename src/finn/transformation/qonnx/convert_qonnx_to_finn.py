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
from finn.transformation.gemm_to_matmul import GemmToMatMul
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.qonnx.extract_conv_bias import ExtractBiasFromConv
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
    AvgPoolAndTruncToQuantAvgPool,
)
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
    default_filter_function_generator,
)
from finn.transformation.remove import RemoveEmptyPadding


class ConvertQONNXtoFINN(Transformation):
    """Converts QONNX dialect to FINN ONNX dialect.
    First the weights are converted using the FoldQuantWeights transformation,
    then the ConvertQuantActToMultiThreshold transformation is used to convert
    the activations.
    If incompatibilities are found a ValueError or RuntimeError is raised.

    The optional keyword argument `filter_function`
    presents a way to control which Quant and BipolarQuant nodes in the activation path
    are converted to MultiThreshold nodes. A warning will be emitted when a Quant node
    is not converted to a MultiThreshold node.

    :param filter_function: Each candidate Quant and BinaryQant node is first evaluated
    by this function. If the function returns False,
    then the node is not converted to a MultiTrheshold node.
    The function is given the model and candidate node as parameters.
    Per default a filter function is inserted, which disables the conversion of
    Quant nodes, which have a bit width of larger than 8.
    Defaults to: default_filter_function_generator(max_multithreshold_bit_width=8)
    """

    def __init__(
        self,
        filter_function=default_filter_function_generator(
            max_multithreshold_bit_width=8
        ),
    ):
        super().__init__()
        self._filter_function = filter_function

    def apply(self, model):
        # Extract the bias from Conv node
        model = model.transform(ExtractBiasFromConv())
        # Gemm operations are not supported by FINN, so we convert them to MatMul
        model = model.transform(GemmToMatMul())
        model = model.transform(FoldTransposeIntoQuantInit())
        # Make sure the datatypes exist, these are required for folding the weights
        model = model.transform(InferDataTypes())
        # Fold weights
        model = model.transform(FoldQuantWeights())
        # Convert activations
        model = model.transform(
            ConvertQuantActToMultiThreshold(
                filter_function=self._filter_function,
            )
        )
        # Recompute datatypes
        model = model.transform(InferDataTypes())
        # Convert AvgPool -> Mul -> Trunc structure to QuantAvgPool2d
        model = model.transform(AvgPoolAndTruncToQuantAvgPool())
        # Remove empty padding if it exists
        model = model.transform(RemoveEmptyPadding())

        return model, False

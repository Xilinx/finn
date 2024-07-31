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

import pytest
import torch
import onnx
from onnx import helper, numpy_helper
import numpy as np
import os
import finn.core.onnx_exec as oxe
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv,
)
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import pynq_part_map
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import onnx
from onnx import helper
import onnxruntime
import torch
import torch.nn as nn
import brevitas.nn as qnn
test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5
export_onnx_path = "softmax_dut_qonnx.onnx"

### Make model wrapper
# 1. make node,


### Test
## 1. Compiler integration
#       1. check all transforms can be applied to a model with a softmax layer
#       2. Check that IP stitching produces valid HLS package

## 2. Functionality test
#       1. Check that we can run cpp/rtl sims
#       2. check values are correct


def create_model():
    '''
    Create a quantized softmax model.
    Input and output are quantized to Int8ActPerTensorFloat, this is to make sure
    that the softmax layer is followed by a Quant node.
    '''
    io_shape = (1, 12, 128, 128)
    class QuantSoftMaxSimple(nn.Module):
        def __init__(self):
            super(QuantSoftMaxSimple, self).__init__()
            # self.input_identity = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat)
            self.output_identity = qnn.QuantIdentity()
            self.softmax = nn.Softmax(dim=3) # softmax along the last dimension

        def forward(self, x):
            # x = self.input_identity(x)
            x = self.softmax(x)
            x = self.output_identity(x)
            return x

    dut = QuantSoftMaxSimple()
    input = torch.randn(io_shape)
    export_qonnx(dut, input, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    # set the model input to INT8
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    # import pdb; pdb.set_trace()
    return model

@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd3", "simd4"])
@pytest.mark.fpgadataflow
def test_convert_to_hw_softmax_layer(exec_mode, simd):
    '''
    Test that all transofrmations can be applied to a model with a softmax layer.
    '''
    # Create the qonnx model
    # modelproto = create_softmax_graph()

    model = create_model()
    simd = int(simd[-1])
    folding_config = {
        "Defaults": {},
        "QuantSoftmax_0": {
            "simd": simd
        }
    }
    try:
        model = model.transform(ConvertQONNXtoFINN())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(to_hw.InferQuantSoftmax())
        model = model.transform(GiveUniqueNodeNames())
        # isolate fpga dataflow layers
        parent_model = model.transform(CreateDataflowPartition())
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node_path = getCustomOp(sdp_node).get_nodeattr("model")
        model = ModelWrapper(sdp_node_path)
        model = model.transform(ApplyConfig(folding_config))
        model = model.transform(SpecializeLayers(test_fpga_part))
        if exec_mode == "cppsim":
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
            model = model.transform(SetExecMode("cppsim"))
        elif exec_mode == "rtlsim":
            model = model.transform(SetExecMode("rtlsim"))
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
            # model = model.transform(PrepareRTLSim())
    except Exception as e:
        pytest.fail(f"Failed to transform the model: {str(e)}")

    # oxe.execute_onnx()

def test_fpgadataflow_quantsoftmax():
    # Create the qonnx model
    # create_model()
    model = create_model()
    try:
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(FoldConstants())
        model = model.transform(to_hw.InferQuantSoftmax())
        model = model.transform(SpecializeLayers(test_fpga_part))

    except Exception as e:
        pytest.fail(f"Failed to transform the model: {str(e)}")
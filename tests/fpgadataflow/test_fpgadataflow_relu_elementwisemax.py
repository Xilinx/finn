# Copyright (C) 2025, Advanced Micro Devices, Inc.
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

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferReLUAsElementwiseMax,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


# Creates a model executing a ReLU operation
def create_relu_model_onnx(inp_dtype, inp_shape):
    # Create a node representing the binary elementwise operation
    node = oh.make_node(
        op_type="Relu",
        inputs=["inp"],
        outputs=["out"],
    )
    if inp_dtype == "FLOAT16":
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT16, inp_shape)
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT16, inp_shape)
    else:
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, inp_shape)
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT, inp_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[inp], outputs=[out], name="relu-eltwisemax")
    model = ModelWrapper(qonnx_make_model(graph, producer_name="relu-eltwisemax"))

    # Add datatype annotation to the value info of tensors
    model.set_tensor_datatype("inp", DataType[inp_dtype])
    model.set_tensor_datatype("out", DataType[inp_dtype])

    return model


# Data type of the input elements
@pytest.mark.parametrize("inp_dtype", ["INT8", "FLOAT32", "FLOAT16", "FIXED<8,3>"])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [[4], [3, 32, 1, 16]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# Exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_relu_elementwisemax(inp_dtype, inp_shape, pe, exec_mode):
    # Make dummy model for testing
    model = create_relu_model_onnx(inp_dtype, inp_shape)
    # Prepare the execution context
    context = {"inp": gen_finn_dt_tensor(DataType[inp_dtype], inp_shape)}
    # Compute ground-truth output in software
    o_ref = np.maximum(context["inp"], 0)

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(InferReLUAsElementwiseMax())

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseMax"
    # Execute the onnx model to collect the result
    o_hw = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_hw == o_ref)

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseMax_hls"

    getCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    model = model.transform(SetExecMode(exec_mode))
    model = model.transform(GiveUniqueNodeNames())
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    else:
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    # Execute the onnx model to collect the result
    o_sim = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_sim == o_ref)

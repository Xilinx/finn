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
import scipy.special
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
    InferElementwiseFunctionOperation,
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

# Mapping of ElementwiseFunctionOperation specializations to numpy reference
# implementation functions
NUMPY_REFERENCES = {
    "ElementwiseRelu": lambda x: np.maximum(x, 0),
    "ElementwiseExp": np.exp,
    "ElementwiseErf": scipy.special.erf,
    "ElementwisePow": lambda x: np.power(x, 2.0),  # Default test with exponent=2
}


# Creates a model executing a elementwise function operation
def create_elementwise_function_operation_onnx(op_type, inp_dtype, out_dtype, inp_shape):
    # Remove "Elementwise" from op_type string which is the onnx ops op_type
    onnx_op_type = op_type[11:]
    # Automatically derive the output shape
    out_shape = inp_shape
    
    # Special handling for Pow operation
    if onnx_op_type == "Pow":
        # Create constant exponent tensor
        exp_tensor = oh.make_tensor(
            name="exponent",
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=[2.0]  # Using exponent=2 for testing
        )
        # Create a node representing the elementwise operation
        node = oh.make_node(
            op_type=onnx_op_type,
            inputs=["inp", "exponent"],
            outputs=["out"],
        )
        initializer = [exp_tensor]
    else:
        # Create a node representing the elementwise operation
        node = oh.make_node(
            op_type=onnx_op_type,
            inputs=["inp"],
            outputs=["out"],
        )
        initializer = []
    if inp_dtype == "FLOAT16":
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT16, inp_shape)
    else:
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, inp_shape)
    if out_dtype == "FLOAT16":
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT16, out_shape)
    else:
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[inp], outputs=[out], initializer=initializer, name="elementwise-function")
    model = ModelWrapper(qonnx_make_model(graph, producer_name="elementwise-function"))

    # Add datatype annotation to the value info of input and output tensors
    model.set_tensor_datatype("inp", DataType[inp_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])

    return model


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    [
        # Test all Numpy references specified above
        *NUMPY_REFERENCES.keys()
    ],
)
# Data type of the input elements
@pytest.mark.parametrize(
    "inp_dtype",
    ["FLOAT32", "FLOAT16", "INT6", "FIXED<8,3>"],
)
# Shape of the input
@pytest.mark.parametrize("inp_shape", [[8]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2])
# Exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_function_operation(op_type, inp_dtype, inp_shape, pe, exec_mode):
    if not op_type.endswith("Relu"):
        if not inp_dtype.startswith("FLOAT"):
            pytest.skip("Non-float inputs are not yet supported for functions except Relu.")
    out_dtype = inp_dtype
    # Make dummy model for testing
    model = create_elementwise_function_operation_onnx(op_type, inp_dtype, out_dtype, inp_shape)
    # Prepare the execution context
    context = {
        "inp": gen_finn_dt_tensor(DataType[inp_dtype], inp_shape),
    }

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(InferElementwiseFunctionOperation())

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}"

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

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

    # Compute ground-truth output in software
    inp = context["inp"]

    o_expected = numpy_reference(inp)
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    if op_type.endswith("Relu"):
        assert np.all(o_expected == o_produced)
    else:
        if inp_dtype == "FLOAT16":
            assert np.allclose(o_expected, o_produced, rtol=1e-3, atol=2**-13)
        else:
            assert np.allclose(o_expected, o_produced)

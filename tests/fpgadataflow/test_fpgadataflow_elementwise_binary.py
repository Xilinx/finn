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
    InferElementwiseBinaryOperation,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
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
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

# Mapping of ElementwiseBinaryOperation specializations to numpy reference
# implementation functions
NUMPY_REFERENCES = {
    "ElementwiseAdd": np.add,
    "ElementwiseSub": np.subtract,
    "ElementwiseMul": np.multiply,
    # TODO: "ElementwiseDiv": np.divide, Cannot guarantee non-zero test input
    # TODO: "ElementwiseMod": np.mode / np.fmod
    "ElementwiseAnd": np.logical_and,
    "ElementwiseOr": np.logical_or,
    "ElementwiseXor": np.logical_xor,
    "ElementwiseEqual": np.equal,
    "ElementwiseLess": np.less,
    "ElementwiseLessOrEqual": np.less_equal,
    "ElementwiseGreater": np.greater,
    "ElementwiseGreaterOrEqual": np.greater_equal,
    "ElementwiseBitwiseAnd": np.bitwise_and,
    "ElementwiseBitwiseOr": np.bitwise_or,
    "ElementwiseBitwiseXor": np.bitwise_xor,
    # TODO: "ElementwiseBitShift": np.left_shift / np.right_shift
    # TODO: "ElementwisePow": np.power
}


# Creates a model executing a binary elementwise operation
def create_elementwise_binary_operation_onnx(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
):
    # Remove "Elementwise" from op_type string which is the onnx ops op_type
    onnx_op_type = op_type[11:]
    # Automatically derive the output shape by broadcasting the inputs
    out_shape = np.broadcast_shapes(lhs_shape, rhs_shape)
    # Create a node representing the binary elementwise operation
    node = oh.make_node(
        op_type=onnx_op_type,
        inputs=["in_x", "in_y"],
        outputs=["out"],
    )
    if lhs_dtype == "FLOAT16":
        lhs = oh.make_tensor_value_info("in_x", TensorProto.FLOAT16, lhs_shape)
    else:
        lhs = oh.make_tensor_value_info("in_x", TensorProto.FLOAT, lhs_shape)
    if rhs_dtype == "FLOAT16":
        rhs = oh.make_tensor_value_info("in_y", TensorProto.FLOAT16, rhs_shape)
    else:
        rhs = oh.make_tensor_value_info("in_y", TensorProto.FLOAT, rhs_shape)
    if out_dtype == "FLOAT16":
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT16, out_shape)
    else:
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[lhs, rhs], outputs=[out], name="elementwise-binary")
    model = ModelWrapper(qonnx_make_model(graph, producer_name="elementwise-binary"))

    # Add datatype annotation to the value info of input tensors
    model.set_tensor_datatype("in_x", DataType[lhs_dtype])
    model.set_tensor_datatype("in_y", DataType[rhs_dtype])
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
# Data type of the left-hand-side and right-hand-side input elements
@pytest.mark.parametrize(
    "lhs_dtype_rhs_dtype",
    [
        ("INT8", "INT8"),
        ("INT8", "FLOAT32"),
        ("FLOAT32", "FLOAT32"),
        ("FLOAT16", "FLOAT16"),
        ("FIXED<8,4>", "FIXED<10,5>"),
    ],
)
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1], [1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize(
    "rhs_shape",
    [
        [3, 32, 1, 16],
    ],
)
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["in_x"], ["in_y"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# Exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation(
    op_type, lhs_dtype_rhs_dtype, lhs_shape, rhs_shape, pe, initializers, exec_mode
):
    lhs_dtype, rhs_dtype = lhs_dtype_rhs_dtype
    if "Bitwise" in op_type and not ("INT" in lhs_dtype and "INT" in rhs_dtype):
        pytest.skip(
            "Non-integer datatypes are not meaningful for bitwise ops, skipping those tests."
        )
    if op_type in ["ElementwiseAnd", "ElementwiseOr", "ElementwiseXor"] and not (
        "INT" in lhs_dtype and "INT" in rhs_dtype
    ):
        pytest.skip(
            "Non-integer datatypes are not meaningful for logical ops, skipping those tests."
        )
    out_dtype = "FLOAT16" if lhs_dtype == "FLOAT16" and rhs_dtype == "FLOAT16" else "FLOAT32"
    # Make dummy model for testing
    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )
    # Prepare the execution context
    context = {
        "in_x": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "in_y": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(InferElementwiseBinaryOperation())

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
    lhs = context["in_x"]
    rhs = context["in_y"]
    # convert container dtype to ensure execution of e.g., bitwise ops
    if "Bitwise" in op_type:
        lhs = lhs.astype(np.int64) if lhs_dtype.startswith("INT") else lhs
        rhs = rhs.astype(np.int64) if rhs_dtype.startswith("INT") else rhs

    o_expected = numpy_reference(lhs, rhs)
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    if out_dtype == "FLOAT16":
        if op_type in ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"]:
            # Equivalence checking is more relaxed for arithmetic operations
            # numpy casts fp16 to fp32, computes in fp32, casts result to fp16
            assert np.allclose(o_expected, o_produced, rtol=1e-3, atol=2**-13)
        else:
            assert np.all(o_expected == o_produced)
    else:
        # Compare the expected to the produced for exact equality
        assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize(
    "op_type",
    ["ElementwiseAdd", "ElementwiseMul"],
)
# Data type of the left-hand-side and right-hand-side input elements
@pytest.mark.parametrize(
    "lhs_dtype_rhs_dtype",
    [
        ("INT8", "INT8"),
        ("INT8", "FLOAT32"),
        ("FLOAT32", "FLOAT32"),
        ("FLOAT16", "FLOAT16"),
        ("FIXED<8,4>", "FIXED<10,5>"),
    ],
)
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [[3, 1, 7, 1]])
# Shape of the right-hand-side input
@pytest.mark.parametrize("rhs_shape", [[3, 32, 1, 16]])
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [[], ["in_x"], ["in_y"]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [2])
# mem_mode
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_stitched_ip(
    op_type, lhs_dtype_rhs_dtype, lhs_shape, rhs_shape, pe, initializers, mem_mode
):
    lhs_dtype, rhs_dtype = lhs_dtype_rhs_dtype
    out_dtype = "FLOAT16" if lhs_dtype == "FLOAT16" and rhs_dtype == "FLOAT16" else "FLOAT32"
    # Make dummy model for testing
    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )
    # Prepare the execution context
    context = {
        "in_x": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "in_y": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape),
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}"

    getCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)
    getCustomOp(model.graph.node[0]).set_nodeattr("mem_mode", mem_mode)

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Specializes all nodes to be implemented as HLS backend
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"

    # Try to minimize the bit-widths of all data types involved
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(HLSSynthIP())

    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    lhs = context["in_x"]
    rhs = context["in_y"]

    o_expected = numpy_reference(lhs, rhs)

    # node-by-node rtlsim
    o_produced = execute_onnx(model, context)[model.graph.output[0].name]

    if out_dtype == "FLOAT16":
        # Equivalence checking is more relaxed for arithmetic operations in fp16
        # numpy casts fp16 to fp32, computes in fp32, casts result to fp16
        assert np.allclose(o_expected, o_produced, rtol=1e-3, atol=2**-14)
    else:
        # Compare the expected to the produced for exact equality
        assert np.all(o_produced == o_expected)

    # prepare for stitched ip rtlsim
    model = model.transform(InsertAndSetFIFODepths("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(
        CreateStitchedIP(
            "xczu7ev-ffvc1156-2-e",
            10,
            vitis=False,
        )
    )

    # Tensor names might have changed during the test, so assembling an updated context dict
    io_dict = {}
    if not initializers:
        io_dict[model.graph.input[0].name] = lhs
        io_dict[model.graph.input[1].name] = rhs
    elif len(initializers) == 1:
        if initializers[0] == "in_x":
            io_dict[model.graph.input[0].name] = rhs
        elif initializers[0] == "in_y":
            io_dict[model.graph.input[0].name] = lhs
    # stitched-ip rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    o_produced = execute_onnx(model, io_dict)[model.graph.output[0].name]

    if out_dtype == "FLOAT16":
        # Equivalence checking is more relaxed for arithmetic operations in fp16
        # numpy casts fp16 to fp32, computes in fp32, casts result to fp16
        assert np.allclose(o_expected, o_produced, rtol=1e-3, atol=2**-14)
    else:
        # Compare the expected to the produced for exact equality
        assert np.all(o_produced == o_expected)

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
    InferElementwiseFunctionOperation,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def create_pow_model_onnx(inp_dtype, out_dtype, inp_shape, exponent):
    """Create ONNX model with Pow operation."""
    out_shape = inp_shape
    
    # Create constant exponent tensor
    exp_tensor = oh.make_tensor(
        name="exponent",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[float(exponent)]
    )
    
    # Create Pow node
    node = oh.make_node(
        op_type="Pow",
        inputs=["inp", "exponent"],
        outputs=["out"],
    )
    
    if inp_dtype == "FLOAT16":
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT16, inp_shape)
    else:
        inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, inp_shape)
    
    if out_dtype == "FLOAT16":
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT16, out_shape)
    else:
        out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    
    # Create graph
    graph = oh.make_graph(
        [node], 
        inputs=[inp], 
        outputs=[out], 
        initializer=[exp_tensor],
        name="pow-test"
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pow-test"))
    
    # Add datatype annotations
    model.set_tensor_datatype("inp", DataType[inp_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])
    
    return model


# Test different exponents
@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0])
# Data type of the input elements - only FLOAT32 supported for Pow
@pytest.mark.parametrize("inp_dtype", ["FLOAT32"])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [[8], [1, 4, 4]])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2])
# Exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_pow_operation(exponent, inp_dtype, inp_shape, pe, exec_mode):
    """Test ElementwisePow operation with various exponents."""
    
    out_dtype = inp_dtype
    
    # Create model
    model = create_pow_model_onnx(inp_dtype, out_dtype, inp_shape, exponent)
    
    # Prepare execution context with test data
    # Use positive values for fractional exponents
    if exponent == 0.5:
        # For sqrt, use positive values only
        inp_data = np.abs(gen_finn_dt_tensor(DataType[inp_dtype], inp_shape))
    else:
        inp_data = gen_finn_dt_tensor(DataType[inp_dtype], inp_shape)
    
    context = {"inp": inp_data}
    
    # Infer shapes and datatypes
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    
    # Convert to ElementwisePow
    model = model.transform(InferElementwiseFunctionOperation())
    
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwisePow"
    
    # Check that exponent attribute was set correctly using custom op
    node = model.graph.node[0]
    custom_op = getCustomOp(node)
    exp_attr = custom_op.get_nodeattr("exponent")
    assert exp_attr == exponent, f"Expected exponent {exponent}, got {exp_attr}"
    
    # Specialize for HLS
    model = model.transform(SpecializeLayers("xcvu9p-flgb2104-2-i"))
    
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwisePow_hls"
    
    # Set PE
    getCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)
    
    # Set execution mode
    model = model.transform(SetExecMode(exec_mode))
    model = model.transform(GiveUniqueNodeNames())
    
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    else:
        model = model.transform(PrepareIP("xcvu9p-flgb2104-2-i", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    
    # Compute expected output
    o_expected = np.power(inp_data, exponent)
    
    # Execute model
    o_produced = execute_onnx(model, context)["out"]
    
    # Compare results (only FLOAT32 supported)
    assert np.allclose(o_expected, o_produced, rtol=1e-5)


# Test that non-constant exponents are not converted
def test_pow_non_constant_exponent():
    """Test that Pow with non-constant exponent is not converted to ElementwisePow."""
    
    # Create model with Pow where exponent is an input (not constant)
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    exp = oh.make_tensor_value_info("exp", TensorProto.FLOAT, [1, 4])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    
    pow_node = oh.make_node("Pow", ["inp", "exp"], ["out"])
    
    graph = oh.make_graph([pow_node], inputs=[inp, exp], outputs=[out], name="pow-non-const")
    model = ModelWrapper(qonnx_make_model(graph))
    
    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("exp", DataType["FLOAT32"])
    model.set_tensor_datatype("out", DataType["FLOAT32"])
    
    # Apply transformation
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseFunctionOperation())
    
    # Check that node was NOT converted
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Pow"  # Still Pow, not ElementwisePow


if __name__ == "__main__":
    # Run a simple test
    test_elementwise_pow_operation(2.0, "FLOAT32", [8], 1, "cppsim")
    print("ElementwisePow test passed!")
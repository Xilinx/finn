# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Testing framework
import pytest

# Numpy math and arrays
import numpy as np

# ONNX graph and tensor utility
from onnx import TensorProto
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx

# Registry of all QONNX CustomOps
from qonnx.custom_op.registry import getCustomOp

# Graph transformation giving unique names to each node in a QONNX model graph
from qonnx.transformation.general import GiveUniqueNodeNames

# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# Utility for wrapping onnx graphs and generating tensor of FINN datatypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


# Specializes all nodes to be implemented as HLS backend
def specialize_hls(model: ModelWrapper):
    # Mark all nodes to be specialized as HLS backend implementations
    for node in model.graph.node:  # noqa: Duplicate test setup code
        # Get the CustomOp instance of the node to get access to the node
        # attributes
        inst = getCustomOp(node)
        # Note: only HLS-based layers execute C++ Simulation
        inst.set_nodeattr("preferred_impl_style", "hls")
    # Turn all HWCustomOp layers into HLS specializations
    return model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))


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
def mock_elementwise_binary_operation(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
):
    # Automatically derive the output shape by broadcasting the inputs
    out_shape = np.broadcast_shapes(lhs_shape, rhs_shape)
    # Create a node representing the binary elementwise operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type=op_type,
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Just one input
        inputs=["lhs", "rhs"],
        # Enumerate the outputs
        outputs=["out"],
        # Data type of the left-hand-side input elements
        lhs_dtype=lhs_dtype,
        # Data type of the right-hand-side input elements
        rhs_dtype=rhs_dtype,
        # Data type of the output elements
        out_dtype=out_dtype,
        # Shape of the left-hand-side input
        lhs_shape=lhs_shape,
        # Shape of the right-hand-side input
        rhs_shape=rhs_shape,
        # Shape of the output, mus correspond to multi-directional
        # broadcasting of the left- and right-hand-side
        out_shape=out_shape,
        # Number of elements to process in parallel
        PE=pe,
    )
    # Construct the input tensor value infos
    lhs = oh.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs_shape)
    rhs = oh.make_tensor_value_info("rhs", TensorProto.FLOAT, rhs_shape)
    # Construct output tensor value infos
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph(
        [node], inputs=[lhs, rhs], outputs=[out], name="elementwise-binary"
    )
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(
        qonnx_make_model(graph, producer_name="elementwise-binary")
    )

    # Add datatype annotation to the value info of input tensors
    model.set_tensor_datatype("lhs", DataType[lhs_dtype])
    model.set_tensor_datatype("rhs", DataType[rhs_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])

    # Return the wrapped onnx model
    return model


# Operator type to be tested
@pytest.mark.parametrize("op_type", [  # noqa: Duplicate test setup
    # Test all Numpy references specified above
    *NUMPY_REFERENCES.keys()
])
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [
    [3, 1, 7, 1], [1]
])
# Shape of the right-hand-side input
@pytest.mark.parametrize("rhs_shape", [
    [3, 32, 1, 16],
])
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [
    [], ["lhs"], ["rhs"], ["lhs", "rhs"]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
def test_elementwise_binary_operation_python(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe,
        initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(  # noqa: Duplicate test setup
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    }

    # Turn selected inputs into initializers
    for name in initializers:
        model.set_initializer(name, context[name])

    # Get the numpy reference implementation for this operation
    numpy_reference = NUMPY_REFERENCES[op_type]

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume out_type to be always of the same kind as the inputs but
        # with bit-width >= the bit-width of either of the inputs. Then,
        # representing the inputs as out_type for numpy simulation is safe.
        context["lhs"].astype(DataType[out_dtype].to_numpy_dt()),
        context["rhs"].astype(DataType[out_dtype].to_numpy_dt()),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize("op_type", [  # noqa: Duplicate test setup
    # Test all Numpy references specified above
    *NUMPY_REFERENCES.keys(),
])
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [
    [3, 1, 7, 1], [1]
])
# Shape of the right-hand-side input
@pytest.mark.parametrize("rhs_shape", [
    [3, 32, 1, 16],
])
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [
    [], ["lhs"], ["rhs"], ["lhs", "rhs"]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_cppsim(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe,
        initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(  # noqa: Duplicate test setup
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
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
    model = specialize_hls(model)
    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume out_type to be always of the same kind as the inputs but
        # with bit-width >= the bit-width of either of the inputs. Then,
        # representing the inputs as out_type for numpy simulation is safe.
        context["lhs"].astype(DataType[out_dtype].to_numpy_dt()),
        context["rhs"].astype(DataType[out_dtype].to_numpy_dt()),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)


# Operator type to be tested
@pytest.mark.parametrize("op_type", [  # noqa: Duplicate test setup
    # Test all Numpy references specified above
    *NUMPY_REFERENCES.keys()
])
# Data type of the left-hand-side input elements
@pytest.mark.parametrize("lhs_dtype", ["INT8"])
# Data type of the right-hand-side input elements
@pytest.mark.parametrize("rhs_dtype", ["INT8"])
# Data type of the output elements
@pytest.mark.parametrize("out_dtype", ["INT32"])
# Shape of the left-hand-side input
@pytest.mark.parametrize("lhs_shape", [
    [3, 1, 7, 1], [1]
])
# Shape of the right-hand-side input
@pytest.mark.parametrize("rhs_shape", [
    [3, 32, 1, 16],
])
# Which inputs to set as initializers
@pytest.mark.parametrize("initializers", [
    [], ["lhs"], ["rhs"], ["lhs", "rhs"]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_elementwise_binary_operation_rtlsim(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe,
        initializers
):
    # Make dummy model for testing
    model = mock_elementwise_binary_operation(  # noqa: Duplicate test setup
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape, pe
    )
    # Prepare the execution context
    context = {
        "lhs": gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape),
        "rhs": gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
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
    model = specialize_hls(model)
    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))  # noqa
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = numpy_reference(
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Assume out_type to be always of the same kind as the inputs but
        # with bit-width >= the bit-width of either of the inputs. Then,
        # representing the inputs as out_type for numpy simulation is safe.
        context["lhs"].astype(DataType[out_dtype].to_numpy_dt()),
        context["rhs"].astype(DataType[out_dtype].to_numpy_dt()),
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Testing framework
import pytest

# Numpy math and arrays
import numpy as np

# Create temporary files automatically deleted after integration test
import tempfile

# PyTorch required for integration test
import torch

# Export brevitas models to QONNX representation in integration test
from brevitas.export import export_qonnx

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

# Cleanup transformations required after QONNX model import
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
)

# Adds data layout annotations to the model graph to correctly convert
# quantizers to multi-thresholds
from qonnx.transformation.infer_data_layouts import InferDataLayouts

# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# Utility for wrapping onnx graphs and generating tensor of FINN datatypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim

# Mapping to hardware operators of the two operations relevant for the
# integration test
# Note: The integration test serves as the test-case for InferUnsqueeze
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferUnsqueeze

# Synthesizes HLS code generated from an operator to IP block
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP

# Transformations preparing the operators for C++ and RTL simulation
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

# Converts between QONNX and FINN dialect of ONNX representation
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

# Standard set of streamlining transformations delivered with FINN
from finn.transformation.streamline import Streamline

# Checks whether a node is a fpgadataflow backend node handled by FINN
from finn.util.fpgadataflow import is_fpgadataflow_node


# Specializes all nodes to be implemented as HLS backend
def specialize_hls(model: ModelWrapper):
    # Mark all nodes to be specialized as HLS backend implementations
    for node in model.graph.node:  # noqa: Duplicate test setup code
        # Skip non-fpgadataflow backend operators as these do not have the
        # preferred_impl_style attribute
        if is_fpgadataflow_node(node):
            # Get the CustomOp instance of the node to get access to the node
            # attributes
            inst = getCustomOp(node)
            # Note: only HLS-based layers execute C++ Simulation
            inst.set_nodeattr("preferred_impl_style", "hls")
    # Turn all HWCustomOp layers into HLS specializations
    return model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))


# Creates a dummy model for testing the Unsqueeze operation
def mock_unsqueeze(axes, inp_dtype, out_dtype, inp_shape, out_shape, pe):
    # Create a node representing the unsqueeze operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type="Unsqueeze",
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Just one input
        inputs=["inp"],
        # Enumerate the outputs
        outputs=["out"],
        # Axes to be squeezed
        axes=axes,
        # Data type of the input elements
        inp_dtype=inp_dtype,
        # Data type of the output elements
        out_dtype=inp_dtype,
        # Shape of the input
        inp_shape=inp_shape,
        # Shape of the output
        out_shape=out_shape,
        # Number of elements to process in parallel
        PE=pe,
    )
    # Construct the input tensor value infos
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, inp_shape)
    # Construct output tensor value infos
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[inp], outputs=[out], name="unsqueeze")
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(
        qonnx_make_model(graph, producer_name="unsqueeze")
    )

    # Add datatype annotation to the value info of input tensors
    model.set_tensor_datatype("inp", DataType[inp_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])

    # Return the wrapped onnx model
    return model


@pytest.mark.xfail(reason="Outstanding ONNX opset issue")
# Axes to be squeezed
@pytest.mark.parametrize(  # noqa: Duplicate test setup
    "axes", [(1,), (1, 3), (-1,)]
)
# Data type of the input elements
@pytest.mark.parametrize("inp_dtype", ["INT8"])
@pytest.mark.parametrize("out_dtype", ["INT8"])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [
    [3, 7]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1])
def test_unsqueeze_python(axes, inp_dtype, out_dtype, inp_shape, pe):
    # Derive the unsqueezed output shape
    out_shape = np.expand_dims(np.zeros(inp_shape), axis=axes).shape  # noqa
    # Make dummy model for testing
    model = mock_unsqueeze(  # noqa: Duplicate test setup
        axes, inp_dtype, out_dtype, inp_shape, out_shape, pe
    )
    # Prepare the execution context
    context = {  # noqa: Duplicate test setup
        "inp": gen_finn_dt_tensor(DataType[inp_dtype], inp_shape),
    }

    # Test running shape and data type inference on the model graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = np.expand_dims(context["inp"], axes)
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)
    # Compare the produced shape to the expected squeezed shape
    assert o_produced.shape == out_shape


@pytest.mark.xfail(reason="Outstanding ONNX opset issue")
# Axes to be squeezed
@pytest.mark.parametrize(  # noqa: Duplicate test setup
    "axes", [(1,), (1, 3), (-1,)]
)
# Data type of the input elements
@pytest.mark.parametrize("inp_dtype", ["INT8"])
@pytest.mark.parametrize("out_dtype", ["INT8"])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [
    [3, 7]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1])
def test_unsqueeze_cppsim(axes, inp_dtype, out_dtype, inp_shape, pe):
    # Derive the unsqueezed output shape
    out_shape = np.expand_dims(np.zeros(inp_shape), axis=axes).shape  # noqa
    # Make dummy model for testing
    model = mock_unsqueeze(  # noqa: Duplicate test setup
        axes, inp_dtype, out_dtype, inp_shape, out_shape, pe
    )
    # Prepare the execution context
    context = {  # noqa: Duplicate test setup
        "inp": gen_finn_dt_tensor(DataType[inp_dtype], inp_shape),
    }

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
    o_expected = np.expand_dims(context["inp"], axes)
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)
    # Compare the produced shape to the expected squeezed shape
    assert o_produced.shape == out_shape


@pytest.mark.xfail(reason="Outstanding ONNX opset issue")
# Axes to be squeezed
@pytest.mark.parametrize(  # noqa: Duplicate test setup
    "axes", [(1,), (1, 3), (-1,)]
)
# Data type of the input elements
@pytest.mark.parametrize("inp_dtype", ["INT8"])
@pytest.mark.parametrize("out_dtype", ["INT8"])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [
    [3, 1, 7, 1]
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1])
def test_unsqueeze_rtlsim(axes, inp_dtype, out_dtype, inp_shape, pe):
    # Derive the unsqueezed output shape
    out_shape = np.expand_dims(np.zeros(inp_shape), axis=axes).shape  # noqa
    # Make dummy model for testing
    model = mock_unsqueeze(  # noqa: Duplicate test setup
        axes, inp_dtype, out_dtype, inp_shape, out_shape, pe
    )
    # Prepare the execution context
    context = {  # noqa: Duplicate test setup
        "inp": gen_finn_dt_tensor(DataType[inp_dtype], inp_shape),
    }

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
    o_expected = np.expand_dims(context["inp"], axes)
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare the expected to the produced for exact equality
    assert np.all(o_produced == o_expected)
    # Compare the produced shape to the expected squeezed shape
    assert o_produced.shape == out_shape


# Axis to unsqueeze
@pytest.mark.parametrize("axis", [0, 1])
# Shape of the input
@pytest.mark.parametrize("inp_shape", [
    [1, 2], [2, 1, 4], [3, 1, 4],
])
# Number of elements to process in parallel
@pytest.mark.parametrize("pe", [1, 2])
def test_integration_unsqueeze(axis, inp_shape, pe):
    # PyTorch model wrapping the component(s) to be tested
    class Dummy(torch.nn.Module):
        # Sets up the test model and initializes parameters
        def __init__(self):
            # Initialize the PyTorch Module superclass
            super().__init__()

        # Model forward squeezing the input
        def forward(self, x):  # noqa: Forward may be static...
            return torch.unsqueeze(x, dim=axis)

    # Create the test instance of the dummy model
    model = Dummy()
    # Create dummy test inputs
    inp = torch.randn(*inp_shape)
    # Do a forward pass with model in training mode to calibrate the quantizers
    _ = model(inp)
    # Switch model to evaluation mode to keep parameters fixed for export
    model = model.eval()
    # Do not accumulate gradients while generating test output
    with torch.no_grad():
        # Model forward pass generating the expected output for verification
        out_expected = model(inp).numpy().astype(np.float32)
    # Generate a temporary directory for running this test
    with tempfile.TemporaryDirectory() as tmp:
        # Export the model to ONNX format to be consumed by FINN
        export_qonnx(model, (inp,), tmp + "/model.onnx")  # noqa: Duplicate
        # Wrap the model with QONNX wrapper for transformations
        model = ModelWrapper(tmp + "/model.onnx")
        # Cleanup transformations preparing the model to be consumed by FINN
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataLayouts())
        model = model.transform(ConvertQONNXtoFINN())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(RemoveUnusedTensors())
        # Do a single round of standard streamlining of the model graph
        model = model.transform(Streamline())
        # Convert layers to hardware custom operations
        model = model.transform(InferUnsqueeze())

        # Apply folding config to set the PE parallelism for hardware layers
        model = model.transform(ApplyConfig({  # noqa: Duplicate test code
            "Defaults": {"PE": [pe, ["Unsqueeze"]]}
        }))

        # Prepare the execution context with dummy data from above and input
        # node names extracted from transformed modelo graph
        context = {  # noqa: Duplicate
            model.graph.input[0].name: inp.numpy().astype(np.float32)
        }

        # Set model execution mode to python simulation
        model = model.transform(SetExecMode("python"))  # noqa: Duplicate
        model = model.transform(GiveUniqueNodeNames())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the model has been
        # streamlined, which may involve rounding
        assert np.allclose(out_produced, out_expected, atol=1e-3), \
            "Python simulation verification failed"

        # # Specializes all nodes to their backend implementation
        model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

        # Set model execution mode to C++ simulation
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(GiveUniqueNodeNames())
        # Generates the C++ source and compiles the C++ simulation
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the model has been
        # streamlined, which may involve rounding
        assert np.allclose(out_produced, out_expected, atol=1e-3), \
            "C++ simulation verification failed"

        # Set model execution mode to RTL simulation
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        # Generates the C++ source and compiles the RTL simulation
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))  # noqa
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        # Execute the onnx model to collect the result
        out_produced = execute_onnx(model, context)[model.graph.output[0].name]
        # Compare the expected to the produced
        # Note: Only test for close up to some tolerance as the model has been
        # streamlined, which may involve rounding
        assert np.allclose(out_produced, out_expected, atol=1e-3), \
            "RTL simulation verification failed"

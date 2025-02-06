# Testing framework
import pytest

# Use numpy for python execution / computing the ground truth expected values
import numpy as np

# Protobuf onnx graph node type
from onnx import TensorProto
# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx
# Registry of all QONNX CustomOps
from qonnx.custom_op.registry import getCustomOp
# Utility for wrapping onnx graphs and generating tensor of FINN datatypes
from qonnx.util.basic import qonnx_make_model, gen_finn_dt_tensor

# Graph transformation giving unique names to each node in a QONNX model graph
from qonnx.transformation.general import GiveUniqueNodeNames

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
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


# Creates a model executing mult-head splitting
def mock_split_multi_heads(seq, dim, heads, dtype):
    # Create a node representing the attention heads splitting operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type="SplitMultiHeads",
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Just one input
        inputs=["inp"],
        # Enumerate the outputs
        outputs=[f"out{i}" for i in range(heads)],
        # Number of attention heads to split the input into
        heads=heads,
        # Packed output is not supported for now
        packed=False,
        # Datatype of inputs and outputs
        dtype=dtype,
        # Number of input elements, i.e., embedding dimension
        num_elems=dim,
        # Number of embeddings in the whole input sequence / feature map
        num_inputs=[seq]
    )
    # Construct the input tensor value info
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [seq, dim])
    # Construct output tensor value infos
    out = [oh.make_tensor_value_info(
        f"out{i}", TensorProto.FLOAT, [seq, dim // heads]) for i in range(heads)
    ]
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[inp], outputs=out, name="split")
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name='split'))

    # Add datatype annotation to the value info of input tensor
    model.set_tensor_datatype("inp", DataType[dtype])
    # Add datatype annotation to the value infor of each output tensor
    for out in (f"out{i}" for i in range(heads)):
        model.set_tensor_datatype(out, DataType[dtype])

    # Return the wrapped onnx model
    return model


# Creates a model executing mult-head merging
def mock_merge_multi_heads(seq, dim, heads, dtype):
    # Create a node representing the attention heads merging operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type="MergeMultiHeads",
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Enumerate the inputs
        inputs=[f"inp{i}" for i in range(heads)],
        # Just one output
        outputs=["out"],
        # Number of attention heads to split the input into
        heads=heads,
        # Packed output is not supported for now
        packed=False,
        # Datatype of inputs and outputs
        dtype=dtype,
        # Number of input elements, i.e., embedding dimension
        num_elems=dim // heads,
        # Number of embeddings in the whole input sequence / feature map
        num_inputs=[seq],
        # Assume squeezed output by default
        squeezed=True
    )
    # Construct input tensor value infos
    inp = [oh.make_tensor_value_info(
        f"inp{i}", TensorProto.FLOAT, [seq, dim // heads]) for i in range(heads)
    ]
    # Construct the output tensor value info
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [seq, dim])
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=inp, outputs=[out], name="merge")
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name='merge'))

    # Add datatype annotation to the value infor of each input tensor
    for inp in (f"inp{i}" for i in range(heads)):
        model.set_tensor_datatype(inp, DataType[dtype])
    # Add datatype annotation to the value info of output tensor
    model.set_tensor_datatype("out", DataType[dtype])

    # Return the wrapped onnx model
    return model


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
# Tests splitting of tensors to multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_split_python(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(seq, dim, heads, dtype)

    # Prepare the execution context
    context = {"inp": gen_finn_dt_tensor(DataType[dtype], (seq, dim))}

    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = np.split(context["inp"], heads, axis=-1)  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(heads))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests splitting of tensors to multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_split_cppsim(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(seq, dim, heads, dtype)

    # Prepare the execution context
    context = {"inp": gen_finn_dt_tensor(DataType[dtype], (seq, dim))}

    # Specializes all nodes to be implemented as HLS backend
    model = specialize_hls(model)
    # Set model execution mode to Python simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = np.split(context["inp"], heads, axis=-1)  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(heads))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests splitting of tensors to multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_split_rtlsim(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(seq, dim, heads, dtype)

    # Prepare the execution context
    context = {"inp": gen_finn_dt_tensor(DataType[dtype], (seq, dim))}

    # Specializes all nodes to be implemented as HLS backend
    model = specialize_hls(model)
    # Set model execution mode to Python simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))  # noqa
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = np.split(context["inp"], heads, axis=-1)  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(heads))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])  # noqa: Duplicate, test setup
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
# Tests merging of tensors from multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_merge_python(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_merge_multi_heads(seq, dim, heads, dtype)

    # Create a random input tensor of shape and datatype
    def make_inp_tensor():
        return gen_finn_dt_tensor(DataType[dtype], (seq, dim // heads))

    # Prepare the execution context
    context = {
        f"inp{i}": make_inp_tensor() for i in range(heads)
    }

    # Set model execution mode to Python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = np.concatenate(
        [context[f"inp{i}"] for i in range(heads)], axis=-1
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare expected to produced output
    assert (o_produced == o_expected).all()  # noqa: Unresolved "all" warning


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])  # noqa: Duplicate, test setup
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests merging of tensors from multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_merge_cppsim(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_merge_multi_heads(seq, dim, heads, dtype)

    # Create a random input tensor of shape and datatype
    def make_inp_tensor():
        return gen_finn_dt_tensor(DataType[dtype], (seq, dim // heads))

    # Prepare the execution context
    context = {
        f"inp{i}": make_inp_tensor() for i in range(heads)
    }

    # Specializes all nodes to be implemented as HLS backend
    model = specialize_hls(model)
    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = np.concatenate(
        [context[f"inp{i}"] for i in range(heads)], axis=-1
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare expected to produced output
    assert (o_produced == o_expected).all()  # noqa: Unresolved "all" warning


# Sequence length to simulate, i.e., number of individual inputs to be split
@pytest.mark.parametrize("seq", [64])  # noqa: Duplicate, test setup
# Number of input elements to be split, i.e., size of embedding dimension
@pytest.mark.parametrize("dim", [32])
# Number of heads to split the input into
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["UINT8"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests merging of tensors from multiple attention heads using python mode
# execution
#   Note: No actual attention operation is performed
def test_attention_heads_merge_rtlsim(seq, dim, heads, dtype):
    # Make dummy model for testing
    model = mock_merge_multi_heads(seq, dim, heads, dtype)

    # Create a random input tensor of shape and datatype
    def make_inp_tensor():
        return gen_finn_dt_tensor(DataType[dtype], (seq, dim // heads))

    # Prepare the execution context
    context = {
        f"inp{i}": make_inp_tensor() for i in range(heads)
    }

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
    o_expected = np.concatenate(
        [context[f"inp{i}"] for i in range(heads)], axis=-1
    )
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["out"]

    # Compare expected to produced output
    assert (o_produced == o_expected).all()  # noqa: Unresolved "all" warning

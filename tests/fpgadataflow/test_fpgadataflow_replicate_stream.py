# Testing framework
import pytest  # noqa pytest dependecy is listed in setup.cfg

# Protobuf onnx graph node type
from onnx import TensorProto
# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType  # noqa qonnx dependency is specified
# in setup.cfg as well as in fetch-repos.sh
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper  # noqa: qonnx
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx  # noqa: qonnx
# Utility for wrapping onnx graphs and generating tensor of FINN datatypes
from qonnx.util.basic import qonnx_make_model, gen_finn_dt_tensor  # noqa

# Graph transformation giving unique names to each node in a QONNX model graph
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: qonnx

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim


# Creates a model executing stream replication
def mock_split_multi_heads(num_inputs, num_elems, num, dtype):
    # Create a node representing the stream replication operation
    node = oh.make_node(
        # Operator type from the name of the fpgadataflow hlscustomop
        op_type="ReplicateStream",
        # Specify the domain, i.e., the package to look for the custom operator
        # implementation
        domain="finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        backend="fpgadataflow",
        # Just one input
        inputs=["inp"],
        # Enumerate the outputs
        outputs=[f"out{i}" for i in range(num)],
        # Number of replicas to produce
        num=num,
        # Datatype of inputs and outputs
        dtype=dtype,
        # Number of input elements received in parallel
        num_elems=num_elems,
        # Number of inputs to be processed sequentially
        num_inputs=num_inputs
    )
    # Shape of the input and each output
    shape = [*num_inputs, num_elems]
    # Construct the input tensor value info
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    # Construct output tensor value infos
    out = [oh.make_tensor_value_info(
        f"out{i}", TensorProto.FLOAT, shape) for i in range(num)
    ]
    # Create a graph connecting the node to the inputs and outputs
    graph = oh.make_graph([node], inputs=[inp], outputs=out, name="replicate")
    # Wrap the ONNX graph in QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name='replicate'))

    # Add datatype annotation to the value info of input tensor
    model.set_tensor_datatype("inp", DataType[dtype])
    # Add datatype annotation to the value infor of each output tensor
    for out in (f"out{i}" for i in range(num)):
        model.set_tensor_datatype(out, DataType[dtype])

    # Return the wrapped onnx model
    return model


# Number of inputs to be processed sequentially
@pytest.mark.parametrize(  # noqa Duplicate
    "num_inputs", [[64], [1, 64], [2, 64], [2, 2, 64]]
)
# Number of input elements received in parallel
@pytest.mark.parametrize("num_elems", [32])
# Number of replicas to produce
@pytest.mark.parametrize("num", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["FLOAT32", "UINT8", "INT4"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
# Tests replicating of tensors/streams to multiple outputs using python mode
# execution
def test_replicate_stream_python(num_inputs, num_elems, num, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(num_inputs, num_elems, num, dtype)

    # Prepare the execution context
    context = {
        "inp": gen_finn_dt_tensor(DataType[dtype], (*num_inputs, num_elems))
    }

    # Set model execution mode to python simulation
    model = model.transform(SetExecMode("python"))
    model = model.transform(GiveUniqueNodeNames())

    # Compute ground-truth output in software
    o_expected = [context["inp"] for _ in range(num)]  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(num))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning


# Number of inputs to be processed sequentially
@pytest.mark.parametrize(  # noqa Duplicate
    "num_inputs", [[64], [1, 64], [2, 64], [2, 2, 64]]
)
# Number of input elements received in parallel
@pytest.mark.parametrize("num_elems", [32])
# Number of replicas to produce
@pytest.mark.parametrize("num", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["FLOAT32", "UINT8", "INT4"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests replicating of tensors/streams to multiple outputs using C++ mode
# execution
def test_replicate_stream_cppsim(num_inputs, num_elems, num, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(num_inputs, num_elems, num, dtype)

    # Prepare the execution context
    context = {
        "inp": gen_finn_dt_tensor(DataType[dtype], (*num_inputs, num_elems))
    }

    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Compute ground-truth output in software
    o_expected = [context["inp"] for _ in range(num)]  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(num))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning


# Number of inputs to be processed sequentially
@pytest.mark.parametrize(  # noqa Duplicate
    "num_inputs", [[64], [1, 64], [2, 64], [2, 2, 64]]
)
# Number of input elements received in parallel
@pytest.mark.parametrize("num_elems", [32])
# Number of replicas to produce
@pytest.mark.parametrize("num", [1, 2, 4, 8])
# Datatypes to simulate
@pytest.mark.parametrize("dtype", ["FLOAT32", "UINT8", "INT4"])
# This is a slow running fpgadataflow type of test which requires vivado
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
# Tests replicating of tensors/streams to multiple outputs using RTL mode
# execution
def test_replicate_stream_rtlsim(num_inputs, num_elems, num, dtype):
    # Make dummy model for testing
    model = mock_split_multi_heads(num_inputs, num_elems, num, dtype)

    # Prepare the execution context
    context = {
        "inp": gen_finn_dt_tensor(DataType[dtype], (*num_inputs, num_elems))
    }

    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))  # noqa
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Compute ground-truth output in software
    o_expected = [context["inp"] for _ in range(num)]  # noqa: Duplicate
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)

    # Validate each output separately
    for i, out in enumerate((f"out{i}" for i in range(num))):
        # Compare expected (retrieved by index) to produced (retrieve by key)
        assert (o_produced[out] == o_expected[i]).all()  # noqa: "all" warning

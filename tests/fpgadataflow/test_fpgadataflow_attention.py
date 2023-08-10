# Testing framework
import pytest  # noqa pytest dependecy is listed in setup.cfg

# Utility types and function for creating onnx nodes and graphs
from onnx import TensorProto, helper

# QONNX utility for generating random input data for testing and for creating
# models
from qonnx.util.basic import (  # noqa qonnx dependency is specified in
    # setup.cfg as well as in fetch-repos.sh
    gen_finn_dt_tensor, DataType, qonnx_make_model
)
# Wrapper around ONNX model with some graph manipulation utility
from qonnx.core.modelwrapper import ModelWrapper  # noqa
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx  # noqa
# Graph transformation giving unique names to each node in a QONNX model graph
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa

# FINN graph transformations for preparing simulation (cppsim or rtlsim)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

# Use numpy for python execution / computing the ground truth expected values
import numpy as np
# Numpy compatible implementation of the softmax operation
from scipy.special import softmax


# Generates a QONNX ModelWrapper for testing scaled dot-product attention
def make_single_sdp_modelwrapper_like(
        q, k, v, mask=None, embfold=1, seqfold=1, **dtypes
):
    # Convert unspecified mask to 'none' mode
    mask = 'none' if mask is None else mask

    # Start building the node as a dictionary of attributes
    node_kwargs = {
        # Refer to this operator type by its name
        "op_type": "ScaledDotProductAttention",
        # Execution will try to look up the implementation in the package
        # referred to by the domain
        "domain": "finn.custom_op.fpgadataflow",
        # Execution backend: Required attribute inherited from HLSCustomOp
        "backend": "fpgadataflow",
        # Folding along the embedding dimensions
        "EmbFold": embfold,
        # Folding along the sequence dimensions
        "SeqFold": seqfold
    }

    # Infer the output shape from the input shapes
    o_shape = (q.shape[0], v.shape[1])

    # Create onnx value info of all inputs and outputs assuming float
    # datatypes
    q_info = helper.make_tensor_value_info("Q", TensorProto.FLOAT, q.shape)
    k_info = helper.make_tensor_value_info("K", TensorProto.FLOAT, k.shape)
    v_info = helper.make_tensor_value_info("V", TensorProto.FLOAT, v.shape)
    o_info = helper.make_tensor_value_info("O", TensorProto.FLOAT, o_shape)

    # Collect input and output nodes in order
    inputs, outputs = [q_info, k_info, v_info], [o_info]

    # Collect all inputs/outputs to the operator node
    io_kwargs = {
        "inputs": ["Q", "K", "V"], "outputs": ["O"], "mask_mode": "none"
    }

    # Start building the shape attributes
    shape_kwargs = {
        # Shared embedding dimension of the queries and keys and embedding
        # dimension of the values
        "QKDim": q.shape[1], "VDim": v.shape[1],
        # Shared sequence length of keys and values and sequence length of
        # the queries
        "KVLen": k.shape[0], "QLen": q.shape[0],
    }

    # Start building the datatype attributes
    dtype_kwargs = {
        # Datatypes of the query, key, value inputs and the output
        "QType": "FLOAT32", "KType": "FLOAT32",
        "VType": "FLOAT32", "OType": "FLOAT32",
    }

    # If the optional mask is specified as an input
    if isinstance(mask, np.ndarray) or mask == "input":
        # Add the mask to the input node names
        io_kwargs["inputs"].append("mask")
        # Configure masking mode via io_kwargs as well
        io_kwargs["mask_mode"] = "input"
        # Always infer the mask shape
        mask_shape = (q.shape[0], k.shape[0])
        # Create value info of the mask input
        mask_info = helper.make_tensor_value_info(
            "mask", TensorProto.FLOAT, mask_shape
        )
        # Append the mask input as fourth input node
        inputs.append(mask_info)
        # Add the mask default datatype to the datatype attributes
        dtype_kwargs["MType"] = "FLOAT32"

    # If a causal mask is to be generated during execution
    if mask == "causal":
        # Configure masking mode via io_kwargs as well
        io_kwargs["mask_mode"] = "causal"
        # Add the mask default datatype to the datatype attributes
        dtype_kwargs["MType"] = "FLOAT32"

    # The optional dtypes keyword arguments must describe a subset of the
    # model inputs and outputs
    assert set(dtypes) <= {*dtype_kwargs, "MType"}, \
        "Specified datatype of unknown input or output"

    # Update the datatype attributes according to the keyword arguments
    dtype_kwargs.update({
        key: value.name for key, value in dtypes.items()
    })

    # Create an onnx graph node by unpacking all prepared keyword arguments
    node = helper.make_node(
        **node_kwargs, **io_kwargs, **shape_kwargs, **dtype_kwargs
    )
    # Create a graph out of the operator node and the input/output nodes
    graph = helper.make_graph(
        [node], inputs=inputs, outputs=outputs, name='attention_graph'
    )
    # Wrap the graph in a qonnx model wrapper
    model = ModelWrapper(qonnx_make_model(
        graph, producer_name='attention-model'
    ))

    # Add datatype annotations to all input tensors
    for tensor_name in io_kwargs["inputs"]:
        # Only annotate if a datatype is specified
        if f'{tensor_name}Type' in dtypes:
            # Update the datatype annotation
            model.set_tensor_datatype(
                tensor_name, dtypes[f'{tensor_name}Type']
            )

    # Add datatype annotations to all output tensors
    for tensor_name in io_kwargs["outputs"]:
        # Only annotate if a datatype is specified
        if f'{tensor_name}Type' in dtypes:
            # Update the datatype annotation
            model.set_tensor_datatype(
                tensor_name, dtypes[f'{tensor_name}Type']
            )

    # Return the constructed qonnx model wrapper
    return model


# Size of query and key embedding dimension
@pytest.mark.parametrize("QKDim", [64])
# Size of value embedding dimension
@pytest.mark.parametrize("VDim", [64])
# Length of key and value sequences
@pytest.mark.parametrize("KVLen", [256])
# Length of query sequence
@pytest.mark.parametrize("QLen", [256])
# Different modes to provide a mask
@pytest.mark.parametrize("mask", ["none"])
# Folding along the embedding dimensions
@pytest.mark.parametrize("EmbFold", [64])
# Folding along the sequence dimensions
@pytest.mark.parametrize("SeqFold", [256])
# Datatypes of queries, keys and values, mask and output
@pytest.mark.parametrize("QType", [DataType["UINT16"]])
@pytest.mark.parametrize("KType", [DataType["UINT16"]])
@pytest.mark.parametrize("VType", [DataType["UINT16"]])
@pytest.mark.parametrize("MType", [DataType["UINT16"]])
@pytest.mark.parametrize("OType", [DataType["UINT32"]])
# Tests python implementation of single scaled dot-product attention head
def test_attention_python(
        QKDim, VDim, KVLen, QLen, mask, EmbFold, SeqFold, QType, KType, VType,
        MType, OType
):
    # Generate random input data
    q = gen_finn_dt_tensor(QType, (QLen, QKDim))
    k = gen_finn_dt_tensor(KType, (KVLen, QKDim))
    v = gen_finn_dt_tensor(VType, (KVLen, VDim))

    dtypes = {
        # Datatypes of the query, key, value inputs and the output
        "QType": QType, "KType": KType,
        "VType": VType, "OType": OType,
    }

    # Generate the operator matching the configuration
    model = make_single_sdp_modelwrapper_like(
        q, k, v, mask, EmbFold, SeqFold, **dtypes, MType=MType
    )

    # Generate random input mask if the operator expects the mask as fourth
    # input
    if mask == "input":
        mask = gen_finn_dt_tensor(DataType["FLOAT32"], (QLen, KVLen))
    # If a causal attention mask is requested, generate upper triangular matrix
    elif mask == "causal":
        # Start zero initialized mask
        mask = 0 * gen_finn_dt_tensor(DataType["FLOAT32"], (QLen, KVLen))
        # Fill upper triangular causal attention mask
        mask[np.triu_indices_from(mask, 1)] = - np.inf
    # No mask input requested
    elif mask == "none":
        # No mask is equivalent to a zero mask
        mask = 0 * gen_finn_dt_tensor(DataType["FLOAT32"], (QLen, KVLen))

    # Prepare execution context
    context = {
        "Q": q, "K": k, "V": v, "mask": mask
    }
    # Set model execution mode to python (numpy execution)
    model = model.transform(SetExecMode("python"))
    # Generates the C++ source to be compiled as C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    # Prepares IP-generation
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))

    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["O"]

    # Compute the attention matrix between queries and keys
    attention = softmax(q @ k.T * (QKDim ** -0.5) + mask, axis=-1)
    # Compute product of attention weights and value input
    o_expected = attention @ v

    # Test whether the expectation and the onnx model output match
    assert (o_produced == o_expected).all(), "python exec failed"  # noqa


# This is a fpgadataflow type of test
@pytest.mark.fpgadataflow
# Tests cpp simulation of single scaled dot-product attention head
def test_fpgadataflow_attention_cppsim():
    pass


# This is a fpgadataflow type of test
@pytest.mark.fpgadataflow
# Tests rtl simulation of single scaled dot-product attention head
def test_fpgadataflow_attention_rtlsim():
    pass

# Testing framework
import pytest

# Use numpy for python execution / computing the ground truth expected values
import numpy as np
from qonnx.transformation.general import GiveUniqueNodeNames
# Numpy compatible implementation of the softmax operation
from scipy.special import softmax

# Generate random input data for testing
from qonnx.util.basic import gen_finn_dt_tensor, DataType
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx
# Attention operator to test
from finn.custom_op.fpgadataflow.attention import ScaledDotProductAttention
from qonnx.custom_op.registry import getCustomOp
# Graphs transformation setting the execution mode attribute
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

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
    model = ScaledDotProductAttention.make_modelwrapper_like(
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

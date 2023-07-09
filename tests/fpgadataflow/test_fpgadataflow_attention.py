# Testing framework
import pytest

# Use numpy for python execution / computing the ground truth expected values
import numpy as np
# Numpy compatible implementation of the softmax operation
from scipy.special import softmax

# Generate random input data for testing
from qonnx.util.basic import gen_finn_dt_tensor, DataType
# Execute onnx model graphs
from qonnx.core.onnx_exec import execute_onnx
# Attention operator to test
from finn.custom_op.fpgadataflow.attention import ScaledDotProductAttention
# Graphs transformation setting the execution mode attribute
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


# Size of query and key embedding dimension
@pytest.mark.parametrize("qk_dim", [64])
# Size of value embedding dimension
@pytest.mark.parametrize("v_dim", [64])
# Length of key and value sequences
@pytest.mark.parametrize("kv_len", [256])
# Length of query sequence
@pytest.mark.parametrize("q_len", [256])
# Different modes to provide a mask
@pytest.mark.parametrize("mask", ["none", "input", "causal"])
# Output parallelism
@pytest.mark.parametrize("pe", [1])
# Input parallelism
@pytest.mark.parametrize("simd", [1])
# Datatypes of queries, keys and values, mask and output
@pytest.mark.parametrize("q_dtype", [DataType["FLOAT32"]])
@pytest.mark.parametrize("k_dtype", [DataType["FLOAT32"]])
@pytest.mark.parametrize("v_dtype", [DataType["FLOAT32"]])
@pytest.mark.parametrize("mask_dtype", [DataType["FLOAT32"]])
@pytest.mark.parametrize("o_dtype", [DataType["FLOAT32"]])
# Tests python implementation of single scaled dot-product attention head
def test_attention_python(
        qk_dim, v_dim, kv_len, q_len, mask, pe, simd, q_dtype, k_dtype, v_dtype,
        mask_dtype, o_dtype
):
    # Generate random input data
    q = gen_finn_dt_tensor(q_dtype, (q_len, qk_dim))
    k = gen_finn_dt_tensor(k_dtype, (kv_len, qk_dim))
    v = gen_finn_dt_tensor(v_dtype, (kv_len, v_dim))

    dtypes = {
        # Datatypes of the query, key, value inputs and the output
        "q_dtype": q_dtype, "k_dtype": k_dtype,
        "v_dtype": v_dtype, "o_dtype": o_dtype,
    }

    # Generate the operator matching the configuration
    model = ScaledDotProductAttention.make_modelwrapper_like(
        q, k, v, mask, pe, simd, **dtypes, mask_dtype=mask_dtype
    )

    # Generate random input mask if the operator expects the mask as fourth
    # input
    if mask == "input":
        mask = gen_finn_dt_tensor(DataType["FLOAT32"], (q_len, kv_len))
    # If a causal attention mask is requested, generate upper triangular matrix
    elif mask == "causal":
        # Start zero initialized mask
        mask = 0 * gen_finn_dt_tensor(DataType["FLOAT32"], (q_len, kv_len))
        # Fill upper triangular causal attention mask
        mask[np.triu_indices_from(mask, 1)] = - np.inf
    # No mask input requested
    elif mask == "none":
        # No mask is equivalent to a zero mask
        mask = 0 * gen_finn_dt_tensor(DataType["FLOAT32"], (q_len, kv_len))

    # Prepare execution context
    context = {
        "q": q, "k": k, "v": v, "mask": mask
    }
    # Set model execution mode to python (numpy execution)
    model = model.transform(SetExecMode("python"))
    # Execute the onnx model to collect the result
    o_produced = execute_onnx(model, context)["o"]

    # Compute the attention matrix between queries and keys
    attention = softmax(q @ k.T * (qk_dim ** -0.5) + mask, axis=-1)
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

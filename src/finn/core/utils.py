import random
import string

import numpy as np
import onnx
from bitstring import BitArray

from finn.core.datatype import DataType


def valueinfo_to_tensor(vi):
    """Creates an all-zeroes numpy tensor from a ValueInfoProto."""

    dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
    return np.zeros(
        dims, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type]
    )


def get_by_name(container, name, name_field="name"):
    """Return item from container by .name field if it exists, None otherwise"""
    names = [getattr(x, name_field) for x in container]
    try:
        ind = names.index(name)
        return container[ind]
    except ValueError:
        return None


def remove_by_name(container, name, name_field="name"):
    """Remove item from container by .name field if it exists"""
    item = get_by_name(container, name, name_field)
    if item is not None:
        container.remove(item)


def random_string(stringLength=6):
    """Randomly generate a string of letters and digits."""
    lettersAndDigits = string.ascii_letters + string.digits
    return "".join(random.choice(lettersAndDigits) for i in range(stringLength))


def array2hexstring(array, dtype, pad_to_nbits):
    """
    Pack given one-dimensional NumPy array with FINN DataType dtype into a hex
    string.
    Any BIPOLAR values will be converted to a single bit with a 0 representing
    -1.
    pad_to_nbits is used to prepend leading zeros to ensure packed strings of
    fixed width. The minimum value for pad_to_nbits is 4, since a single hex
    digit is four bits.

    Examples:
    array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) = "e"
    array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) = "0e"
    """
    if pad_to_nbits < 4:
        pad_to_nbits = 4
    # ensure input is a numpy array with float values
    if type(array) != np.ndarray or array.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        array = np.asarray(array, dtype=np.float32)
    # ensure one-dimensional array to pack
    assert array.ndim == 1
    if dtype == DataType.BIPOLAR:
        # convert bipolar values to binary
        array = (array + 1) / 2
        dtype = DataType.BINARY
    lineval = BitArray(length=0)
    bw = dtype.bitwidth()
    for val in array:
        # ensure that this value is permitted by chosen dtype
        assert dtype.allowed(val)
        if dtype.is_integer():
            if dtype.signed():
                lineval.append(BitArray(int=int(val), length=bw))
            else:
                lineval.append(BitArray(uint=int(val), length=bw))
        else:
            lineval.append(BitArray(float=val, length=bw))
    if pad_to_nbits >= lineval.len:
        # extend to the desired output width (a minimum of 4 bits)
        lineval.prepend(BitArray(length=pad_to_nbits - lineval.len))
    else:
        raise Exception("Number of bits is greater than pad_to_nbits")
    # represent as hex
    return lineval.hex


def pack_innermost_dim_as_hex_string(ndarray, dtype, pad_to_nbits):
    """Pack the innermost dimension of the given numpy ndarray into hex
    strings using array2hexstring. Examples:

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = ["0e", "06"]
    pack_innermost_dim_as_hex_string(A, DataType.BINARY, 8) == eA
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = [[ "0f", "0f"], ["07", "0d"]]
    pack_innermost_dim_as_hex_string(B, DataType.UINT2, 8) == eB
    """

    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)

    def fun(x):
        return array2hexstring(x, dtype, pad_to_nbits)

    return np.apply_along_axis(fun, ndarray.ndim - 1, ndarray)


def interleave_matrix_outer_dim_from_partitions(matrix, n_partitions):
    if type(matrix) != np.ndarray or matrix.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    ndim = matrix.ndim
    # ensure # partitions evenly divide the outermost dimension
    assert shp[0] % n_partitions == 0
    # only tested for matrices
    assert ndim == 2
    # interleave rows between PEs using reshape + transpose
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r


def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x
    assert int(factor) == factor
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))


def pad_tensor_to_multiple_of(ndarray, pad_to_dims, val=0, distr_pad=False):
    """Pad each dimension of given NumPy ndarray using val, so that each
    dimension is a multiple of the respective value in pad_to_dims. -1 means
    do not pad that particular dimension. If distr_pad is False, all padding
    will be inserted after the existing values; otherwise it will be split
    evenly between before and after the existing values, with one extra value
    inserted after if the padding amount is not divisible by two."""
    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)
    assert ndarray.ndim == len(pad_to_dims)
    # compute the desired shape
    desired = zip(list(ndarray.shape), list(pad_to_dims))
    desired = map(lambda x: roundup_to_integer_multiple(x[0], x[1]), desired)
    desired = np.asarray(list(desired), dtype=np.int32)
    current = np.asarray(ndarray.shape, dtype=np.int32)
    pad_amt = desired - current
    # add padding to get to the desired shape
    if distr_pad:
        pad_before = (pad_amt // 2).astype(np.int32)
        pad_after = pad_amt - pad_before
        pad_amt = list(zip(pad_before, pad_after))
    else:
        # all padding is added after the existing values
        pad_amt = list(map(lambda x: (0, x), pad_amt))
    ret = np.pad(ndarray, pad_amt, mode="constant", constant_values=val)
    assert (np.asarray(ret.shape, dtype=np.int32) == desired).all()
    return ret


def gen_finn_dt_tensor(finn_dt, tensor_shape):
    # generates random tensor in given shape and with given FINN data type
    if finn_dt == DataType.BIPOLAR:
        tensor_values = np.random.randint(2, size=tensor_shape)
        tensor_values = 2 * tensor_values - 1
    elif finn_dt == DataType.BINARY:
        tensor_values = np.random.randint(2, size=tensor_shape)
    elif "INT" in finn_dt.name or finn_dt == DataType.TERNARY:
        tensor_values = np.random.randint(
            finn_dt.min(), high=finn_dt.max() + 1, size=tensor_shape
        )
    else:
        raise ValueError(
            "Datatype {} is not supported, no tensor could be generated".format(finn_dt)
        )
    # always use float type as container
    return tensor_values.astype(np.float32)

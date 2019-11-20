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
    pad_to_bits is used to prepend leading zeros to ensure packed strings of
    fixed width. The minimum value for pad_to_bits is 4, since a single hex
    digit is four bits.

    Examples:
    array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) = "e"
    array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) = "0e"
    """
    assert pad_to_nbits >= 4
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

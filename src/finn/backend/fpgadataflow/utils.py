import os
import sys

import numpy as np

from finn.core.datatype import DataType
from finn.core.utils import (
    pack_innermost_dim_as_hex_string,
    unpack_innermost_dim_from_hex_string,
)


def numpy_to_hls_code(
    ndarray, dtype, hls_var_name, pack_innermost_dim=True, no_decl=False
):
    """Return C++ code representation of a numpy ndarray with FINN DataType
    dtype, using hls_var_name as the resulting C++ variable name. If
    pack_innermost_dim is specified, the innermost dimension of the ndarray
    will be packed into a hex string using array2hexstring. If no_decl is
    set to True, no variable name and type will be generated as part of the
    emitted string.
    """
    hls_dtype = dtype.get_hls_datatype_str()
    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)
    if pack_innermost_dim:
        idimlen = ndarray.shape[-1]
        idimbits = idimlen * dtype.bitwidth()
        ndarray = pack_innermost_dim_as_hex_string(ndarray, dtype, idimbits)
        hls_dtype = "ap_uint<%d>" % idimbits
    ndims = ndarray.ndim
    # add type string and variable name
    # e.g. "const ap_uint<64>" "weightMem0"
    ret = "%s %s" % (hls_dtype, hls_var_name)
    # add dimensions
    for d in range(ndims):
        ret += "[%d]" % ndarray.shape[d]
    orig_printops = np.get_printoptions()
    np.set_printoptions(threshold=sys.maxsize)

    # define a function to convert a single element into a C++ init string
    # a single element can be a hex string if we are using packing
    def elem2str(x):
        if type(x) == str or type(x) == np.str_ or type(x) == np.str:
            return '%s("%s", 16)' % (hls_dtype, x)
        elif type(x) == np.float32:
            if dtype == DataType.FLOAT32:
                return str(x)
            else:
                return str(int(x))
        else:
            raise Exception("Unsupported type for numpy_to_hls_code")

    strarr = np.array2string(ndarray, separator=", ", formatter={"all": elem2str})
    np.set_printoptions(**orig_printops)
    strarr = strarr.replace("[", "{").replace("]", "}")
    if no_decl:
        ret = strarr + ";"
    else:
        ret = ret + " = \n" + strarr + ";"
    return ret


def npy_to_rtlsim_input(input_file, input_dtype, pad_to_nbits):
    """Convert the multidimensional NumPy array of integers (stored as floats)
    from input_file into a flattened sequence of Python arbitrary-precision
    integers, packing the innermost dimension. See
    finn.core.utils.pack_innermost_dim_as_hex_string() for more info on how the
    packing works."""

    inp = np.load(input_file)
    ishape = inp.shape
    inp = inp.flatten()
    inp_rev = []
    for i in range(len(inp)):
        inp_rev.append(inp[-1])
        inp = inp[:-1]
    inp_rev = np.asarray(inp_rev, dtype=np.float32).reshape(ishape)
    packed_data = pack_innermost_dim_as_hex_string(inp_rev, input_dtype, pad_to_nbits)
    packed_data = packed_data.flatten()
    packed_data = [int(x[2:], 16) for x in packed_data]
    packed_data.reverse()
    return packed_data


def rtlsim_output_to_npy(output, path, dtype, shape, packedBits, targetBits):
    """Convert a flattened sequence of Python arbitrary-precision integers
    output into a NumPy array. Each arbitrary-precision integer is assumed to
    be a packed array of targetBits-bit elements, which will be unpacked
    as the innermost dimension of the NumPy array."""

    output = [hex(int(x)) for x in output]
    out_array = unpack_innermost_dim_from_hex_string(
        output, dtype, shape, packedBits, targetBits, True
    )
    np.save(os.path.join(path, "output.npy"), out_array)

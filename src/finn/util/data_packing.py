# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import binascii
import numpy as np
import os
import sys
from bitstring import BitArray
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple


def array2hexstring(array, dtype, pad_to_nbits, prefix="0x", reverse=False):
    """
    Pack given one-dimensional NumPy array with FINN DataType dtype into a hex
    string.
    Any BIPOLAR values will be converted to a single bit with a 0 representing
    -1.
    pad_to_nbits is used to prepend leading zeros to ensure packed strings of
    fixed width. The minimum value for pad_to_nbits is 4, since a single hex
    digit is four bits. reverse can be used to reverse the array prior to
    packing.

    Examples:

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 4) = "0xe"

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 8) = "0x0e"

    array2hexstring([1, 1, 0, 1], DataType["BINARY"], 4, reverse=True) = "0xb"

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 8, reverse=True) = "0x07"
    """
    if pad_to_nbits < 4:
        pad_to_nbits = 4
    # ensure input is a numpy array with float values
    if type(array) != np.ndarray or array.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        array = np.asarray(array, dtype=np.float32)
    # ensure one-dimensional array to pack
    assert array.ndim == 1, "The given array is not one-dimensional."
    if dtype == DataType["BIPOLAR"]:
        # convert bipolar values to binary
        array = (array + 1) / 2
        dtype = DataType["BINARY"]
    # reverse prior to packing, if desired
    if reverse:
        array = np.flip(array, -1)
    lineval = BitArray(length=0)
    bw = dtype.bitwidth()
    # special handling for fixed point: rescale, then pack as integers
    if dtype.is_fixed_point():
        sf = dtype.scale_factor()
        array = array / sf
        # replace dtype with signed integer equivalent
        dtype = DataType["INT" + str(bw)]
    for val in array:
        # ensure that this value is permitted by chosen dtype
        assert dtype.allowed(val), "This value is not permitted by chosen dtype."
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
    return prefix + lineval.hex


def hexstring2npbytearray(hexstring, remove_prefix="0x"):
    """Convert a hex string into a NumPy array of dtype uint8.

    Example:

    hexstring2npbytearray("0f01") = array([15,  1], dtype=uint8)
    """
    # remove prefix if found
    if hexstring.startswith(remove_prefix):
        lrp = len(remove_prefix)
        hexstring = hexstring[lrp:]
    # use Python's built-in bytearray
    return np.asarray(bytearray.fromhex(hexstring), dtype=np.uint8)


def npbytearray2hexstring(npbytearray, prefix="0x"):
    """Convert a NumPy array of uint8 dtype into a hex string.

    Example:

    npbytearray2hexstring(array([15,  1], dtype=uint8)) = "0x0f01"
    """
    return prefix + binascii.hexlify(bytearray(npbytearray)).decode("utf-8")


def pack_innermost_dim_as_hex_string(
    ndarray, dtype, pad_to_nbits, reverse_inner=False, prefix="0x"
):
    """Pack the innermost dimension of the given numpy ndarray into hex
    strings using array2hexstring.

    Examples:

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]

    eA = ["0e", "06"]

    pack_innermost_dim_as_hex_string(A, DataType["BINARY"], 8) == eA

    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]

    eB = [[ "0f", "0f"], ["07", "0d"]]

    pack_innermost_dim_as_hex_string(B, DataType["UINT2"], 8) == eB
    """

    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)

    def fun(x):
        return array2hexstring(x, dtype, pad_to_nbits, reverse=reverse_inner, prefix=prefix)

    return np.apply_along_axis(fun, ndarray.ndim - 1, ndarray)


def unpack_innermost_dim_from_hex_string(
    ndarray, dtype, out_shape, packedBits, reverse_inner=False
):
    """Convert a NumPy array of hex strings into a FINN NumPy array by unpacking
    the hex strings into the specified data type. out_shape can be specified
    such that any padding in the packing dimension is removed. If reverse_inner
    is set, the innermost unpacked dimension will be reversed."""

    if type(ndarray) != np.ndarray:
        raise Exception(
            """unpack_innermost_dim_from_hex_string needs ndarray
        as input"""
        )
    if ndarray.dtype.kind not in {"U", "S"}:
        raise Exception(
            """unpack_innermost_dim_from_hex_string needs ndarray of
        hex strings as input"""
        )
    # convert ndarray into flattened list
    data = ndarray.flatten().tolist()
    targetBits = dtype.bitwidth()
    # calculate outer and inner dim shapes
    outer_dim_elems = 1
    for dim in range(len(out_shape) - 1):
        outer_dim_elems = outer_dim_elems * out_shape[dim]
    inner_dim_elems = out_shape[-1]

    array = []
    if dtype.is_fixed_point():
        # convert fixed point as signed integer
        conv_dtype = DataType["INT" + str(targetBits)]
    else:
        conv_dtype = dtype
    for outer_elem in range(outer_dim_elems):
        ar_list = []
        ar_elem = data[0]
        data.pop(0)
        ar_elem = ar_elem.split("x")
        ar_elem_bin = bin(int(ar_elem[1], 16))[2:].zfill(packedBits)
        ar_elem_bin = [int(x) for x in ar_elem_bin]

        ar_elem_bin.reverse()
        for i in range(inner_dim_elems):
            upper_limit = (i + 1) * targetBits
            lower_limit = i * targetBits
            elem = ar_elem_bin[lower_limit:upper_limit]
            elem.reverse()
            elem_str = "".join(map(str, elem))
            if conv_dtype == DataType["FLOAT32"]:
                ar_list.append(BitArray(bin=elem_str).float)
            elif conv_dtype.is_integer():
                ar_list.append(int(elem_str, 2))
            else:
                raise Exception("Not implemented for conv_dtype " + conv_dtype.name)
        # reverse inner dimension back to "normal" positions
        if reverse_inner is False:
            ar_list.reverse()

        # interpret output values correctly

        # interpret values as bipolar
        if conv_dtype == DataType["BIPOLAR"]:
            ar_list = [2 * x - 1 for x in ar_list]
        # interpret values as signed values
        elif conv_dtype.signed() and conv_dtype.is_integer():
            mask = 2 ** (conv_dtype.bitwidth() - 1)
            ar_list = [-(x & mask) + (x & ~mask) for x in ar_list]

        array.append(ar_list)
    array = np.asarray(array, dtype=np.float32).reshape(out_shape)
    if dtype.is_fixed_point():
        # convert signed integer to fixed point by applying scale
        array = array * dtype.scale_factor()
    return array


def numpy_to_hls_code(ndarray, dtype, hls_var_name, pack_innermost_dim=True, no_decl=False):
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
        idimbits = roundup_to_integer_multiple(idimbits, 4)
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
        if type(x) == str or type(x) == np.str_:
            return '%s("%s", 16)' % (hls_dtype, x)
        elif type(x) == np.float32:
            if dtype.is_integer():
                return str(int(x))
            else:
                return str(x)
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


def npy_to_rtlsim_input(input_file, input_dtype, pad_to_nbits, reverse_inner=True):
    """Convert the multidimensional NumPy array of integers (stored as floats)
    from input_file into a flattened sequence of Python arbitrary-precision
    integers, packing the innermost dimension. See
    finn.util.basic.pack_innermost_dim_as_hex_string() for more info on how the
    packing works. If reverse_inner is set, the innermost dimension will be
    reversed prior to packing."""
    pad_to_nbits = roundup_to_integer_multiple(pad_to_nbits, 4)
    if issubclass(type(input_file), np.ndarray):
        inp = input_file
    elif os.path.isfile(input_file):
        inp = np.load(input_file)
    else:
        raise Exception("input_file must be ndarray or filename for .npy")
    if inp.shape[-1] == 1 and input_dtype.is_integer():
        packed_data = inp.flatten().astype(input_dtype.to_numpy_dt())
        packed_data = [int(x) for x in packed_data]
    else:
        packed_data = pack_innermost_dim_as_hex_string(
            inp, input_dtype, pad_to_nbits, reverse_inner=reverse_inner
        )
        packed_data = packed_data.flatten()
        packed_data = [int(x[2:], 16) for x in packed_data]
    return packed_data


def rtlsim_output_to_npy(output, path, dtype, shape, packedBits, targetBits, reverse_inner=True):
    """Convert a flattened sequence of Python arbitrary-precision integers
    output into a NumPy array, saved as npy file at path. Each arbitrary-precision
    integer is assumed to be a packed array of targetBits-bit elements, which
    will be unpacked as the innermost dimension of the NumPy array. If path is
    not None it will also be saved as a npy file."""

    # TODO should have its own testbench?
    output = np.asarray([hex(int(x)) for x in output])
    out_array = unpack_innermost_dim_from_hex_string(
        output, dtype, shape, packedBits=packedBits, reverse_inner=reverse_inner
    )
    # make copy before saving the array
    out_array = out_array.copy()
    if path is not None:
        np.save(path, out_array)
    return out_array


def finnpy_to_packed_bytearray(
    ndarray, dtype, reverse_inner=False, reverse_endian=False, fast_mode=False
):
    """Given a numpy ndarray with FINN DataType dtype, pack the innermost
    dimension and return the packed representation as an ndarray of uint8.
    The packed innermost dimension will be padded to the nearest multiple
    of 8 bits. The returned ndarray has the same number of dimensions as the
    input.

    If fast_mode is enabled, will attempt to use shortcuts  to save
    on runtime for certain cases:
    * 8-bit ndarray -> 8-bit
    * ndarray -> 1-bit and total bits % 8 == 0
    This mode is currently not well-tested, use at your own risk!
    """

    # handle fast_mode cases (currently only called from driver):
    if issubclass(type(ndarray), np.ndarray) and fast_mode:
        inp_is_byte = ndarray.dtype in [np.uint8, np.int8]
        out_is_byte = dtype.bitwidth() == 8
        double_reverse = reverse_inner and reverse_endian
        # fast mode case: byte -> byte: cast
        if inp_is_byte and out_is_byte and double_reverse:
            return ndarray.view(np.uint8)
        # fast mode case: xxx -> bit with nbits % 8 == 0: np.packbits
        out_is_bit = dtype.bitwidth() == 1
        bits = dtype.bitwidth() * ndarray.shape[-1]
        bits_padded = roundup_to_integer_multiple(bits, 8)
        no_pad = bits_padded == bits
        if out_is_bit and no_pad and double_reverse:
            in_as_int8 = ndarray.astype(np.int8)
            # bipolar -> binary if needed
            if dtype == DataType["BIPOLAR"]:
                in_as_int8 = (in_as_int8 + 1) // 2
            # reverse inner
            in_as_int8 = np.flip(in_as_int8, axis=-1)
            # pack with numpy
            packed_data = np.packbits(in_as_int8, axis=-1)
            # reverse endianness and return
            return np.flip(packed_data, axis=-1)

    if (not issubclass(type(ndarray), np.ndarray)) or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)
    # pack innermost dim to hex strings padded to 8 bits
    bits = dtype.bitwidth() * ndarray.shape[-1]
    bits_padded = roundup_to_integer_multiple(bits, 8)
    packed_hexstring = pack_innermost_dim_as_hex_string(
        ndarray, dtype, bits_padded, reverse_inner=reverse_inner
    )

    def fn(x):
        return np.asarray(list(map(hexstring2npbytearray, x)))

    if packed_hexstring.ndim == 0:
        # scalar, call hexstring2npbytearray directly
        ret = hexstring2npbytearray(np.asscalar(packed_hexstring))
    else:
        # convert ndarray of hex strings to byte array
        ret = np.apply_along_axis(fn, packed_hexstring.ndim - 1, packed_hexstring)
    if reverse_endian:
        # reverse the endianness of packing dimension
        ret = np.flip(ret, axis=-1)
    return ret


def packed_bytearray_to_finnpy(
    packed_bytearray,
    dtype,
    output_shape=None,
    reverse_inner=False,
    reverse_endian=False,
    fast_mode=False,
):
    """Given a packed numpy uint8 ndarray, unpack it into a FINN array of
    given DataType.

    output_shape can be specified to remove padding from the
    packed dimension, or set to None to be inferred from the input.

    If fast_mode is enabled, will attempt to use shortcuts (casting) to save
    on runtime for certain cases.
    This mode is currently not well-tested, use at your own risk.

    """

    if (not issubclass(type(packed_bytearray), np.ndarray)) or packed_bytearray.dtype != np.uint8:
        raise Exception("packed_bytearray_to_finnpy needs NumPy uint8 arrays")
    if packed_bytearray.ndim == 0:
        raise Exception("packed_bytearray_to_finnpy expects at least 1D ndarray")
    packed_dim = packed_bytearray.ndim - 1
    packed_bits = packed_bytearray.shape[packed_dim] * 8
    target_bits = dtype.bitwidth()
    if output_shape is None:
        # determine output shape from input shape
        assert (
            packed_bits % target_bits == 0
        ), """packed_bits are not divisable by
        target_bits."""
        n_target_elems = packed_bits // target_bits
        output_shape = packed_bytearray.shape[:-1] + (n_target_elems,)
    # handle no-packing cases (if fast_mode) via casting to save on compute
    out_is_byte = target_bits in [8, 16]
    double_reverse = reverse_inner and reverse_endian
    if out_is_byte and double_reverse and fast_mode:
        no_unpad = np.prod(packed_bytearray.shape) == np.prod(output_shape)
        if no_unpad:
            as_np_type = packed_bytearray.view(dtype.to_numpy_dt())
            return as_np_type.reshape(output_shape).astype(np.float32)
    if reverse_endian:
        packed_bytearray = np.flip(packed_bytearray, axis=-1)
    # convert innermost dim of byte array to hex strings
    packed_hexstring = np.apply_along_axis(npbytearray2hexstring, packed_dim, packed_bytearray)
    ret = unpack_innermost_dim_from_hex_string(
        packed_hexstring, dtype, output_shape, packed_bits, reverse_inner
    )

    return ret

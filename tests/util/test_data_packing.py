# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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

import os
import shutil
import subprocess

import pytest

import numpy as np

import finn.util.basic as cutil
from finn.core.datatype import DataType
from finn.util.data_packing import (
    array2hexstring,
    finnpy_to_packed_bytearray,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    packed_bytearray_to_finnpy,
)


@pytest.mark.parametrize("dtype", [DataType.BINARY, DataType.INT2, DataType.INT32])
@pytest.mark.parametrize("test_shape", [(1, 2, 4), (1, 1, 64), (2, 64)])
@pytest.mark.vivado
def test_npy2apintstream(test_shape, dtype):
    ndarray = cutil.gen_finn_dt_tensor(dtype, test_shape)
    test_dir = cutil.make_build_dir(prefix="test_npy2apintstream_")
    shape = ndarray.shape
    elem_bits = dtype.bitwidth()
    packed_bits = shape[-1] * elem_bits
    packed_hls_type = "ap_uint<%d>" % packed_bits
    elem_hls_type = dtype.get_hls_datatype_str()
    npy_in = test_dir + "/in.npy"
    npy_out = test_dir + "/out.npy"
    # restrict the np datatypes we can handle
    npyt_to_ct = {
        "float32": "float",
        "float64": "double",
        "int8": "int8_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint8": "uint8_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
    }
    npy_type = npyt_to_ct[str(ndarray.dtype)]
    shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")
    test_app_string = []
    test_app_string += ["#include <cstddef>"]
    test_app_string += ["#define AP_INT_MAX_W 4096"]
    test_app_string += ['#include "ap_int.h"']
    test_app_string += ['#include "stdint.h"']
    test_app_string += ['#include "hls_stream.h"']
    test_app_string += ['#include "cnpy.h"']
    test_app_string += ['#include "npy2apintstream.hpp"']
    test_app_string += ["int main(int argc, char *argv[]) {"]
    test_app_string += ["hls::stream<%s> teststream;" % packed_hls_type]
    test_app_string += [
        'npy2apintstream<%s, %s, %d, %s>("%s", teststream);'
        % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
    ]
    test_app_string += [
        'apintstream2npy<%s, %s, %d, %s>(teststream, %s, "%s");'
        % (packed_hls_type, elem_hls_type, elem_bits, npy_type, shape_cpp_str, npy_out)
    ]
    test_app_string += ["return 0;"]
    test_app_string += ["}"]
    with open(test_dir + "/test.cpp", "w") as f:
        f.write("\n".join(test_app_string))
    cmd_compile = """
g++ -o test_npy2apintstream test.cpp /workspace/cnpy/cnpy.cpp \
-I/workspace/cnpy/ -I{}/include -I/workspace/finn/src/finn/qnn-data/cpp \
--std=c++11 -lz""".format(
        os.environ["VIVADO_PATH"]
    )
    with open(test_dir + "/compile.sh", "w") as f:
        f.write(cmd_compile)
    compile = subprocess.Popen(
        ["sh", "compile.sh"], stdout=subprocess.PIPE, cwd=test_dir
    )
    (stdout, stderr) = compile.communicate()
    # make copy before saving the array
    ndarray = ndarray.copy()
    np.save(npy_in, ndarray)
    execute = subprocess.Popen(
        "./test_npy2apintstream", stdout=subprocess.PIPE, cwd=test_dir
    )
    (stdout, stderr) = execute.communicate()
    produced = np.load(npy_out)
    success = (produced == ndarray).all()
    # only delete generated code if test has passed
    # useful for debug otherwise
    if success:
        shutil.rmtree(test_dir)
    assert success


def test_array2hexstring():
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) == "0xe"
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) == "0x0e"
    assert array2hexstring([1, 1, 1, -1], DataType.BIPOLAR, 8) == "0x0e"
    assert array2hexstring([3, 3, 3, 3], DataType.UINT2, 8) == "0xff"
    assert array2hexstring([1, 3, 3, 1], DataType.UINT2, 8) == "0x7d"
    assert array2hexstring([1, -1, 1, -1], DataType.INT2, 8) == "0x77"
    assert array2hexstring([1, 1, 1, -1], DataType.INT4, 16) == "0x111f"
    assert array2hexstring([-1], DataType.FLOAT32, 32) == "0xbf800000"
    assert array2hexstring([17.125], DataType.FLOAT32, 32) == "0x41890000"
    assert array2hexstring([1, 1, 0, 1], DataType.BINARY, 4, reverse=True) == "0xb"
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 8, reverse=True) == "0x07"


def test_pack_innermost_dim_as_hex_string():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(["0x0e", "0x06"])
    assert (pack_innermost_dim_as_hex_string(A, DataType.BINARY, 8) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    assert (pack_innermost_dim_as_hex_string(B, DataType.UINT2, 8) == eB).all()
    C = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eC = np.asarray([["0x0f", "0x0f"], ["0x0d", "0x07"]])
    assert (
        pack_innermost_dim_as_hex_string(C, DataType.UINT2, 8, reverse_inner=True) == eC
    ).all()


def test_numpy_to_hls_code():
    def remove_all_whitespace(s):
        return "".join(s.split())

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    ret = numpy_to_hls_code(A, DataType.BINARY, "test", True)
    eA = """ap_uint<4> test[2] =
    {ap_uint<4>("0xe", 16), ap_uint<4>("0x6", 16)};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eA)
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    ret = numpy_to_hls_code(B, DataType.UINT2, "test", True)
    eB = """ap_uint<4> test[2][2] =
    {{ap_uint<4>("0xf", 16), ap_uint<4>("0xf", 16)},
     {ap_uint<4>("0x7", 16), ap_uint<4>("0xd", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)
    ret = numpy_to_hls_code(B, DataType.UINT2, "test", True, True)
    eB = """{{ap_uint<4>("0xf", 16), ap_uint<4>("0xf", 16)},
     {ap_uint<4>("0x7", 16), ap_uint<4>("0xd", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)


def test_finnpy_to_packed_bytearray():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray([[14], [6]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(A, DataType.BINARY) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([[[15], [15]], [[7], [13]]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(B, DataType.UINT2) == eB).all()
    C = [1, 7, 2, 5]
    eC = np.asarray([23, 37], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(C, DataType.UINT4) == eC).all()
    D = [[1, 7, 2, 5], [2, 5, 1, 7]]
    eD = np.asarray([[23, 37], [37, 23]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(D, DataType.UINT4) == eD).all()
    E = [[-4, 0, -4, -4]]
    eE = np.asarray(
        [[255, 255, 255, 252, 0, 0, 0, 0, 255, 255, 255, 252, 255, 255, 255, 252]],
        dtype=np.uint8,
    )
    assert (finnpy_to_packed_bytearray(E, DataType.INT32) == eE).all()


def test_packed_bytearray_to_finnpy():
    A = np.asarray([[14], [6]], dtype=np.uint8)
    eA = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(eA, dtype=np.float32)
    shapeA = eA.shape
    assert (packed_bytearray_to_finnpy(A, DataType.BINARY, shapeA) == eA).all()
    B = np.asarray([[[15], [15]], [[7], [13]]], dtype=np.uint8)
    eB = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray(eB, dtype=np.float32)
    shapeB = eB.shape
    assert (packed_bytearray_to_finnpy(B, DataType.UINT2, shapeB) == eB).all()
    C = np.asarray([23, 37], dtype=np.uint8)
    eC = [1, 7, 2, 5]
    eC = np.asarray(eC, dtype=np.float32)
    shapeC = eC.shape
    assert (packed_bytearray_to_finnpy(C, DataType.UINT4, shapeC) == eC).all()
    D = np.asarray([[23, 37], [37, 23]], dtype=np.uint8)
    eD = [[1, 7, 2, 5], [2, 5, 1, 7]]
    eD = np.asarray(eD, dtype=np.float32)
    shapeD = eD.shape
    assert (packed_bytearray_to_finnpy(D, DataType.UINT4, shapeD) == eD).all()
    E = np.asarray(
        [[255, 255, 255, 252, 0, 0, 0, 0, 255, 255, 255, 252, 255, 255, 255, 252]],
        dtype=np.uint8,
    )
    eE = [[-4, 0, -4, -4]]
    eE = np.asarray(eE, dtype=np.float32)
    shapeE = eE.shape
    assert (packed_bytearray_to_finnpy(E, DataType.INT32, shapeE) == eE).all()
    F = np.asarray(
        [[252, 255, 255, 255, 0, 0, 0, 0, 252, 255, 255, 255, 252, 255, 255, 255]],
        dtype=np.uint8,
    )
    eF = [[-4, 0, -4, -4]]
    eF = np.asarray(eE, dtype=np.float32)
    shapeF = eF.shape
    assert (
        packed_bytearray_to_finnpy(
            F, DataType.INT32, shapeF, reverse_inner=True, reverse_endian=True
        )
        == eF
    ).all()

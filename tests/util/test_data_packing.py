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

import pytest

import numpy as np
import os
import shutil
import subprocess
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, numpy_to_hls_code


@pytest.mark.util
@pytest.mark.parametrize(
    "dtype",
    [
        DataType["BINARY"],
        DataType["INT2"],
        DataType["INT32"],
        DataType["FIXED<9,6>"],
        DataType["FLOAT32"],
    ],
)
@pytest.mark.parametrize("test_shape", [(1, 2, 4), (1, 1, 64), (2, 64)])
@pytest.mark.vivado
def test_npy2apintstream(test_shape, dtype):
    ndarray = gen_finn_dt_tensor(dtype, test_shape)
    test_dir = make_build_dir(prefix="test_npy2apintstream_")
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
g++ -o test_npy2apintstream test.cpp $FINN_ROOT/deps/cnpy/cnpy.cpp \
-I$FINN_ROOT/deps/cnpy/ -I{}/include -I{}/include -I$FINN_ROOT/src/finn/qnn-data/cpp \
--std=c++11 -lz""".format(
        os.environ["HLS_PATH"], os.environ["VITIS_PATH"]
    )
    with open(test_dir + "/compile.sh", "w") as f:
        f.write(cmd_compile)
    compile = subprocess.Popen(["sh", "compile.sh"], stdout=subprocess.PIPE, cwd=test_dir)
    (stdout, stderr) = compile.communicate()
    # make copy before saving the array
    ndarray = ndarray.copy()
    np.save(npy_in, ndarray)
    execute = subprocess.Popen("./test_npy2apintstream", stdout=subprocess.PIPE, cwd=test_dir)
    (stdout, stderr) = execute.communicate()
    produced = np.load(npy_out)
    success = (produced == ndarray).all()
    # only delete generated code if test has passed
    # useful for debug otherwise
    if success:
        shutil.rmtree(test_dir)
    assert success


@pytest.mark.util
def test_numpy_to_hls_code():
    def remove_all_whitespace(s):
        return "".join(s.split())

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    ret = numpy_to_hls_code(A, DataType["BINARY"], "test", True)
    eA = """ap_uint<4> test[2] =
    {ap_uint<4>("0xe", 16), ap_uint<4>("0x6", 16)};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eA)
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    ret = numpy_to_hls_code(B, DataType["UINT2"], "test", True)
    eB = """ap_uint<4> test[2][2] =
    {{ap_uint<4>("0xf", 16), ap_uint<4>("0xf", 16)},
     {ap_uint<4>("0x7", 16), ap_uint<4>("0xd", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)
    ret = numpy_to_hls_code(B, DataType["UINT2"], "test", True, True)
    eB = """{{ap_uint<4>("0xf", 16), ap_uint<4>("0xf", 16)},
     {ap_uint<4>("0x7", 16), ap_uint<4>("0xd", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)


@pytest.mark.util
@pytest.mark.parametrize(
    "dtype",
    [
        DataType["BINARY"],
        DataType["BIPOLAR"],
        DataType["TERNARY"],
        DataType["INT2"],
        DataType["INT7"],
        DataType["INT8"],
        DataType["INT22"],
        DataType["INT32"],
        DataType["UINT7"],
        DataType["UINT8"],
        DataType["UINT15"],
        DataType["FIXED<9,6>"],
        DataType["FLOAT32"],
    ],
)
def test_npy_to_rtlsim_input(dtype):
    # check if slow and fast data packing produce the same non-sign-extended input for rtlsim
    # fast mode is triggered for certain data types if last (SIMD) dim = 1
    inp_fast = gen_finn_dt_tensor(dtype, (1, 8, 8, 8 // 1, 1))  # N H W FOLD SIMD
    inp_slow = inp_fast.reshape((1, 8, 8, 8 // 2, 2))  # N H W FOLD SIMD

    output_fast = npy_to_rtlsim_input(inp_fast, dtype, 1 * dtype.bitwidth())
    output_slow = npy_to_rtlsim_input(inp_slow, dtype, 2 * dtype.bitwidth())

    output_slow_split = []
    for x in output_slow:
        # least significant bits = first element:
        output_slow_split.append(x & ((1 << dtype.bitwidth()) - 1))
        # remaining bits = second element:
        output_slow_split.append(x >> dtype.bitwidth())

    assert all([(x >> dtype.bitwidth()) == 0 for x in output_fast]), "extraneous bits detected"
    assert np.all(output_fast == output_slow_split), "different behavior of packing modes detected"

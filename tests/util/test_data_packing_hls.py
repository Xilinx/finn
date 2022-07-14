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

import finn.util.basic as cutil
from finn.core.datatype import DataType
from finn.util.data_packing import numpy_to_hls_code


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

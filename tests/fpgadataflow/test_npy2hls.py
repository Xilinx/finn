import shutil
import subprocess
import tempfile as tmp

import numpy as np

import finn.core.utils as cutil
from finn.backend.fpgadataflow.utils import numpy_to_hls_code
from finn.core.datatype import DataType


def make_npy2apintstream_testcase(ndarray, dtype):
    test_dir = tmp.mkdtemp(prefix="test_npy2apintstream_")
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
-I/workspace/cnpy/ -I/workspace/vivado-hlslib -I/workspace/finn/src/finn/data/cpp \
--std=c++11 -lz"""
    with open(test_dir + "/compile.sh", "w") as f:
        f.write(cmd_compile)
    compile = subprocess.Popen(
        ["sh", "compile.sh"], stdout=subprocess.PIPE, cwd=test_dir
    )
    (stdout, stderr) = compile.communicate()
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


test_shapes = [(1, 2, 4), (1, 1, 64), (2, 64)]


def test_npy2apintstream_binary():
    for test_shape in test_shapes:
        dt = DataType.BINARY
        W = cutil.gen_finn_dt_tensor(dt, test_shape)
        make_npy2apintstream_testcase(W, dt)


def test_npy2apintstream_int2():
    for test_shape in test_shapes:
        dt = DataType.INT2
        W = cutil.gen_finn_dt_tensor(dt, test_shape)
        make_npy2apintstream_testcase(W, dt)


def test_array2hexstring():
    assert cutil.array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) == "e"
    assert cutil.array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) == "0e"
    assert cutil.array2hexstring([1, 1, 1, -1], DataType.BIPOLAR, 8) == "0e"
    assert cutil.array2hexstring([3, 3, 3, 3], DataType.UINT2, 8) == "ff"
    assert cutil.array2hexstring([1, 3, 3, 1], DataType.UINT2, 8) == "7d"
    assert cutil.array2hexstring([1, -1, 1, -1], DataType.INT2, 8) == "77"
    assert cutil.array2hexstring([1, 1, 1, -1], DataType.INT4, 16) == "111f"
    assert cutil.array2hexstring([-1], DataType.FLOAT32, 32) == "bf800000"
    assert cutil.array2hexstring([17.125], DataType.FLOAT32, 32) == "41890000"


def test_pack_innermost_dim_as_hex_string():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(["0e", "06"])
    assert (cutil.pack_innermost_dim_as_hex_string(A, DataType.BINARY, 8) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([["0f", "0f"], ["07", "0d"]])
    assert (cutil.pack_innermost_dim_as_hex_string(B, DataType.UINT2, 8) == eB).all()


def test_numpy_to_hls_code():
    def remove_all_whitespace(s):
        return "".join(s.split())

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    ret = numpy_to_hls_code(A, DataType.BINARY, "test", True)
    eA = """ap_uint<4> test[2] =
    {ap_uint<4>("e", 16), ap_uint<4>("6", 16)};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eA)
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    ret = numpy_to_hls_code(B, DataType.UINT2, "test", True)
    eB = """ap_uint<4> test[2][2] =
    {{ap_uint<4>("f", 16), ap_uint<4>("f", 16)},
     {ap_uint<4>("7", 16), ap_uint<4>("d", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)
    ret = numpy_to_hls_code(B, DataType.UINT2, "test", True, True)
    eB = """{{ap_uint<4>("f", 16), ap_uint<4>("f", 16)},
     {ap_uint<4>("7", 16), ap_uint<4>("d", 16)}};"""
    assert remove_all_whitespace(ret) == remove_all_whitespace(eB)

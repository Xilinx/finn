import numpy as np

from finn.core.datatype import DataType
from finn.core.utils import array2hexstring, pack_innermost_dim_as_hex_string


def test_array2hexstring():
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) == "e"
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) == "0e"
    assert array2hexstring([1, 1, 1, -1], DataType.BIPOLAR, 8) == "0e"
    assert array2hexstring([3, 3, 3, 3], DataType.UINT2, 8) == "ff"
    assert array2hexstring([1, 3, 3, 1], DataType.UINT2, 8) == "7d"
    assert array2hexstring([1, -1, 1, -1], DataType.INT2, 8) == "77"
    assert array2hexstring([1, 1, 1, -1], DataType.INT4, 16) == "111f"
    assert array2hexstring([-1], DataType.FLOAT32, 32) == "bf800000"
    assert array2hexstring([17.125], DataType.FLOAT32, 32) == "41890000"


def test_pack_innermost_dim_as_hex_string():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(["0e", "06"])
    assert (pack_innermost_dim_as_hex_string(A, DataType.BINARY, 8) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([["0f", "0f"], ["07", "0d"]])
    assert (pack_innermost_dim_as_hex_string(B, DataType.UINT2, 8) == eB).all()

import numpy as np

from finn.core.datatype import DataType
from finn.util.data_packing import unpack_innermost_dim_from_hex_string


def test_unpack_innermost_dim_from_hex_string():
    A = np.asarray(["0x0e", "0x06"])
    dtype = DataType.BINARY
    shape = (1, 2, 4)
    eA = [[1, 1, 1, 0], [0, 1, 1, 0]]
    A_unpacked = unpack_innermost_dim_from_hex_string(A, dtype, shape)
    assert (A_unpacked == eA).all()

    A = np.asarray(["0x0e", "0x06"])
    eA_flipped = [[0, 1, 1, 1], [0, 1, 1, 0]]
    A_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        A, dtype, shape, reverse_inner=True
    )
    assert (A_unpacked_flipped == eA_flipped).all()

    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    dtype = DataType.UINT2
    shape = (1, 2, 2, 2)
    eB = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    B_unpacked = unpack_innermost_dim_from_hex_string(B, dtype, shape)
    assert (B_unpacked == eB).all()

    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    eB_flipped = [[[3, 3], [3, 3]], [[3, 1], [1, 3]]]
    B_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        B, dtype, shape, reverse_inner=True
    )
    assert (B_unpacked_flipped == eB_flipped).all()

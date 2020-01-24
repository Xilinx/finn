import numpy as np

from finn.core.utils import unpack_innermost_dim_from_hex_string


def test_unpack_innermost_dim_from_hex_string():
    A = np.asarray(["0x0e", "0x06"])
    A = A.flatten()
    A = list(A)
    shape = (1, 2, 4)
    packedBits = 8
    targetBits = 1
    eA = [[1, 1, 1, 0], [0, 1, 1, 0]]
    A_unpacked = unpack_innermost_dim_from_hex_string(A, shape, packedBits, targetBits)
    assert (A_unpacked == eA).all()

    A = np.asarray(["0x0e", "0x06"])
    A = A.flatten()
    A = list(A)
    eA_flipped = [[0, 1, 1, 1], [0, 1, 1, 0]]
    A_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        A, shape, packedBits, targetBits, True
    )
    assert (A_unpacked_flipped == eA_flipped).all()

    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    B = B.flatten()
    B = list(B)
    shape = (1, 2, 2, 2)
    packedBits = 8
    targetBits = 2
    eB = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    B_unpacked = unpack_innermost_dim_from_hex_string(B, shape, packedBits, targetBits)
    assert (B_unpacked == eB).all()

    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    B = B.flatten()
    B = list(B)
    eB_flipped = [[[3, 3], [3, 3]], [[3, 1], [1, 3]]]
    B_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        B, shape, packedBits, targetBits, True
    )
    assert (B_unpacked_flipped == eB_flipped).all()

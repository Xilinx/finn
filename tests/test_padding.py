import numpy as np

from finn.core.utils import pad_tensor_to_multiple_of


def test_pad_tensor_to_multiple_of():
    A = np.eye(3)
    B = pad_tensor_to_multiple_of(A, [2, 2], val=-1)
    assert B.shape == (4, 4)
    assert (B[:3, :3] == A).all()
    assert (B[3, :] == -1).all()
    assert (B[:, 3] == -1).all()
    B = pad_tensor_to_multiple_of(A, [5, 5], val=-1, distr_pad=True)
    assert B.shape == (5, 5)
    assert (B[1:4, 1:4] == A).all()
    assert (B[0, :] == -1).all()
    assert (B[:, 0] == -1).all()
    assert (B[4, :] == -1).all()
    assert (B[:, 4] == -1).all()
    # using -1 in pad_to parameter should give an unpadded dimension
    B = pad_tensor_to_multiple_of(A, [-1, 5], val=-1, distr_pad=True)
    assert B.shape == (3, 5)
    assert (B[:, 1:4] == A).all()
    assert (B[:, 0] == -1).all()
    assert (B[:, 4] == -1).all()
    # if odd number of padding pixels required, 1 more should go after existing
    B = pad_tensor_to_multiple_of(A, [6, 6], val=-1, distr_pad=True)
    assert B.shape == (6, 6)
    assert (B[1:4, 1:4] == A).all()

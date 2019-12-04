import numpy as np

import finn.core.utils as util


def test_interleave_matrix_outer_dim_from_partitions():
    A = np.eye(10)
    n_parts = 2
    Ax = util.interleave_matrix_outer_dim_from_partitions(A, n_parts)
    part_size = 10 // n_parts
    assert Ax.shape == (n_parts, part_size, 10)
    for r_ind in range(A.shape[0]):
        assert (A[r_ind] == Ax[r_ind % n_parts][r_ind // n_parts]).all()

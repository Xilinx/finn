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

import numpy as np

from finn.core.datatype import DataType
from finn.util.data_packing import unpack_innermost_dim_from_hex_string


def test_unpack_innermost_dim_from_hex_string():
    # BINARY
    A = np.asarray(["0x0e", "0x06"])
    dtype = DataType.BINARY
    shape = (1, 2, 4)
    eA = [[1, 1, 1, 0], [0, 1, 1, 0]]
    A_unpacked = unpack_innermost_dim_from_hex_string(A, dtype, shape, 8)
    assert (A_unpacked == eA).all()

    A = np.asarray(["0x0e", "0x06"])
    eA_flipped = [[0, 1, 1, 1], [0, 1, 1, 0]]
    A_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        A, dtype, shape, 8, reverse_inner=True
    )
    assert (A_unpacked_flipped == eA_flipped).all()

    # UINT2
    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    dtype = DataType.UINT2
    shape = (1, 2, 2, 2)
    eB = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    B_unpacked = unpack_innermost_dim_from_hex_string(B, dtype, shape, 8)
    assert (B_unpacked == eB).all()

    B = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    eB_flipped = [[[3, 3], [3, 3]], [[3, 1], [1, 3]]]
    B_unpacked_flipped = unpack_innermost_dim_from_hex_string(
        B, dtype, shape, 8, reverse_inner=True
    )
    assert (B_unpacked_flipped == eB_flipped).all()

    # INT2
    C = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    dtype = DataType.INT2
    shape = (1, 2, 2, 2)
    eC = [[[-1, -1], [-1, -1]], [[1, -1], [-1, 1]]]
    C_unpacked = unpack_innermost_dim_from_hex_string(C, dtype, shape, 8)
    assert (C_unpacked == eC).all()

    C = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    dtype = DataType.INT2
    shape = (1, 2, 2, 2)
    eC = [[[-1, -1], [-1, -1]], [[-1, 1], [1, -1]]]
    C_unpacked = unpack_innermost_dim_from_hex_string(
        C, dtype, shape, 8, reverse_inner=True
    )
    assert (C_unpacked == eC).all()

    # INT4
    D = np.asarray(["0x0e", "0x06"])
    dtype = DataType.INT4
    shape = (2, 1)
    eD = [[-2], [6]]
    D_unpacked = unpack_innermost_dim_from_hex_string(D, dtype, shape, 8)
    assert (D_unpacked == eD).all()

    D_unpacked = unpack_innermost_dim_from_hex_string(
        D, dtype, shape, 8, reverse_inner=True
    )
    assert (D_unpacked == eD).all()

    # INT32
    E = np.asarray(["0xffffffff", "0xfffffffe", "0x02", "0xffffffef"])
    dtype = DataType.INT32
    shape = (1, 4, 1)
    eE = [[[-1], [-2], [2], [-17]]]
    E_unpacked = unpack_innermost_dim_from_hex_string(E, dtype, shape, 32)
    assert (E_unpacked == eE).all()

    E_unpacked = unpack_innermost_dim_from_hex_string(
        E, dtype, shape, 32, reverse_inner=True
    )
    assert (E_unpacked == eE).all()

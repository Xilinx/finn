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

from finn.util.basic import pad_tensor_to_multiple_of


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

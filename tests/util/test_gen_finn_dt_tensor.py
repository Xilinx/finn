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

import finn.util.basic as util
from finn.core.datatype import DataType


def test_finn_tensor_generator():
    # bipolar
    shape_bp = [2, 2]
    dt_bp = DataType.BIPOLAR
    tensor_bp = util.gen_finn_dt_tensor(dt_bp, shape_bp)
    # test shape
    for i in range(len(shape_bp)):
        assert (
            shape_bp[i] == tensor_bp.shape[i]
        ), """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_bp.flatten():
        assert dt_bp.allowed(
            value
        ), """Data type of generated tensor
            does not match the desired Data type"""

    # binary
    shape_b = [4, 2, 3]
    dt_b = DataType.BINARY
    tensor_b = util.gen_finn_dt_tensor(dt_b, shape_b)
    # test shape
    for i in range(len(shape_b)):
        assert (
            shape_b[i] == tensor_b.shape[i]
        ), """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_b.flatten():
        assert dt_b.allowed(
            value
        ), """Data type of generated tensor
            does not match the desired Data type"""

    # ternary
    shape_t = [7, 1, 3, 1]
    dt_t = DataType.TERNARY
    tensor_t = util.gen_finn_dt_tensor(dt_t, shape_t)
    # test shape
    for i in range(len(shape_t)):
        assert (
            shape_t[i] == tensor_t.shape[i]
        ), """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_t.flatten():
        assert dt_t.allowed(
            value
        ), """Data type of generated tensor
            does not match the desired Data type"""

    # int2
    shape_int2 = [7, 4]
    dt_int2 = DataType.INT2
    tensor_int2 = util.gen_finn_dt_tensor(dt_int2, shape_int2)
    # test shape
    for i in range(len(shape_int2)):
        assert (
            shape_int2[i] == tensor_int2.shape[i]
        ), """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_int2.flatten():
        assert value in [
            -2,
            -1,
            0,
            1,
        ], """Data type of generated tensor
            does not match the desired Data type"""

    # import pdb; pdb.set_trace()

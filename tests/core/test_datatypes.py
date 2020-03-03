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

from finn.core.datatype import DataType


def test_datatypes():
    assert DataType.BIPOLAR.allowed(-1)
    assert DataType.BIPOLAR.allowed(0) is False
    assert DataType.BINARY.allowed(-1) is False
    assert DataType.BINARY.allowed(1)
    assert DataType.TERNARY.allowed(2) is False
    assert DataType.TERNARY.allowed(-1)
    assert DataType.UINT2.allowed(2)
    assert DataType.UINT2.allowed(10) is False
    assert DataType.UINT3.allowed(5)
    assert DataType.UINT3.allowed(-7) is False
    assert DataType.UINT4.allowed(15)
    assert DataType.UINT4.allowed(150) is False
    assert DataType.UINT8.allowed(150)
    assert DataType.UINT8.allowed(777) is False
    assert DataType.UINT16.allowed(14500)
    assert DataType.UINT16.allowed(-1) is False
    assert DataType.UINT32.allowed(2 ** 10)
    assert DataType.UINT32.allowed(-1) is False
    assert DataType.INT2.allowed(-1)
    assert DataType.INT2.allowed(-10) is False
    assert DataType.INT3.allowed(5) is False
    assert DataType.INT3.allowed(-2)
    assert DataType.INT4.allowed(15) is False
    assert DataType.INT4.allowed(-5)
    assert DataType.INT8.allowed(150) is False
    assert DataType.INT8.allowed(-127)
    assert DataType.INT16.allowed(-1.04) is False
    assert DataType.INT16.allowed(-7777)
    assert DataType.INT32.allowed(7.77) is False
    assert DataType.INT32.allowed(-5)
    assert DataType.INT32.allowed(5)
    assert DataType.BINARY.signed() is False
    assert DataType.FLOAT32.signed()
    assert DataType.BIPOLAR.signed()
    assert DataType.TERNARY.signed()


def test_smallest_possible():
    assert DataType.get_smallest_possible(1) == DataType.BINARY
    assert DataType.get_smallest_possible(1.1) == DataType.FLOAT32
    assert DataType.get_smallest_possible(-1) == DataType.BIPOLAR
    assert DataType.get_smallest_possible(-3) == DataType.INT3
    assert DataType.get_smallest_possible(-3.2) == DataType.FLOAT32

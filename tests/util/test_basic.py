# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

import pytest

import finn.util.basic as basic


@pytest.mark.util
def test_next_power_of_2():
    test_vector = [
        {"input": -2, "expected_result": 0},
        {"input": -1, "expected_result": 0},
        {"input": 0, "expected_result": 0},
        {"input": 1, "expected_result": 2},
        {"input": 2, "expected_result": 2},
        {"input": 3, "expected_result": 4},
        {"input": 4, "expected_result": 4},
        {"input": 7, "expected_result": 8},
        {"input": 8, "expected_result": 8},
        {"input": 11, "expected_result": 16},
        {"input": 15, "expected_result": 16},
        {"input": 16, "expected_result": 16},
        {"input": 18, "expected_result": 32},
        {"input": 27, "expected_result": 32},
        {"input": 31, "expected_result": 32},
        {"input": 32, "expected_result": 32},
        {"input": 42, "expected_result": 64},
        {"input": 65, "expected_result": 128},
    ]

    for test_dict in test_vector:
        output = basic.find_next_power_of_2(test_dict["input"])
        assert output >= test_dict["input"]
        assert output == test_dict["expected_result"]

    return

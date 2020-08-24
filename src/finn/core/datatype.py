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


from enum import Enum, auto

import numpy as np


class DataType(Enum):
    """Enum class that contains FINN data types to set the quantization annotation.
    ONNX does not support data types smaller than 8-bit integers, whereas in FINN we are
    interested in smaller integers down to ternary and bipolar.

    Assignment of DataTypes to indices based on following ordering:

    * unsigned to signed

    * fewer to more bits

    Currently supported DataTypes: """

    # important: the get_smallest_possible() member function is dependent on ordering.
    BINARY = auto()
    UINT2 = auto()
    UINT3 = auto()
    UINT4 = auto()
    UINT5 = auto()
    UINT6 = auto()
    UINT7 = auto()
    UINT8 = auto()
    UINT9 = auto()
    UINT10 = auto()
    UINT11 = auto()
    UINT12 = auto()
    UINT13 = auto()
    UINT14 = auto()
    UINT15 = auto()
    UINT16 = auto()
    UINT17 = auto()
    UINT18 = auto()
    UINT19 = auto()
    UINT20 = auto()
    UINT21 = auto()
    UINT22 = auto()
    UINT23 = auto()
    UINT24 = auto()
    UINT25 = auto()
    UINT26 = auto()
    UINT27 = auto()
    UINT28 = auto()
    UINT29 = auto()
    UINT30 = auto()
    UINT31 = auto()
    UINT32 = auto()
    UINT64 = auto()
    BIPOLAR = auto()
    TERNARY = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    INT5 = auto()
    INT6 = auto()
    INT7 = auto()
    INT8 = auto()
    INT9 = auto()
    INT10 = auto()
    INT11 = auto()
    INT12 = auto()
    INT13 = auto()
    INT14 = auto()
    INT15 = auto()
    INT16 = auto()
    INT17 = auto()
    INT18 = auto()
    INT19 = auto()
    INT20 = auto()
    INT21 = auto()
    INT22 = auto()
    INT23 = auto()
    INT24 = auto()
    INT25 = auto()
    INT26 = auto()
    INT27 = auto()
    INT28 = auto()
    INT29 = auto()
    INT30 = auto()
    INT31 = auto()
    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()

    def bitwidth(self):
        """Returns the number of bits required for this DataType."""

        if self.name.startswith("UINT"):
            return int(self.name.strip("UINT"))
        elif self.name.startswith("INT"):
            return int(self.name.strip("INT"))
        elif "FLOAT" in self.name:
            return int(self.name.strip("FLOAT"))
        elif self.name in ["BINARY", "BIPOLAR"]:
            return 1
        elif self.name == "TERNARY":
            return 2
        else:
            raise Exception("Unrecognized data type: %s" % self.name)

    def min(self):
        """Returns the smallest possible value allowed by this DataType."""

        if self.name.startswith("UINT") or self.name == "BINARY":
            return 0
        elif self.name.startswith("INT"):
            return -(2 ** (self.bitwidth() - 1))
        elif self.name == "FLOAT32":
            return np.finfo(np.float32).min
        elif self.name == "BIPOLAR":
            return -1
        elif self.name == "TERNARY":
            return -1
        else:
            raise Exception("Unrecognized data type: %s" % self.name)

    def max(self):
        """Returns the largest possible value allowed by this DataType."""

        if self.name.startswith("UINT"):
            return (2 ** (self.bitwidth())) - 1
        elif self.name == "BINARY":
            return +1
        elif self.name.startswith("INT"):
            return (2 ** (self.bitwidth() - 1)) - 1
        elif self.name == "FLOAT32":
            return np.finfo(np.float32).max
        elif self.name == "BIPOLAR":
            return +1
        elif self.name == "TERNARY":
            return +1
        else:
            raise Exception("Unrecognized data type: %s" % self.name)

    def allowed(self, value):
        """Check whether given value is allowed for this DataType.

        * value (float32): value to be checked"""

        if "FLOAT" in self.name:
            return True
        elif "INT" in self.name:
            return (
                (self.min() <= value)
                and (value <= self.max())
                and float(value).is_integer()
            )
        elif self.name == "BINARY":
            return value in [0, 1]
        elif self.name == "BIPOLAR":
            return value in [-1, +1]
        elif self.name == "TERNARY":
            return value in [-1, 0, +1]
        else:
            raise Exception("Unrecognized data type: %s" % self.name)

    def get_num_possible_values(self):
        """Returns the number of possible values this DataType can take. Only
        implemented for integer types for now."""
        assert self.is_integer(), """This function only works for integers for now,
        not for the DataType you used this function with."""
        if "INT" in self.name:
            return abs(self.min()) + abs(self.max()) + 1
        elif self.name == "BINARY" or self.name == "BIPOLAR":
            return 2
        elif self.name == "TERNARY":
            return 3

    def get_smallest_possible(value):
        """Returns smallest (fewest bits) possible DataType that can represent
      value. Prefers unsigned integers where possible."""
        if not int(value) == value:
            return DataType["FLOAT32"]
        for k in DataType.__members__:
            dt = DataType[k]
            if (dt.min() <= value) and (value <= dt.max()):
                return dt

    def signed(self):
        """Returns whether this DataType can represent negative numbers."""
        return self.min() < 0

    def is_integer(self):
        """Returns whether this DataType represents integer values only."""
        # only FLOAT32 is noninteger for now
        return self != DataType.FLOAT32

    def get_hls_datatype_str(self):
        """Returns the corresponding Vivado HLS datatype name."""
        if self.is_integer():
            if self.signed():
                return "ap_int<%d>" % self.bitwidth()
            else:
                return "ap_uint<%d>" % self.bitwidth()
        else:
            return "float"

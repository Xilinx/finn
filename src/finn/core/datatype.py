# Copyright (c) 2018, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum, auto

import numpy as np


class DataType(Enum):
    # important to maintain ordering here: unsigned to signed, fewer to more
    # bits. The get_smallest_possible() member function is dependent on this.
    BINARY = auto()
    UINT2 = auto()
    UINT3 = auto()
    UINT4 = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    BIPOLAR = auto()
    TERNARY = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
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

    value (float32): value to be checked"""

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

    def get_smallest_possible(value):
        """Return smallest (fewest bits) possible DataType that can represent
      value. Prefers unsigned integers where possible."""
        if not int(value) == value:
            return DataType["FLOAT32"]
        for k in DataType.__members__:
            dt = DataType[k]
            if (dt.min() <= value) and (value <= dt.max()):
                return dt

    def signed(self):
        """Return whether this DataType can represent negative numbers."""
        return self.min() < 0

    def is_integer(self):
        """Return whether this DataType represents integer values only."""
        # only FLOAT32 is noninteger for now
        return self != DataType.FLOAT32

    def get_hls_datatype_str(self):
        """Return the corresponding Vivado HLS datatype name."""
        if self.is_integer():
            if self.signed():
                return "ap_int<%d>" % self.bitwidth()
            else:
                return "ap_uint<%d>" % self.bitwidth()
        else:
            return "float"

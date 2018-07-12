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
import numpy as np

# Various quantization functions used in FINN.

def quantize_matrix(W, b):
    """
    Return (Wb, a) where Wb is a b-bit signed integer matrix and a is scale
    factors a s.t. W ~= a*Wb. If b is 1, a bipolar matrix is returned instead.
    """
    assert(b > 0)
    # compute the maxnorm of each row
    alpha = np.max(np.abs(W), axis=1)
    # normalize wrt maxnorm. Wnorm elements are now in [-1, +1].
    Wnorm = W / alpha[:, None]
    if b == 1:
        # just apply sign function to get a bipolar matrix
        Wnorm = np.sign(Wnorm)
    else:
        # apply fixed-point scaling factor of 2^(b-1)
        fixpscale = (1 << (b-1)) - 1
        # Wnorm elements are now in [-fixpscale, fixpscale]
        Wnorm = Wnorm * fixpscale
        # adjust alpha to cancel out the fixed-point scaling
        alpha = alpha / fixpscale

    # round Wnorm elements to nearest integers
    Wint = np.rint(Wnorm)
    return (Wint, alpha)

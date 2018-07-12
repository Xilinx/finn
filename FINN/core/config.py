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
"""Global FINN config """
__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"

import os
import sys

# Directory variables
try:
    FINN_ROOT = os.environ['FINN_ROOT']
except:
    print "Environmental variable FINN_ROOT must be set top level FINN directory...exiting"
    sys.exit(-1)


DEVICE_DIR = FINN_ROOT+'/devices/'
DATA_DIR = FINN_ROOT+'/data/'

# Constants on folding factors
SIMD_MAX = 32
PE_MAX = 32
MMV_MAX = 64 


# Resource constraints
LUT_PROPORTION = 0.7
BRAM_PROPORTION = 1
LUTS_PER_OP = 2.5

# index: product of bitwidths, values: op cost in LUTs
# TODO make these more precise from MVTU data
prec_op_cost = [0, 2.5, 4.1, 5.7, 7.3, 8.9, 10.5, 12.1, 13.7]

# Include Paths for external tools
GXX_ARGS = ""

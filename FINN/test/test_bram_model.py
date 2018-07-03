#!/usr/bin/env python
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


from FINN.core import perf_model
from FINN.frontend.caffeloader import CaffeLoader
import FINN.core.nn
from FINN.core import device as device


def demo_hwgq_import():
    l = CaffeLoader(None, "inputs/sfc.prototxt")
    net = FINN.core.nn.NN(l)
    dev = device.Device('XLNX:KU115.json')
    perf = perf_model.PerfModel(net,dev)
    
    perf.print_folding_factors()
    perf.print_hardware_cost()
    

    for idx,val in enumerate(perf.SIMD):
        perf.SIMD[idx]  = 5
        #perf.PE[idx]  = 10
    perf.print_folding_factors()
    perf.print_hardware_cost()
    
    for idx,val in enumerate(perf.SIMD):
        perf.SIMD[idx]  = 20
        #perf.PE[idx]  = 100
    perf.print_folding_factors()
    perf.print_hardware_cost()

demo_hwgq_import()

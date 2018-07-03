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

__author__ = "Ken O'Brien"
__email__ = "kennetho@xilinx.com"

from FINN.core import perf_model
from FINN.frontend.caffeloader import CaffeLoader
import FINN.core.nn
from FINN.core import device as device
from FINN.core import layers as layers
import numpy as np
import logging

def demo_lfc():
    logging.basicConfig(filename='FINN.log', level=logging.INFO) # Changed WARNING to INFO if you want logging
    lfcnetwork = []
    W0 = np.zeros((1024, 832)) # OutChans, InChans
    W1 = np.zeros((1024, 1024))
    W2 = np.zeros((1024, 1024))
    W3 = np.zeros((64, 1024))
 
    lfcnetwork.append(layers.FullyConnectedLayer(W0, 1,1,1)) # wbits, ibits, obits
    lfcnetwork.append(layers.FullyConnectedLayer(W1, 1,1,1))
    lfcnetwork.append(layers.FullyConnectedLayer(W2, 1,1,1)) 
    lfcnetwork.append(layers.FullyConnectedLayer(W3, 1,1,1)) 

    net = FINN.core.nn.NN(layers=lfcnetwork)
    
    dev = device.Device('XLNX:VU9P.json', frequency=192.4)
    perf = perf_model.PerfModel(net,dev)
    
    fps = perf.maximise_fps() 
    
   # perf.SIMD[0] = 64
   # perf.SIMD[1] = 64
   # perf.SIMD[2] = 64
   # perf.SIMD[3] = 64
   # 
   # perf.PE[0] = 256
   # perf.PE[1] = 256
   # perf.PE[2] = 256
   # perf.PE[3] = 16

    fps = perf.fps()

    
    perf.nswg.calculate_neural_folding()
    perf.nswg.calculate_write_block_cycles()
    perf.nswg.calculate_read_block_cycles()
    perf.nswg.calculate_total_cycles()
    perf.nswg.calculate_input_multipliers()
    perf.print_folding_factors()
    perf.print_hardware_cost()
    perf.print_topology()
    perf.print_cycles()
    fps = perf.fps()

    print "Achieved fps of %f with %f%% LUT utilisation and %f%% BRAM utilisation at %f Mhz" % (fps, perf.network_utilisation()['luts']/dev.luts*100, perf.network_utilisation()['brams']/dev.brams*100, dev.frequency) 

if __name__ == "__main__":
    demo_lfc()

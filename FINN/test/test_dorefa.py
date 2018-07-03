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

def demo_dorefa():
    logging.basicConfig(filename='FINN.log', level=logging.INFO) # Changed WARNING to INFO if you want logging
    dorefanetwork = []
    
    W0 = np.zeros((68, 3, 12, 12)) # out, in, kernel, kernel
    W1 = np.zeros((90, 34, 5,5))
    W2 = np.zeros((272, 180, 3,3))
    W3 = np.zeros((192, 136,3,3))
    W4 = np.zeros((128, 192,3,3))
    W5 = np.zeros((4096, 9216))
    W6 = np.zeros((4096, 4096))
    W7 = np.zeros((1000, 4096))
 
    dorefanetwork.append(layers.ConvolutionLayer(W0, 227, 0, 4,1,1,1,0)) # in_dim, pad, stride, wbits, ibits, obits 
    dorefanetwork.append(layers.ConvolutionLayer(W1, 58, 0, 1,1,1,1,0)) 
    dorefanetwork[-1].parallel = 2
    dorefanetwork.append(layers.ConvolutionLayer(W2, 29, 0, 1,1,1,1,0)) 
    dorefanetwork.append(layers.ConvolutionLayer(W3, 16, 0, 1,1,1,1,0)) 
    dorefanetwork[-1].parallel = 2
    dorefanetwork.append(layers.ConvolutionLayer(W4, 16, 0, 1,1,1,1,0)) 
    dorefanetwork[-1].parallel = 2

    dorefanetwork.append(layers.FullyConnectedLayer(W5, 1,1,1))
    dorefanetwork.append(layers.FullyConnectedLayer(W6, 1,1,1)) 
    dorefanetwork.append(layers.FullyConnectedLayer(W7, 1,1,1)) 



    net = FINN.core.nn.NN(layers=dorefanetwork)
    
    dev = device.Device('XLNX:VU9P.json', frequency=101) # Measured on AWS
    perf = perf_model.PerfModel(net,dev)
    
    
    # From BNN spreadsheet, t3
    perf.SIMD[0] = 3
    perf.SIMD[1] = 34
    perf.SIMD[2] = 45
    perf.SIMD[3] = 34
    perf.SIMD[4] = 64
    perf.SIMD[5] = 64
    perf.SIMD[6] = 64
    perf.SIMD[7] = 8
    
    perf.PE[0] = 68
    perf.PE[1] = 90
    perf.PE[2] = 136
    perf.PE[3] = 64
    perf.PE[4] = 32
    perf.PE[5] = 32
    perf.PE[6] = 16
    perf.PE[7] = 32
  
    perf.MMV[0] = 18
    perf.MMV[1] =  3
    perf.MMV[2] =  3
    perf.MMV[3] =  1
    perf.MMV[4] =  1
    perf.MMV[5] =  1
    perf.MMV[6] =  1
    perf.MMV[7] =  1
    
    # FPS given the above folding factors
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

    print(perf.nswg)
    print "Achieved fps of %f with %f%% LUT utilisation and %f%% BRAM utilisation at %f Mhz" % (fps, perf.network_utilisation()['luts']/dev.luts*100, perf.network_utilisation()['brams']/dev.brams*100, dev.frequency) 

if __name__ == "__main__":
    demo_dorefa()

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

__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"


from FINN.core import layers as lb
from FINN.backend.fpga import layers_fpga as lfpga

class NSWG:

    def __init__(self, net, perf):
        self.net = net
        self.perf = perf
        self.input_multiplier = self._ones() 
        self.initial_buffer = self._zeros()
        self.write_block_cycles = self._zeros()
        self.read_block_cycles = self._zeros()
        self.total_cycles = self._zeros()
        self.calculate_matrix_sizes()
        self.calculate_neural_folding()
        self.calculate_initial_buffers()        

    def _zeros(self):
        zeros = []
        for i in range(len(self.net.layers)):
            zeros.append(0)
        return zeros
    
    def _ones(self):
        ones = []
        for i in range(len(self.net.layers)):
            ones.append(1)
        return ones

    def calculate_input_multipliers(self):
        """If the NSWG is the bottleneck of the MVTU, increase the input multiplier to 2"""
        self.calculate_total_cycles()
        mvtu_cycles = self.perf.calculate_layer_cycles()
        bottleneck = True
        while bottleneck:
            largest_mvtu = max(mvtu_cycles)
            largest_swg = max(self.total_cycles)
            if largest_swg > largest_mvtu:
                idx = self.total_cycles.index(largest_swg)
                if self.input_multiplier[idx] == 2:
                    bottleneck = False # Can't increase
                else:
                    self.input_multiplier[idx] = 2
                    self.calculate_total_cycles()
            else:
                bottleneck = False
        self.calculate_total_cycles()

    def calculate_total_cycles(self):
        self.calculate_initial_buffers()
        self.calculate_read_block_cycles()
        self.calculate_write_block_cycles()
        for i in range(len(self.net.layers)):
            layer = self.net.layers[i]
            if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
                self.total_cycles[i] = self.initial_buffer[i] + layer.get_out_dim() * max(self.write_block_cycles[i], self.read_block_cycles[i])

            
    def calculate_initial_buffers(self):
        for i in range(len(self.net.layers)):
            layer = self.net.layers[i]
            if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
                self.initial_buffer[i] = (layer.get_in_dim() * layer.get_filter_dim()) / self.input_multiplier[i]
        
    def calculate_write_block_cycles(self):
        for i in range(len(self.net.layers)):
            layer = self.net.layers[i]
            if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
                self.write_block_cycles[i] = (layer.get_out_dim() *layer.get_filter_dim()*layer.get_filter_dim()) / self.perf.MMV[i]
        
    def calculate_read_block_cycles(self):
        for i in range(len(self.net.layers)):
            layer = self.net.layers[i]
            if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
                self.read_block_cycles[i] = (layer.get_stride() *layer.get_in_dim()) / self.input_multiplier[i]

    def calculate_matrix_sizes(self):
        self.matrixH = self._zeros()
        self.matrixW = self._zeros()
        
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                self.matrixW[i] = self.net.layers[i].getOutputSize()
                self.matrixH[i] = self.net.layers[i].getInputSize() * self.net.layers[i].get_filter_dim() * self.net.layers[i].get_filter_dim()
    


    def __str__(self):
        strbuf = 'NSWG: \n'
        strbuf += '{0:>8} {1:>8}\n'.format('MatrixH', 'MatrixW')
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                strbuf += '{0:>8} {1:>8}\n'.format(str(self.matrixH[i]), str(self.matrixW[i]))
        strbuf += '\n'
        strbuf += '{0:>8} {1:>8}\n'.format('Synaptic', 'Neuron')
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                strbuf += '{0:>8} {1:>8}\n'.format(self.synapse_fold[i], self.neuron_fold[i])
        strbuf += '\n'
        strbuf += '{0:>16} {1:>20} {2:>20} {3:>20} {4:>20}\n'.format('Initial Buffer', 'Write Block Cycles','Read Block Cycles', 'Total Cycles', 'Input Multiplier')
        for i in range(len(self.net.layers)):
            if lb.isConvLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                strbuf += '{0:>16} {1:>20} {2:>20} {3:>20} {4:>20}\n'.format(self.initial_buffer[i], self.write_block_cycles[i], self.read_block_cycles[i], self.total_cycles[i], self.input_multiplier[i])
        return strbuf 

    def calculate_neural_folding(self):
        self.synapse_fold = self._zeros()
        self.neuron_fold = self._zeros()
        
        for i in range(len(self.net.layers)):
            layer = self.net.layers[i]
            if lb.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                self.synapse_fold[i] = self.matrixH[i] / self.perf.SIMD[i]
                self.neuron_fold[i] = self.matrixW[i] / self.perf.PE[i]
        

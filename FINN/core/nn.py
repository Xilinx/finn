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

import nnimporter
import sys
from FINN.core import layers
from FINN.frontend.loader import Loader
from FINN.backend.fpga import layers_fpga as lfpga
from FINN.frontend.caffeloader import CaffeLoader
from util import factors, next_factor

class NN:
	def __init__(self, loader=None, layers=None):
		"""Standard class representation of a Neural Network"""
		if loader is not None:
			self.layers = loader.load()
		elif layers is not None:
			self.layers = layers
		else:
			raise Exception("At least one of loader or layers must be specified")
		assert(len(self.layers)>0)
		self.num_weights = []
		self.num_activations = []
		self.SIMD = []
		self.PE = []
		self.MMV = []
		self.ops = []
		#self.output+= _bits()
		#self.filter_relevant_layers()
		#self.set_ops()
		self.calculate_weight_counts()
		self.calculate_activation_counts()
                #self.layers = self.conv_layers()

	def execPipeline(self, i):
		"""
		Forward-propagate given image through this network.
		Returns a tuple (result, intermediates) where result is the output
		from the final layer, and intermediates is an array containing outputs
		from all intermediate layers in the network.
		"""
		intermediates = [i]
		for L in self.layers:
		    i = L.execute(i)
		    intermediates += [i]
		return (i, intermediates)

	def count_matrix_layers(self):
	    count = 0
	    for l in self.layers:
	        if layers.isMatrixLayer(l):
	            count+=1
	    return count

	def print_bits(self):
	    for layer in self.layers:
	        output+=  layer.ibits

	def filter_relevant_layers(self):
	    layers = []
	    for l in self.layers:
	        if layers.isMatrixLayer(l):
	            layers.append(l)
	    self.layers = layers

	def __repr__(self):
	    return summarizePipeline(self.layers)

	def calculate_weight_counts(self):
	    for layer in self.layers:
	        if layers.isMatrixLayer(layer):
	            self.num_weights.append(layer.getParamSize())
                     #self.num_weights.append(layer.getOutputSize() * layer.get_filter_dim() * layer.getInputSize()  * self.parallel_per_layer(layer))

	def calculate_activation_counts(self):
		for layer in self.layers:
			if layers.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(layer):
				self.num_activations.append(layer.get_out_dim() * layer.get_out_dim() * layer.getOutputSize() )

	def conv_layers(self):
		"""Filter for conv layers only"""
		return [conv for conv in self.layers if conv['type']== u"conv" or conv['type']==u"fc"]

	def ops_per_layer(self, layer):
		""" if layerType is pool:
				return out_dim * out_dim * filter_dim * filter_dim
			else layerType is conv or fc
				return parallel * 2 * out_dim * out_dim * filter_dim * filter_dim * in_channels * out_channel
		"""
		if layers.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(layer):
			return layer.getNumOps() #/2
		return 0

	def parallel_per_layer(self, layer):
            parallel = 1
	    if layers.isMatrixLayer(layer):
	        if hasattr(layer, 'parallel'):
                    parallel = layer.parallel
	    return parallel

        def print_weights(self):
            print "Layer # | weights"
            for idx, weight in enumerate(self.num_weights):
                print idx, weight
            print "\n"
        
        def print_activations(self):
            print "Layer # | activations"
            for idx, activation in enumerate(self.num_activations):
                print idx, activation
            print "\n"
                

	def generate_header(self):
		"""Write a C/C++ header file with folding factors for inclusion in HLS"""
		with open('config.h', 'w') as config:
			config.write("/*\n")
			config.write("* FINN header file\n")
			config.write("*/\n\n")
			for idx, layer in enumerate(self.layers):
				config.write("#define %s %d\n" % ("SIMD_"+str(idx), layer['SIMD']))
				config.write("#define %s %d\n" % ("PE_"+str(idx), layer['PE']))
				config.write("#define %s %d\n\n" % ("MMV_"+str(idx), layer['MMV']))

def execPipeline(i, pipeline):
	"""
	Deprecated: use FINN.core.nn.NN.execPipeline instead.
	Forward-propagate given image through given pipeline.
	"""
	intermediates = [i]
	for L in pipeline:
	    i = L.execute(i)
	    intermediates += [i]
	return (i, intermediates)

def summarizePipeline(pipeline):
    totalParams = 0
    totalParamBits = 0
    totalOps = 0
    totalComputeLayers = 0
    output = ""
    output+=  "Per-layer details\n"
    output+=  "==================\n"
    for l in pipeline:
        print l.__class__.__name__
        if layers.isMatrixLayer(l):
            np = l.getParamSize()
            op = l.getNumOps()
            ins = l.getInputSize()
            outs = l.getOutputSize()
            npbits = l.getTotalParamBits()
            totalParamBits += npbits
            totalParams += np
            totalOps += op
            totalComputeLayers += 1
            output += "Type: %s, params: %d, ops: %d, in = %s, out = %s\n" % (l.__class__.__name__, np, op, str(ins), str(outs))
            output += "Bitwidths: input %d, weight %d, output %d\n" % (l.ibits, l.wbits, l.obits)
            inbits = l.getTotalInputBits()
            outbits = l.getTotalOutputBits()
            output += "Total in bits: %d, total weight bits: %d, total out bits: %d\n" % (inbits, npbits, outbits)
            # arithmetic intensity with some components on-chip
            # TODO include output activations once threshold fusion is in place
            #ai_none = float(op) / float(inbits + outbits + npbits)
            #ai_w = float(op) / float(inbits + outbits)
            #ai_wi = float(op) / float(outbits)
            #ai_i = float(op) / float(npbits + outbits)
            ai_wo = float(op) / float(inbits)
            ai_io = float(op) / float(npbits)
            ai_o = float(op) / float(inbits + npbits)
            #output+=  "AI none: %f, w: %f, wi: %f, wo: %f, io: %f, i: %f, o: %f" % (ai_none, ai_w, ai_wi, ai_wo, ai_io, ai_i, ai_o)
            output += "AI on-chip wo: %f, io: %f, o: %f\n" % (ai_wo, ai_io, ai_o)
            output += "-----\n"
    output+=  "Neural network pipeline summary\n"
    output+=  "================================\n"
    output+=  "Pipeline contains %d layers, %d of which are matrix layers\n" % (len(pipeline), totalComputeLayers)
    output+=  "Number of parameters: %f million\n" % (float(totalParams) / 1000000.0)
    output+=  "Total parameter volume: %f MB\n" % (float(totalParamBits) / (8*1024*1024))
    output+=  "Operations per inference: %f million\n" % (float(totalOps) / 1000000.0)
    return output

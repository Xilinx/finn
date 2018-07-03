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

import FINN.core.nn
import FINN.core.device
import FINN.core.perf_candidate as pc
import logging

import config
from util import factors, next_factor, prev_factor, is_factor
import numpy as np
from nswg import NSWG
import sys
import copy

from FINN.core import layers as lb
from FINN.backend.fpga import layers_fpga as lfpga


class PerfModel:

    def __init__(self, net, dev, enableMMV):
        self.net = net
        self.dev = dev
        self.candidates = []
        self.init_folding_factor()
        self.nswg = NSWG(net, self)
        self.enableMMV = enableMMV

    def init_folding_factor(self):
        self.SIMD = []
        self.PE = []
        self.MMV = []
        for i in xrange(len(self.net.layers)):
            self.SIMD.append(1)
            self.PE.append(1)
            self.MMV.append(1)

    def fits_in_device(self):
        """ Given a device, calculate the resources used and determine if the network with this configuration fits"""
        total_cost = self.network_utilisation()
        enoughLuts = total_cost['luts'] <= (
            self.dev.luts * self.dev.lut_proportion)
        enoughBrams = total_cost['brams'] <= self.dev.brams*self.dev.bram_proportion
        if enoughLuts and enoughBrams:
            logging.info(
                "Design fits in device LUTS: %s BRAMS: %s  %d/%d %d/%d", enoughLuts, enoughBrams, total_cost['luts'], self.dev.luts*self.dev.lut_proportion, total_cost['brams'], self.dev.brams)
        else:
            logging.info(
                "Design does not fit in device LUTS: %s BRAMS: %s  %d/%d %d/%d", enoughLuts, enoughBrams, total_cost['luts'], self.dev.luts*self.dev.lut_proportion, total_cost['brams'], self.dev.brams)
        logging.info(self.SIMD)
        logging.info(self.PE)
        return enoughLuts and enoughBrams

    def network_utilisation(self):
        total_cost = {}
        total_cost['luts'] = 0
        total_cost['brams'] = 0
        # XXX Need to update bitprecision
        bitprecision = 1
        for idx, layer in enumerate(self.net.layers):
            if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer", "FullyConnectedLayer", "FPGABipolarMatrixThresholdLayer", "FPGABipolarMatrixLayer"]:
                logging.info("Layer %d, LUTS: %d BRAM %d %d" % (idx, self.lut_cost(idx), self.bram_cost(idx)[0], self.bram_cost(idx)[1]))
                total_cost['luts'] += self.lut_cost(idx)
                total_cost['brams'] += sum(self.bram_cost(idx))
        return total_cost

    def generate_header(self):
        """Write a C/C++ header file with folding factors for inclusion in HLS"""
        with open('config.h', 'w') as config:
            config.write("/*\n")
            config.write("* FINN header file\n")
            config.write("*/\n\n")
            for idx, layer in enumerate(self.net.layers):
                config.write("#define %s %d\n" %
                             ("SIMD_" + str(idx), self.SIMD[idx]))
                config.write("#define %s %d\n" %
                             ("PE_" + str(idx), self.PE[idx]))
                config.write("#define %s %d\n\n" %
                             ("MMV_" + str(idx), self.MMV[idx]))

    def max_ops(self):
        max = -1
        maxidx = -1
        for idx, layer in enumerate(self.net.layers):
            if self.net.ops_per_layer(layer) > max:
                max = self.net.ops_per_layer(layer)
                maxidx = idx
        return maxidx, max

    def print_hardware_cost(self):
        print "\nHardware Cost:"
        print '{0:>35} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format('Layer', 'idx', 'Input BRAMS', 'Weights BRAM', 'Total LUTS', 'Total BRAM')
        total_input_brams = 0
        total_weights_brams = 0
        total_buffer_brams = 0
        total_luts = 0
        total_brams = 0
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                print '{0:>35} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format(self.net.layers[i].get_type(), i, self.bram_cost(i)[0], self.bram_cost(i)[1], self.lut_cost(i), sum(self.bram_cost(i)))
                brams = self.bram_cost(i)
                total_input_brams += brams[0]
                total_weights_brams += brams[1]
                total_luts += self.lut_cost(i)
                total_brams += sum(brams)
        print '{0:>35} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format("Totals", "ALL", total_input_brams, total_weights_brams, total_luts, total_brams)
        print ""

    def print_cycles(self):
        print "\nCycles per layer: "
        layer_cycles = self.calculate_layer_cycles()  # Same as est MVC
        print '{0:>35} {1:>8}  {2:>10} {3:>10}'.format('NAME', 'idx', 'ops/layer', 'MVC')
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                print '{0:>35} {1:>8}  {2:>10} {3:>10}'.format(self.net.layers[i].get_type(), i, self.net.ops_per_layer(self.net.layers[i]), layer_cycles[i])
        print ""

    def print_folding_factors(self):
        print "\nFolding factors: "
        print '{0:>35} {1:>8} {2:>5} {3:>5} {4:>5}'.format('NAME', 'idx', 'SIMD', 'PE', 'MMV')
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                print '{0:>35} {1:>8} {2:>5} {3:>5} {4:>5} '.format(self.net.layers[i].get_type(), i, self.SIMD[i], self.PE[i], self.MMV[i])
        print ""

    def print_topology(self):
        print "\nNetwork Topology: "
        print '{0:>35} {1:>10} {2:>10} {3:>10} {4:>10} {5:>8} {6:>8} {7:>8}'.format('NAME', 'idx', 'out_dim', 'filter_dim', 'in_chan', 'out_chan', 'stride', 'in_dim')
        for i in range(len(self.net.layers)):
            if lb.isMatrixLayer(self.net.layers[i]) or lfpga.isFPGAMatrixLayer(self.net.layers[i]):
                print '{0:>35} {1:>10} {2:>10} {3:>10} {4:>10} {5:>8} {6:>8} {7:>8}'.format(self.net.layers[i].get_type(), i, self.net.layers[i].get_out_dim(), self.net.layers[i].get_filter_dim(), self.net.layers[i].getInputSize(), self.net.layers[i].getOutputSize(), self.net.layers[i].get_stride(), self.net.layers[i].get_in_dim())
        print ""

    def bram_cost(self, idx):
        """ input_gen = (ceil(kernel/stride)+1) * ceil(ifm_ch/36) * ceil((ifm_dim*stride)/512)
            wmem = PE * ceil(SIMD/36) * ceil(wmem0*36/512)
            wmem0 =  ((KERNEL*KERNEL) *IFM_CH * OFM_CH) / SIMD * PE
            """
        input_gen = 0
        wmem0 = 0
        wmem = 0
        in_chans = self.net.layers[idx].getInputSize()
        out_chans = self.net.layers[idx].getOutputSize()
        out_dim = self.net.layers[idx].get_out_dim()

        if isinstance(in_chans, tuple):
            in_chans = in_chans[0]
        if isinstance(out_chans, tuple):
            out_chans = out_chans[0]
        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        # Sane preprocessing of the inputs
        M = float(self.MMV[idx])
        K = float(self.net.layers[idx].kernel)
        N = float(out_dim)
        C = float(in_chans)
        Cp = float(out_chans)
        SIMD = float(self.SIMD[idx])
        PE = float(self.PE[idx])
        A = 1.0
        W = 1.0
        S = None
		# SWG
        input_gen = 0
        layer = self.net.layers[idx]
        if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
            S = float(self.net.layers[idx].get_stride())
            input_gen =  (np.ceil(K/S)+1) * (np.ceil(S*N*C/SIMD)/512) * np.ceil((SIMD*A)/36.0)
		# WM
	    WM = 0
        if layer.get_type() in ["ConvolutionLayer", "FPGABipolarConvThresholdLayer"]:
            WM = (K*K*C*Cp)/ (SIMD*PE)
        elif layer.get_type() in ["FullyConnectedLayer", "FPGABipolarMatrixThresholdLayer", "FPGABipolarMatrixLayer"]:
            WM = (C*Cp) / (SIMD*PE)
        else:
            WM = 0
        wmem = np.ceil(PE * (np.ceil((WM)/512.0) * np.ceil((SIMD*W)/36.0)) * min(SIMD*W/36.0, 1.0 ))
        return (input_gen, wmem,0)

    def lut_cost(self, idx):
        """ LUT Cost =  Q-element dot products = Q*W*A
                     += Accumulators = W + A + log2(C) + 2 * log2(K)
                     += Counters = log2(N) + log2(C) + log2(C')
        """
        in_chans = self.net.layers[idx].getInputSize()
        out_chans = self.net.layers[idx].getOutputSize()
        out_dim = self.net.layers[idx].get_out_dim()

        if isinstance(in_chans, tuple):
            in_chans = in_chans[0]
        if isinstance(out_chans, tuple):
            out_chans = out_chans[0]
        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]
        cost = (self.ops_per_cycle(idx) * self.net.layers[idx].ibits * self.net.layers[idx].wbits) + (self.net.layers[idx].wbits + self.net.layers[idx].ibits + np.log2(in_chans) + 2 * +np.log2(self.net.layers[idx].kernel)) + (np.log2(in_chans) + np.log2(in_chans) + np.log2(out_chans))
        return cost


    def find_first_matrix_layer(self):
        first = -1
        for idx, layer in enumerate(self.net.layers):
            if lb.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(layer):
                first = idx
                break
        assert(first != -1)
        return first

    def find_slowest_layer(self):
        """Find worst case layer as index into layers"""
        slowest_layer = self.find_first_matrix_layer()
        cycles = self.calculate_layer_cycles()
        for idx, cycle in enumerate(cycles):
            if cycle > cycles[slowest_layer] and (lb.isMatrixLayer(self.net.layers[idx]) or lfpga.isFPGAMatrixLayer(self.net.layers[idx])):
                slowest_layer = idx
        return slowest_layer

    def calculate_layer_cycles(self):  # Same as est MVC
        """ For each layer, calculate cycles required
        Formula is ops_per_layer() / ops_per_cycle()"""
        layer_cycles = []
        for idx, layer in enumerate(self.net.layers):
            if lb.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(layer):
                layer_cycles.append(self.net.ops_per_layer(
                    layer) / (self.ops_per_cycle(idx) * self.net.parallel_per_layer(layer)))
            else:
                layer_cycles.append(0)
        return layer_cycles

	
    def legal_folding_factors(self):
		for idx in range(len(self.net.layers)):
			layer_legal = self.SIMD[idx] <= config.SIMD_MAX and self.PE[idx] <= config.PE_MAX and self.MMV[idx] <= config.MMV_MAX
			if not layer_legal:
				logging.info(
                	"Illegal layer idx: %d %d %d %d", idx, self.SIMD[idx], self.PE[idx], self.MMV[idx])
				return False
		return True

    def increase_resources(self):
        """	Increase SIMD, PE, MMV of each layer, balancing the throughput
        Choose which parameter to increase
        SIMD can be increased to IFM channels
        PE can be increased to OFM channels
        MMV can be increased to OFM dim"""
        slowest_layer = self.find_slowest_layer()
        parallel = 1
        in_chans = self.net.layers[slowest_layer].getInputSize()
        out_chans = self.net.layers[slowest_layer].getOutputSize()
        out_dim = self.net.layers[slowest_layer].get_out_dim()

        if self.SIMD[slowest_layer] < in_chans and self.SIMD[slowest_layer] < config.SIMD_MAX and next_factor(in_chans, self.SIMD[slowest_layer]) <= config.SIMD_MAX:
            self.SIMD[slowest_layer] = next_factor(in_chans, self.SIMD[slowest_layer])
        # XXX Usually out_channels/parallel
        elif self.PE[slowest_layer] < out_chans and self.PE[slowest_layer] < config.PE_MAX and next_factor(out_chans, self.PE[slowest_layer]) <= config.PE_MAX:
            self.PE[slowest_layer] = next_factor(out_chans, self.PE[slowest_layer])
        elif self.enableMMV and self.MMV[slowest_layer] < out_dim and self.MMV[slowest_layer] < config.MMV_MAX and next_factor(out_dim, self.MMV[slowest_layer]) <= config.MMV_MAX:
            self.MMV[slowest_layer] = next_factor(out_dim, self.MMV[slowest_layer])
        else:
            logging.info(
                "Cannot raise parallelism of slowest layer %d", slowest_layer)
            return False
        # Now that we've increased within the constraints, does the resulting
        # network fit? If not, revert
        if self.fits_in_device():
			self.candidates.append(pc.PerfCandidate(self.SIMD, self.PE, self.MMV, self.fps()))
        return True

    def maximise_resources(self, target_fps):
        """ Algorithm:
        All layer's SIMD = PE = MMV = 1
        while the estimated fps is less than target
        Find the layer with the highest cycle count, increase its resources
        Check it fits on device"""
        while self.fps() <= target_fps:
            if not self.increase_resources():
                break
        return self.fps()

    def maximise_fps(self):
        """Increase the resources until we can't fit or parallelise any further"""
        while(self.increase_resources()):
			pass
        if len(self.candidates) == 0:
			print "This network cannot fit in this device"
			sys.exit(1)
        else:
            print "There are %d candidates" % (len(self.candidates))
            self.candidates = sorted(self.candidates)
        self.SIMD = list(self.candidates[-1].SIMD)
        self.PE = list(self.candidates[-1].PE)
        self.MMV = list(self.candidates[-1].MMV)
        logging.info("SIMD")
        logging.info('[%s]' % ', '.join(map(str,self.SIMD)))
        logging.info("PE")
        logging.info('[%s]' % ', '.join(map(str,self.PE)))
        logging.info("MMV")
        logging.info('[%s]' % ', '.join(map(str,self.MMV)))
        logging.info(self.nswg)
        out = sys.stdout 
        with open("FINN.rpt", "w") as sys.stdout:
            self.print_topology()
            self.print_folding_factors()
            print self.nswg
            self.print_hardware_cost()
            self.net.print_weights()
            self.net.print_activations()
        sys.stdout = out
        return self.fps()

    def ops_per_cycle(self, layer_idx):
        if lb.isMatrixLayer(self.net.layers[layer_idx]) or lfpga.isFPGAMatrixLayer(self.net.layers[layer_idx]):
            # 2 because MAC
            return self.SIMD[layer_idx] * self.PE[layer_idx] * self.MMV[layer_idx] * 2

    def calculate_matrix_cycles(self):
        layers = []
        for idx, layer in enumerate(self.net.layers):
            if lb.isMatrixLayer(layer) or lfpga.isFPGAMatrixLayer(layer):
                layers.append(self.net.ops_per_layer(layer) /
                              (self.ops_per_cycle(idx) * layer.get_parallel()))
            else:
                layers.append(0)
        return layers

    def fps(self):
        """FPS is cycles per second divided by worst case throughput of layer in cycles
           Device frequency is stored in Mhz, so we convert to Hz here"""
        self.nswg.calculate_neural_folding()
        self.nswg.calculate_write_block_cycles()
        self.nswg.calculate_read_block_cycles()
        self.nswg.calculate_total_cycles()
        max_mvtu = max(self.calculate_matrix_cycles())
        max_nswg = max(self.nswg.total_cycles)
        logging.info("FPS: %f " %
                      ((self.dev.frequency * 1e6) / max(max_nswg, max_mvtu)))
        return (self.dev.frequency * 1e6) / max(max_nswg, max_mvtu)

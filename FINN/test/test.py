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

import unittest
import FINN.core.nn as nn
import FINN.core.device as device
import FINN.core.perf_model as pm
from FINN.frontend.caffeloader import CaffeLoader
from FINN.frontend.excelloader import ExcelLoader

class TestSanity(unittest.TestCase):
	def test_cycles_per_op(self):
                l   = CaffeLoader("./FINN/inputs/sfc.caffemodel", "./FINN/inputs/sfc.prototxt")
		net = nn.NN(l)
		dev = device.Device('XLNX:VU9P.json')
		perfmodel = pm.PerfModel(net, dev)
                ops = perfmodel.network_utilisation()
		num_matrix_layers = net.count_matrix_layers()
                self.assertEqual(ops['luts'], 2* num_matrix_layers * dev.lut_cost_per_op())

	def test_cycles_per_layer(self):
		l   = CaffeLoader(None, "./FINN/inputs/dorefanet-pruned-without-extra-messages.prototxt")
                net = nn.NN(l)
		dev = device.Device('XLNX:KU115.json')
		perfmodel = pm.PerfModel(net, dev)
		fps = perfmodel.maximise_fps()
		for idx, layer in enumerate(net.layers):
                    in_chans = net.layers[idx].getInputSize()
                    out_chans = net.layers[idx].getOutputSize()
                    out_dim = net.layers[idx].get_out_dim()

                    if isinstance(in_chans, tuple):
                        print in_chans
                        in_chans = in_chans[0]
                    if isinstance(out_chans, tuple):
                        print out_chans
                        out_chans = out_chans[0]
                    if isinstance(out_dim, tuple):
                        print out_dim
                        out_dim = out_dim[0]

		    print perfmodel.SIMD[idx], in_chans
                    print perfmodel.PE[idx], out_chans
                    print perfmodel.MMV[idx], out_dim
                    self.assertLessEqual(perfmodel.SIMD[idx], in_chans)
		    self.assertLessEqual(perfmodel.PE[idx], out_chans)
		    self.assertLessEqual(perfmodel.MMV[idx], out_dim)

	def test_simd_pe_mmv_constraints(self):
		l   = CaffeLoader(None, "./FINN/inputs/sfc.prototxt")
                net = nn.NN(l)
		dev = device.Device('XLNX:KU115.json')
		perfmodel = pm.PerfModel(net, dev)
		fps = perfmodel.maximise_fps()
		for idx, layer in enumerate(net.layers):
                        self.assertLessEqual(perfmodel.SIMD[idx], layer.getInputSize())
			self.assertLessEqual(perfmodel.PE[idx], layer.getOutputSize())
			self.assertLessEqual(perfmodel.MMV[idx], layer.get_out_dim())


if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestSanity)
	unittest.TextTestRunner(verbosity=2).run(suite)

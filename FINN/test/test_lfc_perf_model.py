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
from FINN.core.config import FINN_ROOT
import FINN.core.nn as nn
import FINN.core.device as device
import FINN.core.perf_model as pm
from FINN.frontend.caffeloader import CaffeLoader
from FINN.core.coverification import testOnMNIST
import FINN.transforms.transformations as transform
import FINN.backend.fpga.backend_fpga as fpga_backend
import copy
import numpy as np
import sys
import tempfile
import shutil


class TestHWGQLFCw1a1(unittest.TestCase):
    """Test HWGQ network import and streamlining using a small binarized FC net."""

    def setUp(self):
        nname = "lfc-w1a1"
        proto = FINN_ROOT + "/inputs/%s.prototxt" % nname
        weights = FINN_ROOT + "/inputs/%s.caffemodel" % nname
        l = CaffeLoader(weights, proto)
        self.net = nn.NN(l)
        frequency = 300
        self.dev = device.Device('XLNX:VU9P.json', frequency)
        self.streamlined_net = copy.deepcopy(self.net)
        print self.streamlined_net.layers
        self.streamlined_net.layers = transform.makeCromulent(
            self.streamlined_net.layers)
        print self.streamlined_net.layers
        # use the first numImagesToTest of the test set for verification
        self.numImagesToTest = 1000
        # expected number of successful predictions
        self.ok_golden = 967
        # expected number of unsuccessful predictions
        self.nok_golden = 33
	

    def test_fpgabackend_rawhls(self):
        # resource allocation function to set number of PE/SIMD per layer
        # the allocation is statically determined for this test case.
        def res_alloc_predetermined(pipeline, net, dev):
            ret_pipeline = copy.deepcopy(pipeline)
            print "PIPELINE: ", ret_pipeline
            net.layers = ret_pipeline
            perfmodel = pm.PerfModel(net, dev)
            fps = perfmodel.maximise_fps()
            for i in range(len(ret_pipeline)):
                ret_pipeline[i].simd = perfmodel.SIMD[i]
                print "SIMD:", ret_pipeline[i].simd
                ret_pipeline[i].pe = perfmodel.PE[i]
                print "PE:", ret_pipeline[i].pe
                ret_pipeline[i].mmv = perfmodel.MMV[i]
                print "MMV:", ret_pipeline[i].mmv
            return ret_pipeline
        dirpath = tempfile.mkdtemp()
        # pick all layers except first (input quantization) and last
        # (final batchnorm) of the streamlined network
        hlslayers = self.streamlined_net.layers[1:-1]
        # call the FPGA backend to generate HLS and compile raw HLS sim
        print "Synthesising"
        ret = fpga_backend.synthesize(
            hlslayers, self.net, self.dev, res_alloc_predetermined, dirpath, "sfcall-")
        print "Synthesised"
        print ret.getFPGAPerformanceModel().SIMD
        print ret.getFPGAPerformanceModel().PE
        print ret.getFPGAPerformanceModel().MMV
        print ret.ir
        print ret.getFPGAPerformanceModel().network_utilisation()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHWGQLFCw1a1)
    unittest.TextTestRunner(verbosity=2).run(suite)


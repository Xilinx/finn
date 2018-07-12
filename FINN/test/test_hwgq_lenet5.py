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
from FINN.frontend.caffeloader import CaffeLoader
from FINN.core.datasets import loadMNIST
from FINN.core.coverification import testOnMNIST
import FINN.transforms.transformations as transform
import FINN.backend.fpga.backend_fpga as be_fpga
import FINN.core.layers as lb
import copy
import numpy as np
import sys
import tempfile
import shutil

class TestHWGQLeNet5(unittest.TestCase):
    """Test HWGQ network import and streamlining using a small convnet."""
    def setUp(self):
        nname = "lenet-hwgq-w1a2"
        proto =  FINN_ROOT +"/inputs/%s.prototxt" % nname
        weights = FINN_ROOT + "/inputs/%s.caffemodel" % nname
        l = CaffeLoader(weights, proto)
        self.net = nn.NN(l)
        self.streamlined_net = nn.NN(layers=transform.makeCromulent(self.net.layers))
        # use the first numImagesToTest of the test set for verification
        self.numImagesToTest = 10
        # expected number of successful predictions
        self.ok_golden = 10
        # expected number of unsuccessful predictions
        self.nok_golden = 0

    def test_0_dataflow_1convlayer(self):
        net = self.streamlined_net.layers
        # make a temp dir for generated HLS
        dirpath = tempfile.mkdtemp()
        # subset of layers for dataflow synthesis -- 1 quantized conv, 1 threshold
        hlslayers=net[3:5]
        def myresalloc(pipeline):
            ret = copy.deepcopy(pipeline)
            # weights matrix is (ofm=50) x (k=5 * k=5 * ifm=20)
            # set simd = 20 for faster sim
            ret[0].simd = 20
            return ret
        ret = be_fpga.synthesize(hlslayers, myresalloc, dirpath)
        hlspipeline = ret.getSimLayer()
        # set up mixed pipeline
        preproc = net[:3]
        postproc = net[5:]
        mixed_net = nn.NN(layers=preproc + hlspipeline + postproc)
        (ok, nok) = testOnMNIST(mixed_net, self.numImagesToTest)
        pm = ret.getFPGAPerformanceModel()
        cost = pm.network_utilisation()
        # remove temp dir
        shutil.rmtree(dirpath)
        # check result correctness
        self.assertTrue(ok == self.ok_golden and nok == self.nok_golden)
        # check BRAM and LUT usage from performance model
        self.assertTrue(cost['luts'] == 164)
        self.assertTrue(cost['brams'] == 16)

if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestHWGQLeNet5)
	unittest.TextTestRunner(verbosity=2).run(suite)

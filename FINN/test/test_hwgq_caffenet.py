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
import FINN.core.config as config
import FINN.core.nn as nn
from FINN.frontend.caffeloader import CaffeLoader
from FINN.core.datasets import getImageNetVal_1ksubset_LMDB
from FINN.core.coverification import testOnImageNet1kSubset
import FINN.transforms.transformations as transform
import FINN.backend.caffe.backend_caffe as cpu_backend
import copy
import numpy as np
import sys
import os.path
import caffe
import logging
import tempfile
import shutil

import sys
if sys.version_info[0] == 2:
    from urllib import urlretrieve
    from urllib import getproxies
else:
    from urllib.request import urlretrieve

class TestHWGQCaffeNet(unittest.TestCase):
    """Test HWGQ network import and streamlining using w1a2 HWGQ CaffeNet."""
    def setUp(self):
        self.nname = "caffenet-hwgq-w1a2"
        proto =  config.FINN_ROOT +"/inputs/%s.prototxt" % self.nname
        weights = config.FINN_ROOT + "/inputs/%s.caffemodel" % self.nname
        # download weights if not already on disk
        weights_url = "http://www.svcl.ucsd.edu/projects/hwgq/AlexNet_HWGQ.caffemodel"
        if not os.path.exists(weights):
            print("Downloading HWGQ CaffeNet weights")
            urlretrieve(weights_url, weights)
        l = CaffeLoader(weights, proto)
        self.net = nn.NN(l)
        # use the first numImagesToTest of the test set for verification
        self.numImagesToTest = 10
        # expected number of successful predictions
        self.ok_golden = 7
        # expected number of unsuccessful predictions
        self.nok_golden = 3

    def test_import_correctness(self):
        (ok, nok) = testOnImageNet1kSubset(self.net, self.numImagesToTest)
        self.assertTrue(ok == self.ok_golden and nok == self.nok_golden)

    def test_streamline_correctness(self):
        streamlined_net = copy.deepcopy(self.net)
        streamlined_net.layers = transform.makeCromulent(streamlined_net.layers)
        (ok, nok) = testOnImageNet1kSubset(streamlined_net, self.numImagesToTest)
        self.assertTrue(ok == self.ok_golden and nok == self.nok_golden)

    def test_quantize_first_layer(self):
        # quantize first convolutional (float) layer to use 8-bit weights
        qnt_layers = transform.directlyQuantizeLayer(self.net.layers[0], 8)
        qnt_net = nn.NN(layers=transform.makeCromulent(qnt_layers+self.net.layers[1:]))
        (ok, nok) = testOnImageNet1kSubset(qnt_net, self.numImagesToTest)
        self.assertTrue(ok == self.ok_golden and nok == self.nok_golden)

    def test_0_quantize_all_float_layers(self):
        qnt_net = transform.directlyQuantizeAllFloatWeights(self.net.layers, 8)
        qnt_net = nn.NN(layers=transform.makeCromulent(qnt_net))
        (ok, nok) = testOnImageNet1kSubset(qnt_net, self.numImagesToTest)
        self.assertTrue(ok == self.ok_golden and nok == self.nok_golden)

    def test_num_matrix_layers(self):
        self.assertIs(8, self.net.count_matrix_layers())

    @unittest.skipUnless("IntegerConvolution" in caffe.layer_type_list(), "requires Caffe with bit-serial layer support")
    def test_cpu_backend(self):
        # build streamlined version
        streamlined_net = copy.deepcopy(self.net)
        streamlined_net.layers = transform.makeCromulent(streamlined_net.layers)
        # set up data layer
        lmdb_path=getImageNetVal_1ksubset_LMDB()
        # set up data transformations for CaffeNet
        tp = dict(crop_size=227, mean_value=[104, 117, 123], mirror=False)
        # set up Caffe data layer
        datal = caffe.layers.Data(source=lmdb_path, backend=caffe.params.Data.LMDB, batch_size=1, ntop=2, transform_param=tp)
        datas = [1, 3, 227, 227]
        # make a temp dir for generated Caffe model and weights
        tmpdirpath = tempfile.mkdtemp()
        # call CPU bit serial backend for synthesis
        cpu_backend.synthesize(streamlined_net.layers, tmpdirpath, datas, datal, self.nname+"-")
        testproto_ok = os.path.isfile(tmpdirpath + "/" + self.nname + "-test.prototxt")
        deployproto_ok = os.path.isfile(tmpdirpath + "/" + self.nname + "-deploy.prototxt")
        weights_ok = os.path.isfile(tmpdirpath + "/" + self.nname + "-weights.caffemodel")
        # remove temp dir
        shutil.rmtree(tmpdirpath)
        # TODO can test the returned net to ensure correctness
        self.assertTrue(testproto_ok and deployproto_ok and weights_ok)

    # TODO add correctness tests once we decide on how to do this for ImageNet

if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestHWGQCaffeNet)
	unittest.TextTestRunner(verbosity=2).run(suite)

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


import sys
import os
import numpy as np
import FINN.core.config as config
import gzip
import tarfile

# utilities to download dataset files, when they are not found.

# stolen from Lasagne
if sys.version_info[0] == 2:
    from urllib import urlretrieve
    from urllib import getproxies
else:
    from urllib.request import urlretrieve

def download(filename, source=''):
    print("Downloading %s" % filename)
    print "PROXIES:"+ str(getproxies())
    urlretrieve(source + filename, config.DATA_DIR+filename)

def loadCIFAR10():
    url = 'https://www.cs.toronto.edu/~kriz/'
    filename = 'cifar-10-binary.tar.gz'
    if not os.path.exists(config.DATA_DIR+filename):
        download(filename, url)
    gunzip(filename)

    # Currently don't return data, just download file
def gunzip(resource):
    print "Gunzipping " + config.DATA_DIR+resource
    with tarfile.open(config.DATA_DIR+resource, 'r:gz') as f:
    #with open(config.DATA_DIR+(os.path.splitext(resource)[0]), 'w') as n :
        for tar in f:
            print tar.name

def getImageNetVal_1ksubset_LMDB():
    """Get path to an LMDB with 1k images from the ImageNet validation database,
    downloading it if necessary"""
    if not os.path.exists(config.DATA_DIR+"ilsvrc12_val_tiny_subset_lmdb"):
        import zipfile
        download("ilsvrc12_val_1ksubset_lmdb.zip", "http://www.idi.ntnu.no/~yamanu/")
        zip_ref = zipfile.ZipFile(config.DATA_DIR+"ilsvrc12_val_1ksubset_lmdb.zip", 'r')
        zip_ref.extractall(config.DATA_DIR)
        zip_ref.close()
        os.remove(config.DATA_DIR+"ilsvrc12_val_1ksubset_lmdb.zip")
    return config.DATA_DIR+"ilsvrc12_val_tiny_subset_lmdb"

def loadMNIST():

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    url = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        if not os.path.exists(config.DATA_DIR+filename):
            download(filename, url)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(config.DATA_DIR+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data

    def load_mnist_labels(filename):
        if not os.path.exists(config.DATA_DIR+filename):
            download(filename, url)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(config.DATA_DIR+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the test set images and labels.
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_test, y_test

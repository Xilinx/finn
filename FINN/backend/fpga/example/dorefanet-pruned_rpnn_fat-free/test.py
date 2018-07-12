
# 
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
#

import numpy as np
import rpnn as nn
import json


import random
import time
import sys

from os import listdir
from os.path import isfile, join

import cv2
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

import tensorflow as tf

from subprocess import call

### DoReFa-Net stuff ###

# Quantize input number
def quantize(x, k):
    n = float(2**k-1)
    return tf.round(x * n) / n

# Binarize convolutional layer weights
# W in input contains floating points weights in the range [-1, 1] or [-E, E] where E is the mean of weights
# 	[-E, E] if W is quantized using 1 bit
#	[-1, 1] if W is quantized using more than 1 bits
# TODO: This function is currently useful only for 1 bit weights
# 		In case of multiple bits npW output equals W * (2**bits - 1) input and npE must equal to 1
def fw(W): 
	E = tf.reduce_mean(tf.abs(W))
	W = tf.sign(W/E)

	npE = E.eval(session=session)
	npW = W.eval(session=session)

	return [npW, npE]

def fw_Wbits(W,Wbits):
     E = tf.clip_by_value(W,-1.0, 1.0)
     W = W * 0.5 + 0.5
     W = 2 * quantize(W, Wbits) - 1 
     npW = W.eval(session=session)

     npE = E.eval(session=session)
     return [npW,npE] 

# Quantize x using bits bits
def fa(x, bits):
    if bits == 32:
        return x
    return quantize(x, bits)

# Apply non linear transformation to the input
def nonlin(x, bits):
    if bits == 32:
        return tf.nn.relu(x)    # still use relu for 32bit cases
    return tf.clip_by_value(x, 0.0, 1.0)

# Quantize Outputs
def activate(x, bits):
	return fa(nonlin(x, bits), bits)

####### End of DoReFa-Net stuff ########

# Return a 2d shape where each dimension is set to size a
def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))

# Return a 4d shape where first and last dimension are of size 1
def shape4d(a):
    # for use with tensorflow NHWC ops
    return [1] + shape2d(a)  + [1]

# Get layer json information from the network description
def getLayer(networkJson, layerName):
	for layer in networkJson:
		if layer['name'] == layerName:
			return layer

# Call C backend to configure the network parameters in the backend
def initNetwork(networkJson):
	for layer in networkJson:
		if layer['func'] == "conv_layer":
			nn.createConvolutionalLayer(str(layer['name'].encode('utf-8')), layer['kernel_shape'], layer['stride'], layer['split'], layer['basemem'], layer['SIMD'], layer['PE'], layer['WMEM'], layer['TMEM'], layer['input_bits'], layer['output_bits'], layer['weight_bits']);

		if layer['func'] == "fc_layer":
			nn.createConvolutionalLayer(str(layer['name'].encode('utf-8')), 1, 1, 1, layer['basemem'], layer['SIMD'], layer['PE'], layer['WMEM'], layer['TMEM'], layer['input_bits'], layer['output_bits'], layer['weight_bits']);

	
# Read and trasform input image.
# Borrowed from tensorpack
def loadImage(image):
	# Get image metadata from ILSVRC dataset
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]

    # Resize image to 224x224
    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),\
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    
    # Do other transformations
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    
    assert os.path.isfile(image)
    img = cv2.imread(image).astype('float32')
    assert img is not None

    img = transformers.augment(img)[np.newaxis, :,:,:]

    return img


def inference(imgList, target, simulation, imgPath):
	# # Load Images
	for imgFile in imgList:
		img = loadImage(imgPath + os.sep + imgFile)
		img = img / 255.0
		start = time.time()

		# Apply precomputation to the image
		# Trasform [0, 1] floating point output in an integer output
		# TODO: 2**2 should be 2**Precision depending on inputp recision
		imgPreprocessed = img * (2**7 -1)
		
		#imgPreprocessed = imgPreprocessed.astype(int)
		imgPreprocessed[imgPreprocessed<0] = (255+imgPreprocessed[imgPreprocessed<0])
		# Add image to the batch of image to process in HW
		nn.addImage(imgPreprocessed, 8, 512, simulation)

		end = time.time()

		print "Precompute: " + str(end-start)

	# Inference
	print "HW Inference"
	start = time.time()
	nn.inference(simulation)
	end = time.time()

	print "Convolution: " + str(end-start)

	for imgFile in imgList:

		# Prepare output buffer
		networkOutput = np.zeros(shape=(1, 6*6*256), dtype=np.float32, order='C')

		# Fetch output data
		nn.fetchResult(networkOutput, 2, simulation)

		# Transform output from HW in the range [0, 2**2 - 1] and scales it to [0, 1]
		networkOutput = networkOutput / (2**2 - 1 )

		# Reshape the output to the right dimensions
		networkOutput = np.reshape(networkOutput, (1, 6, 6, 256), order='C')

		start = time.time()

		# Convert numpy vector to tensor
		tfInput = tf.convert_to_tensor(networkOutput)

		[Wfc0, Wfc0_mean] = fw(networkModel['fc0/W'])
		Wfc0 = Wfc0 
		Xfc0 = batch_flatten(tfInput)
		tfFc0 = tf.identity(tf.matmul(Xfc0, Wfc0))
		# tfFc0 = tf.Print(tfFc0, [tfFc0])
		tfBnFc0 = tf.nn.batch_normalization(tfFc0, networkModel["bnfc0/mean/EMA"], networkModel["bnfc0/variance/EMA"], networkModel["bnfc0/beta"], networkModel["bnfc0/gamma"], 1e-4, 'bnfc0')
		tfActivateFc0 = activate(tfBnFc0, 2)

		# tfActivateFc0 = tf.Print(tfActivateFc0, [tfActivateFc0], summarize=100000)

		[Wfc1, Wfc1_mean] = fw(networkModel['fc1/W'])
		Wfc1 = Wfc1 
		Xfc1 = batch_flatten(tfActivateFc0)
		tfFc1 = tf.identity(tf.matmul(Xfc1, Wfc1))
		tfBnFc1 = tf.nn.batch_normalization(tfFc1, networkModel["bnfc1/mean/EMA"], networkModel["bnfc1/variance/EMA"], networkModel["bnfc1/beta"], networkModel["bnfc1/gamma"], 1e-4, 'bnfc1')
		tfActivateFc1 = nonlin(tfBnFc1, 2)

		[WfcT, WfcT_mean] = fw_Wbits(networkModel['fct/W'], 4)
		WfcT = networkModel['fct/W']
		XfcT = batch_flatten(tfActivateFc1)
		BfcT = networkModel['fct/b']
		tfFct = tf.identity(tf.nn.xw_plus_b(XfcT, WfcT, BfcT))

		end = time.time()
		print "Fully Connected: " + str(end-start)

		start = time.time()

		# Apply softmax
		tfProb = tf.nn.softmax(tfFct)
		prob = tfProb.eval(session=session)[0]

		# Load metadata from ILSVRC dataset
		meta = dataset.ILSVRCMeta()
		words = meta.get_synset_words_1000()

		# Get first 10 probabilities
		ret = prob.argsort()[-10:][::-1]

		# Get name for first 10 probabilities
		names = [words[i] for i in ret]

		# Get classes from names
		classes = [name.split(" ")[0] for name in names]

		# Correct input image class (derived by file name)
		imgClass = imgFile.split("_")[0].split("/")[-1]
		
		# Compute accuracy
		top1 = imgClass in classes[0:1]
		top3 = imgClass in classes[0:3]
		top5 = imgClass in classes[0:5]
		top10 = imgClass in classes
		
		# Output results
		result = ["ACCURACY", target, imgClass, imgFile.split("/")[-1], str(top1), str(top3), str(top5), str(top10), str(prob[ret[0]])]
		print "\t".join(result)

		end = time.time()

		print "Scoring: " + str(end-start)

# Setup tensorflow
tf.reset_default_graph()
device_name = "/cpu:0"
config = tf.ConfigProto(
    device_count = {'GPU': 0}
)
tf.device(device_name)

# Network layer to batch normalization layer, only for layer with binary activation
layer2bn = dict()
layer2bn['conv0'] = None
layer2bn['conv1'] = "bn1"
layer2bn['conv2'] = "bn2"
layer2bn['conv3'] = "bn3"
layer2bn['conv4'] = "bn4"
layer2bn['fc0'] = "bnfc0"
layer2bn['fc1'] = "bnfc1"
# layer2bn['fct'] = None

if len(sys.argv)!=6:
	print "USAGE: python " + sys.argv[0] + " <mode> <network.net> <imgPath> <numImages> <Network npy> "
	print "where <mode> = 0: HW, 1:HLS Simulation"
	print "where <network.net> = Network Json"
	print "where <imgPath> = Path for test images"
	print "where <numImages> = Number of images to test"
	print "where <Network npy> = Npy of network"
	sys.exit(0)

# Load input parameters
mode = int(sys.argv[1])
file = sys.argv[2]
imgPath = sys.argv[3]
numImages = int(sys.argv[4])

# Load input images
imgList = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
imgList = imgList[0:numImages]
# Load input parameters
json_data=open(file).read()
networkJson = json.loads(json_data)
networkModel = np.load(sys.argv[5], encoding="latin1").item()

with tf.device('/cpu:0'):
	session = tf.Session(config=config)

	# HW Configure Network
	nn.initAccelerator(mode)
	initNetwork(networkJson)

	# Allocate data structures in C backend and configure it depending on actual network
	nn.mallocBuffers(224*224*3, 8, 6*6*256, 2, len(imgList), 256)

	# Perform inference
	inference(imgList, "HW", mode, imgPath)

	# Free backend data structures
	nn.freeBuffers()

	session.close()

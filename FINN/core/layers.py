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


import numpy as np
import math
from im2col import im2col_indices
from abc import ABCMeta, abstractproperty

class Layer:
    __metaclass__ = ABCMeta

    @abstractproperty
    def __init__(self):
        pass

    @abstractproperty
    def execute(self):
        pass

  #  @abstractproperty
  #  def updateBitwidths(self, inBitWidth):
  ##      pass

    def get_type(self):
        return self.__class__.__name__

    def get_stride(self):
        if hasattr(self, 'stride'):
            return self.stride
        return 1

    def get_in_dim(self):
        if hasattr(self, 'padded_idim'):
            return self.padded_idim
        elif hasattr(self, 'in_dim'):
            return self.in_dim
        elif hasattr(self, 'idim'):
            return self.idim
        return -1

    def get_pad(self):
        if hasattr(self, 'pad'):
            return self.pad
        return 0

    def getOutputSize(self):
        return (self.outsize)

    def get_filter_dim(self):
        if hasattr(self, 'kernel'):
            return self.kernel
        return 0

    def getInputSize(self):
        return (self.insize)

    def get_parallel(self):
        if hasattr(self, 'parallel'):
          return self.parallel
        return 1

    def get_out_dim(self):
        #print "using baseclass def: ", self.get_in_dim(), self.get_filter_dim(), self.get_stride()
        return int(math.floor(float(self.get_in_dim() - self.get_filter_dim())/self.get_stride()+1)) # Why was it this? math.ceil(self.get_in_dim() + ( 2 * self.get_pad() - self.get_filter_dim() + 1/self.get_stride()))

    def __repr__(self):
        return self.__class__.__name__

# device-independent FINN layer types

class DummyLayer(Layer):
    def __init__(self):
        self.ibits = 32
        self.obits = 32

    def execute(self):
        pass

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = self.ibits
        return self.obits

class ExternalExecutionLayer(Layer):
    "Call an executable to process the data. I/O is done via npy files."
    def __init__(self, execmd):
        import tempfile
        self.execmd = execmd
        self.ifilename = tempfile.mktemp() + ".npy"
        self.ofilename = tempfile.mktemp() + ".npy"

    def execute(self, v):
        from subprocess import check_call
        np.save(self.ifilename, v.astype(np.float32))
        check_call([self.execmd, self.ifilename, self.ofilename])
        return np.load(self.ofilename)

class MatrixThresholdLayer(Layer):
    "Fused layer type for a matrix operation followed by thresholding"
    def __init__(self, name, mlayer, tlayer):
        self.name = name
        self.mlayer = mlayer
        self.kernel = mlayer.kernel
        self.tlayer = tlayer
        self.ibits = self.mlayer.ibits
        self.wbits = self.mlayer.wbits
        self.obits = self.tlayer.obits
        self.outsize = self.mlayer.getOutputSize()
        self.insize = self.mlayer.getInputSize()
    
    def getNumOps(self):
        return self.mlayer.getNumOps() 

    def execute(self, v):
        return self.tlayer.execute(self.mlayer.execute(v))

    def updateBitwidths(self, inBitWidth):
        self.tlayer.updateBitwidths(self.mlayer.updateBitwidths(inBitWidth))
        self.ibits = self.mlayer.ibits
        self.obits = self.tlayer.obits
        return self.obits

class SoftmaxLayer(Layer):
    "Compute softmax values for each sets of scores."
    def __init__(self):
        self.ibits = 32
        self.obits = 32

    def execute(selv, v):
        e_x = np.exp(v - np.max(v))
        return e_x / e_x.sum()

    def updateBitwidths(self, inBitWidth):
        return self.obits

class ReLULayer(Layer):
    "Apply elementwise ReLU to the vector."
    def __init__(self):
        self.ibits = 32
        self.obits = 32

    def execute(self, v):
        return np.asarray(map(lambda x: x if x>0 else 0, v))

    def updateBitwidths(self, inBitWidth):
        # strictly speaking, ReLU can actually reduce the output
        # bitwidth since everything below 0 becomes a 0
        self.ibits = inBitWidth
        self.obits = self.ibits
        return self.obits

class LinearLayer(Layer):
    "Using vectors A and B, apply Ax+B to incoming x."
    def __init__(self, A, B):
        if A.shape != B.shape:
            raise Exception("LinearLayer A and B shapes do not match")
        self.A = A
        self.B = B
        self.ibits = 32
        self.wbits = 32
        self.obits = 32
        # TODO this is not always correct -- actual size must be propagated from
        # previous layer. the param shape used here can be much smaller than
        # the actual incoming image size (in which case the param is repeated/
        # broadcasted)
        self.insize = A.shape[0]
        self.outsize = A.shape[0]

    def execute(self, v):
        # the outermost dimension is the channel dimension
        # reshape as inner dimension to apply transform
        vr = v.reshape((self.A.shape[0], -1)).transpose()
        return (self.A*vr+self.B).transpose().flatten()

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        # for now, we assume the LinearLayer always performs 32-bit float math
        return self.obits

class ThresholdingLayer(Layer):
    "Given a set of thresholds, return the number of thresholds crossed."
    def __init__(self, thresholds):
        # we expect the thresholds array in the following format:
        # thresholds = [levels][channels]
        if thresholds.ndim == 1:
            self.thresholds = thresholds.reshape((len(thresholds),-1))
        elif thresholds.ndim == 2:
            self.thresholds = thresholds
        else:
            raise Exception("Thresholds array must be 1- or 2-dimensional")
        self.ibits = 32
        self.obits = int(math.ceil(math.log(self.thresholds.shape[0]+1, 2)))

    def execute(self, v):
        # interpret as multi-channel image, where the number of channels is
        # decided as the number of threshold channels
        vr = v.reshape((self.thresholds.shape[1], -1))
        ret = np.zeros(vr.shape, dtype=np.int)
        for t in self.thresholds:
            for c in range(self.thresholds.shape[1]):
                ret[c] += map(lambda x: 1 if x == True else 0, vr[c] >= t[c])
        return ret.flatten()

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        # output bit width stays unchanged for ThresholdingLayer
        return self.obits

class BipolarThresholdingLayer(ThresholdingLayer):
    "A 1-level ThresholdingLayer that returns -1 and +1 instead of 0 and 1."
    def __init__(self, thresholds):
        super(BipolarThresholdingLayer, self).__init__(thresholds)
        if self.thresholds.shape[0] != 1:
            raise Exception("BipolarThresholdingLayer can only have one level")

    def execute(self, v):
        # just the base implementation, but scaled by 2x-1 such that the output
        # is -1, +1 instead of 0, 1. this could have been done with a following
        # LinearLayer, but that LinearLayer may disappear as a result of
        # streamlining. we have an interest in keeping the bipolar thresholding
        # intact since there are special XNOR primitives for it.
        ret = super(BipolarThresholdingLayer, self).execute(v)
        return 2*ret - 1

# TODO add a LookupTableLayer for nonlinear quantization support
class FullyConnectedLayer(Layer):
    """
    A layer that implements fully-connected network layers.
    Note that bias is not implemented, this can be done by adding a LinearLayer
    following the FullyConnectedLayer.
    """
    def __init__(self, W, wbits, ibits, obits):
        self.kernel = 1
        self.wbits = wbits
        self.ibits = ibits
        self.obits = obits
        self.W = W
        self.outsize = W.shape[0]
        self.insize = W.shape[1]

    def execute(self, v):
        return np.dot(self.W, v)

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        if self.ibits == 32 or self.wbits == 32:
            # produce float outputs for float inputs (since 32bits means
            # float at the moment)
            self.obits = 32
        else:
            # find the number of bits necessary to represent the largest possible
            # sum for a result element. assume maximum valued weight and input:
            maxWVal = (1 << self.wbits) - 1
            maxIVal = (1 << self.ibits) - 1
            # assume every single input is maximum:
            self.obits = int(self.insize*maxWVal*maxIVal).bit_length()
        return self.obits

    def getParamSize(self):
        return self.W.size

    def getNumOps(self):
        return self.W.size * 2

    def getInputSize(self):
        """in_channels"""
        return (self.insize)

    def getOutputSize(self):
        return (self.outsize)

    def getTotalParamBits(self):
        return self.wbits * self.getParamSize()

    def getTotalInputBits(self):
        return self.ibits * np.prod(self.getInputSize())

    def getTotalOutputBits(self):
        return self.obits * np.prod(self.getOutputSize())

class ChanInterleaveLayer(Layer):
    """
    Interleaves multichannel image data passing though the layer. For instance,
    a typical RGB image may be normally laid out as three single-channel
    images (R, G, B) such that we have img[chan][row][col]. After passing
    through this layer, it will be converted to a single image of RGB pixels,
    such that it is laid out as img[row][col][chan].
    """
    def __init__(self, inDim, inChans):
        self.dim = inDim
        self.chans = inChans
        self.ibits = 32
        self.obits = 32

    def execute(self, v):
        # first, convert the incoming flattened vector into a multidim array
        img = v.reshape((self.chans, self.dim, self.dim))
        # tranpose axes, flatten and return
        return img.transpose((1, 2, 0)).flatten()

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = self.ibits
        return self.obits

class ChanDeinterleaveLayer(Layer):
    "Does the inverse of ChanInterleaveLayer, see explanation there."
    def __init__(self, inDim, inChans):
        self.dim = inDim
        self.chans = inChans

    def execute(self, v):
        # first, convert the incoming flattened vector into a multidim array
        img = v.reshape((self.dim, self.dim, self.chans))
        # tranpose axes, flatten and return
        return img.transpose((2, 0, 1)).flatten()

class PaddingLayer(Layer):
    "A layer that adds padding around the edges of the image."
    def __init__(self, inDim, inChans, padCount, padVal):
        self.dim = inDim
        self.chans = inChans
        self.padCount = padCount
        self.padVal = padVal

    def execute(self, v):
        img = v.reshape((self.chans, self.dim, self.dim))
        padCounts = ((0, 0),
                     (self.padCount, self.padCount),
                     (self.padCount, self.padCount))
        img = np.pad(img, padCounts, "constant", constant_values=self.padVal)
        return img.flatten()

class SlidingWindowLayer(Layer):
    "Slide a window over a multichannel image (im2col)"
    def __init__(self, inDim, inChans, windowDim, stride=1):
        self.idim = inDim
        self.chans = inChans
        self.k = windowDim
        self.s = stride

    def execute(self, v):
        # reshape the input vector into a 2D image
        img = v.reshape((1, self.chans, self.idim, self.idim))
        # call im2col to get the sliding window result
        res = im2col_indices(img, self.k, self.k, padding=0,
                            stride_y=self.s, stride_x=self.s)
        return res.flatten()


class ConvolutionLayer(Layer):
    "Convolution via im2col and matrix-matrix multiplication"
    def __init__(self, W, inDim, pad, stride, wbits, ibits, obits, padVal=0):
        self.wbits = wbits
        self.ibits = ibits
        self.obits = obits
        self.ofm = W.shape[0]
        self.ifm = W.shape[1]
        self.kernel = W.shape[2]
        self.idim = inDim
        self.padded_idim = inDim + 2*pad
        self.odim =int(math.floor((float(self.padded_idim - self.kernel) / stride)  +1))
        self.in_dim = inDim
        self.out_dim = self.odim
        self.stride = stride
        self.pad = pad
        self.padVal = padVal
        if(W.shape[2] != W.shape[3]):
            raise Exception("Only square conv filters supported for now")
        # instantiate internal layer components
        self.layers = []
        if pad != 0:
            self.layers += [PaddingLayer(self.idim, self.ifm, pad, padVal)]
        self.layers += [SlidingWindowLayer(self.padded_idim, self.ifm, self.kernel, self.stride)]
        self.W = W.reshape((self.ofm, self.ifm*self.kernel*self.kernel))
        self.outsize = self.ofm * self.odim * self.odim

    def execute(self, v):
        # execute internal padding/sliding window layers first
        vn = v
        for l in self.layers:
            vn = l.execute(vn)
        # reconstruct image matrix
        vn = vn.reshape((self.ifm*self.kernel*self.kernel, self.odim*self.odim))
        # matrix-matrix multiply
        res = np.dot(self.W, vn)
        return res.flatten()

    def get_filter_dim(self):
        return self.kernel

    def get_in_dim(self):
        return self.padded_idim
    
    def getParamSize(self):
        return self.W.size

    def getNumOps(self):
        if hasattr(self, 'parallel'):
            return self.W.size * self.odim * self.odim * 2 * self.parallel 
        return self.W.size * self.odim * self.odim * 2

    def getInputSize(self):
        return (self.ifm, self.idim, self.idim)[0]

    def getOutputSize(self):
        return (self.ofm, self.odim, self.odim)[0]

    def getTotalParamBits(self):
        return self.wbits * self.getParamSize()

    def getTotalInputBits(self):
        return self.ibits * np.prod(self.getInputSize())

    def getTotalOutputBits(self):
        return self.obits * np.prod(self.getOutputSize())

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        if self.ibits == 32 or self.wbits == 32:
            # produce float outputs for float inputs (since 32bits means
            # float at the moment)
            self.obits = 32
        else:
            # find the number of bits necessary to represent the largest possible
            # sum for a result element. assume maximum valued weight and input:
            maxWVal = (1 << self.wbits) - 1
            maxIVal = (1 << self.ibits) - 1
            # assume every single input is maximum:
            self.obits = int(self.W.shape[1]*maxWVal*maxIVal).bit_length()
        return self.obits

class PoolingLayer(Layer):
    "Perform pooling"
    def __init__(self, inDim, inChans, poolSize, strideSize, poolFxn = "max"):
        self.ibits = 32
        self.obits = 32
        self.idim = inDim
        self.chans = inChans
        self.k = poolSize
        self.s = strideSize
        self.odim = math.ceil((float(self.idim - self.k) / float(self.s))+1)
        self.poolFxn = poolFxn
        self.outsize = (self.chans, self.odim, self.odim)[0]
        self.insize = self.idim * self.idim * self.chans

    def execute(self, v):
        img = v.reshape((self.chans, self.idim, self.idim))
        out_img = np.zeros((self.chans, self.odim*self.odim), dtype=np.float32)
        for c in range(self.chans):
            chan_img = img[c].reshape((1, 1, self.idim, self.idim))
            # extract parts of image with sliding window
            wnd = im2col_indices(chan_img, self.k, self.k, padding=0,
            stride_y=self.s, stride_x=self.s)
            # each window is a column -- get the reduction along columns
            if self.poolFxn == "MAX":
                out_img[c]=wnd.max(axis = 0).flatten()
            elif self.poolFxn == "AVE":
                out_img[c]=wnd.mean(axis = 0).flatten()
            else:
                raise Exception("Unsupported pooling function")
        return out_img.flatten()

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = self.ibits
        return self.obits

class MonitorLayer(Layer):
    "A layer that prints the numpy array data passing through."
    def __init__(self, tag):
        self.tag = tag
        self.i = 0

    def execute(self, v):
        print("\n\nMonitorLayer %s at execution %d:" % (self.tag, self.i))
        o = np.get_printoptions()
        np.set_printoptions(threshold=np.nan)
        print(np.array_repr(v))
        np.set_printoptions(**o)
        self.i += 1
        return v

def isLinearLayer(layer):
    lname = layer.__class__.__name__
    return (lname == "LinearLayer")

def isScalarLinearLayer(layer):
    if isLinearLayer(layer):
        return layer.A.shape == (1,)
    else:
        return False

def isMatrixLayer(layer):
    lname = layer.__class__.__name__
    return lname == "FullyConnectedLayer" or lname == "ConvolutionLayer"

def isThresholdLayer(layer):
    return isinstance(layer, ThresholdingLayer)

def isMatrixThresholdLayer(layer):
    return isinstance(layer, MatrixThresholdLayer)

def isPoolingLayer(layer):
    return isinstance(layer, PoolingLayer)

def isMaxPoolingLayer(layer):
    if isPoolingLayer(layer):
        return layer.poolFxn == "MAX"
    else:
        return False

def isFCLayer(layer):
    return isinstance(layer, FullyConnectedLayer)

def isConvLayer(layer):
    return isinstance(layer, ConvolutionLayer)

def isReLULayer(layer):
    return isinstance(layer, ReLULayer)

def isSoftmaxLayer(layer):
    return isinstance(layer, SoftmaxLayer)

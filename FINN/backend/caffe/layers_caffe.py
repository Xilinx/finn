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
import FINN.core.layers as layers_base
from caffe import layers as L
import caffe

# Layer types specific to the Caffe backend

def isCaffeLayer(layer):
    lname = layer.__class__.__name__
    return lname.startswith("Caffe")

def isCaffeIntMatrixLayer(layer):
    return isinstance(layer, CaffeIntegerConvolutionLayer) or isinstance(layer, CaffeIntegerInnerProductLayer)

def isCaffeMultiThresholdLayer(layer):
    return isinstance(layer, CaffeMultiThresholdLayer)

class CaffeMLBPOffloadLayer(layers_base.Layer):
    """
    MLBP offload layer (on FPGA).
    """
    def __init__(self, name, ibits, obits, inshape, outshape, do_intl, do_deintl, bitfile_load_cmd):
        self.ibits = ibits
        self.obits = obits
        self.inshape = inshape
        self.outshape = outshape
        self.bitfile_load_cmd = bitfile_load_cmd
        self.interleave_input = do_intl
        self.deinterleave_output = do_deintl

    def paramfill(self, net):
        # no parameters
        next

    def codegen(self, model):
        mlbp_params = dict(
            input_shape = self.inshape,
            output_shape = self.outshape,
            interleave_input = self.interleave_input,
            deinterleave_output = self.deinterleave_output,
            bitfile_load_cmd = self.bitfile_load_cmd
        )
        # TODO add more options:
        # use_8bit_input
        # use_8bit_output
        return L.MLBPOffload(model, mlbp_offload_param=mlbp_params)

    def execute(self, v):
        raise Exception("Not supported for simulation")

    def updateBitwidths(self, inBitWidth):
        # do nothing
        return self.obits

class CaffeReLULayer(layers_base.ReLULayer):
    """
    Caffe ReLU layer.
    """
    def __init__(self, name):
        layers_base.ReLULayer.__init__(self)
        self.name = name

    def paramfill(self, net):
        # ReLU does not have any parameters
        next

    def codegen(self, model):
        return L.ReLU(model)

class CaffeSoftmaxLayer(layers_base.SoftmaxLayer):
    """
    Caffe Softmax layer.
    """
    def __init__(self, name):
        layers_base.SoftmaxLayer.__init__(self)
        self.name = name

    def paramfill(self, net):
        # ReLU does not have any parameters
        next

    def codegen(self, model):
        return L.Softmax(model)


class CaffePoolLayer(layers_base.PoolingLayer):
    """
    Caffe pool layer.
    """
    def __init__(self, name, PLayer):
        layers_base.PoolingLayer.__init__(self, PLayer.idim, PLayer.chans, PLayer.k, PLayer.s, PLayer.poolFxn)
        self.name = name

    def paramfill(self, net):
        # pooling does not have any parameters
        next

    def codegen(self, model):
        str_to_caffepmode = {"MAX" : caffe.params.Pooling.MAX, "AVE" : caffe.params.Pooling.AVE}
        return L.Pooling(model, name=self.name, pool=str_to_caffepmode[self.poolFxn], kernel_size=self.k, stride=self.s)

class CaffeIntegerConvolutionLayer(layers_base.ConvolutionLayer):
    """
    Caffe integer convolution layer.
    """
    def __init__(self, name, CLayer, qnnengine):
        # restore weights to the 4d format expected by Caffe
        Wp = CLayer.W.reshape((CLayer.ofm, CLayer.ifm, CLayer.k, CLayer.k))
        layers_base.ConvolutionLayer.__init__(self, Wp, CLayer.idim, CLayer.pad, CLayer.stride, CLayer.wbits, CLayer.ibits, CLayer.obits, CLayer.padVal)
        if self.wbits > 8:
            raise Exception("Unsupported bitwidth for IntConv layer")
        self.name = name
        # TODO do range analysis to set this intelligently
        self.wsigned = True
        self.isigned = False # this needs to be propagated!
        self.use_byte_input = False
        self.engine = qnnengine

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.W.reshape((self.ofm, self.ifm, self.k, self.k))

    def codegen(self, model):
        icp=dict(
            use_byte_input=self.use_byte_input, num_output=self.ofm,
            wbits=self.wbits, ibits=self.ibits, wsigned=self.wsigned,
            isigned=self.isigned, pad=self.pad, kernel_size=self.k,
            stride=self.stride, engine=self.engine
            )
        return L.IntegerConvolution(model, name=self.name, integer_convolution_param=icp)

class CaffeConvolutionLayer(layers_base.ConvolutionLayer):
    """
    Caffe regular convolution layer.
    """
    def __init__(self, name, CLayer):
        # restore weights to the 4d format expected by Caffe
        Wp = CLayer.W.reshape((CLayer.ofm, CLayer.ifm, CLayer.k, CLayer.k))
        layers_base.ConvolutionLayer.__init__(self, Wp, CLayer.idim, CLayer.pad, CLayer.stride, CLayer.wbits, CLayer.ibits, CLayer.obits, CLayer.padVal)
        self.name = name

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.W.reshape((self.ofm, self.ifm, self.k, self.k))

    def codegen(self, model):
        cp=dict(num_output=self.ofm, pad=self.pad, kernel_size=self.k, stride=self.stride, bias_term=False)
        return L.Convolution(model, name=self.name, convolution_param=cp)


class CaffeIntegerInnerProductLayer:
    """
    Caffe integer inner product layer.
    """
    def __init__(self, name, FCLayer, qnnengine):
        if FCLayer.wbits > 8:
            raise Exception("Unsupported bitwidth!")
        self.wbits = FCLayer.wbits
        self.ibits = FCLayer.ibits
        self.obits = 32 # always using 32-bit accumulators
        self.W = np.asarray(FCLayer.W, dtype=np.float32)
        self.name = name
        # TODO do range analysis to set these intelligently
        self.isigned = False
        self.wsigned = True
        self.rows = self.W.shape[0]
        self.cols = self.W.shape[1]
        self.use_byte_input = False
        self.engine = qnnengine

    def execute(self, v):
        raise Exception("TODO: not yet implemented")

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.W

    def codegen(self, model):
        iipp=dict(
            use_byte_input=self.use_byte_input, num_output=self.rows,
            wbits=self.wbits, ibits=self.ibits, wsigned=self.wsigned,
            isigned=self.isigned, engine=self.engine
        )
        return L.IntegerInnerProduct(model, name=self.name, integer_inner_product_param=iipp)

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = 32 # always using 32-bit accumulators
        return self.obits

class CaffeInnerProductLayer:
    """
    Caffe float inner product layer.
    """
    def __init__(self, name, FCLayer):
        self.wbits = 32
        self.ibits = 32
        self.obits = 32 # always using 32-bit accumulators
        self.W = np.asarray(FCLayer.W, dtype=np.float32)
        self.name = name
        self.rows = self.W.shape[0]
        self.cols = self.W.shape[1]

    def execute(self, v):
        raise Exception("TODO: not yet implemented")

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.W

    def codegen(self, model):
        return L.InnerProduct(model, name=self.name, inner_product_param=dict(num_output=self.rows))

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = 32 # always using 32-bit (float) math
        return self.obits

class CaffeMultiThresholdLayer:
    """
    Caffe multi-thresholding layer.
    """
    def __init__(self, name, TLayer):
        if TLayer.obits > 8:
            raise Exception("Unsupported bitwidth!")
        self.ibits = TLayer.ibits
        self.obits = TLayer.obits
        self.T = TLayer.thresholds.astype(np.float32)
        self.num_thres = self.T.shape[0]
        self.num_channels = self.T.shape[1]
        self.name = name
        self.use_byte_output = False

    def execute(self, v):
        raise Exception("TODO: not yet implemented")

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.T

    def codegen(self, model):
        return L.MultiThreshold(model, name=self.name, multi_threshold_param=dict(use_byte_output=self.use_byte_output, num_thres=self.num_thres, num_channels=self.num_channels))

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        # output bit width stays unchanged for ThresholdingLayer
        return self.obits

class CaffeScaleLayer:
    """Caffe scale layer. """
    def __init__(self, name,  LLayer):
        self.ibits = LLayer.ibits
        self.wbits = LLayer.wbits
        self.obits = LLayer.obits
        self.A = LLayer.A.astype(np.float32)
        self.B = LLayer.B.astype(np.float32)
        self.name = name

    def execute(self, v):
        raise Exception("TODO: not yet implemented")

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        # output bit width stays unchanged for LinearLayer
        return self.obits

    def paramfill(self, net):
        net.params[self.name][0].data[...] = self.A
        net.params[self.name][1].data[...] = self.B

    def codegen(self, model):
        return L.Scale(model, name=self.name, scale_param=dict(bias_term=True))

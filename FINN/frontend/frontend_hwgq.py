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
import os
# remove Caffe chatter on terminal
os.environ['GLOG_minloglevel'] = '2'

import caffe
import numpy as np
import FINN.core.layers as lb
import logging
from google.protobuf.text_format import Merge

# frontend for importing Caffe HWGQ networks


def importCaffeNetwork(modeldef, params):
    """
    Imports a trained HWGQ Caffe model and returns FINN representation. At the
    moment, data source layers are not supported, so the deploy.prototxt variant
   
    """
    if params is None:
        net = caffe.Net(modeldef, caffe.TEST)
    else:
        net = caffe.Net(modeldef, params, caffe.TEST)
    model = caffe.proto.caffe_pb2.NetParameter()
    Merge(open(modeldef, "rb").read(), model)
    numLayers = len(model.layer)
    outLayerStr = []
    outParams = []
    ret = []
    # TODO check that net is linear (no branching)
    dataLayerName = net.inputs[0] # any better way to get this?
    dataShape = net.blobs[dataLayerName].data.shape
    if dataShape[2] != dataShape[3]:
        raise Exception("Only square images supported for now")
    inDim = dataShape[2]
    inChans = dataShape[1]
    for i in range(numLayers):
        layerModel = model.layer[i]
        layerType = layerModel.type
        layerName = layerModel.name
        logging.info("Processing layer: %s (type %s). input (chans,dim)=(%d,%d)" % (layerName, layerType, inChans, inDim))
        if net.params.has_key(layerName):
            layerParams = net.params[layerName]
        if layerType == "Input":
            # TODO we should support some of the transformations that Caffe
            # supports on the input
            raise Exception("Input layer is not yet convertable, need input data shape instead")
        elif layerType == "Scale":
            A = layerParams[0].data
            if layerModel.scale_param.bias_term:
                B = layerParams[1].data
            else:
                B = np.zeros(shape=A.shape)
            ret += [lb.LinearLayer(A, B)]
        elif layerType == "BatchNorm":
            # epsilon to ensure non-zero operand to square root
            eps = layerModel.batch_norm_param.eps
            # batchnorm layer has the following data blobs:
            # [mean, variance, moving average factor]
            # BUG: mavf can be zero, causing invalid divide below
            mavf = layerParams[2].data[0]
            if mavf == 0:
                mavf = 1
            m = layerParams[0].data / mavf
            i = 1 / (np.sqrt( (layerParams[1].data / mavf) + eps ))
            numBatchNormChans = m.shape[0]
            # Caffe BN layers do not have b and g
            b = np.zeros((numBatchNormChans), dtype=np.float32)
            g = np.ones((numBatchNormChans), dtype=np.float32)
            # we want to implement batchnorm as a linear operation Mx+N
            # where Mx+N = g*i*(x-m)+b = g*i*x - g*i*m + b
            # so M = g*i and N = b - g*i*m
            M = g*i
            N = b - g*i*m
            #outLayerStr += ["linear"]
            #outParams += [M, N]
            ret += [lb.LinearLayer(M, N)]
        elif layerType == "Quant":
            # quantization layer
            # get quantization type and levels
            qfxn = layerModel.quant_param.forward_func
            qlevels = np.asarray(layerModel.quant_param.centers, dtype=np.float32)
            if qfxn == "hwgq":
                # add zero as an explicit level for HWGQ
                qlevels = np.concatenate((np.asarray([0.0], dtype=np.float32), qlevels))
                # check for uniform quantization -- all levels equally spaced
                isUniform = np.all(np.isclose(np.diff(qlevels, 2), 0))
                if not isUniform:
                    # TODO add a LookupTableLayer for nonlinear quantization support
                    raise Exception("Nonuniform quantization not yet supported")
                else:
                    # uniform quantization = threshold followed by linear transform
                    # compute thresholds as HWGQ does
                    qlevels_t = qlevels[1:] # exclude the zero level for thres. comp
                    thr = (qlevels_t[:-1] + qlevels_t[1:]) / 2.0
                    # add explicit zero threshold
                    thr = np.concatenate((np.asarray([0.0], dtype=np.float32), thr))
                    # emit threshold layer
                    #outLayerStr += ["thres"]
                    #outParams += [thr]
                    ret += [lb.ThresholdingLayer(thr)]
                    # TODO this should be ideally propagated (similar to bitwidths)
                    # using a transform
                    ret[-1].insize = inChans
                    ret[-1].outsize = inChans
                    # find the coefficients for the linear transform Fx + G
                    G = np.asarray([qlevels[0]])
                    F = np.asarray([qlevels[1] - qlevels[0]])
                    # emit linear layer with scalars
                    #outLayerStr += ["linear"]
                    #outParams += [F, G]
                    ret += [lb.LinearLayer(F, G)]
            elif qfxn == "sign":
                # sign quantization has its own layer type, but the core logic
                # still uses 0 as a threshold.
                thr = np.asarray([[0.0]], dtype=np.float32)
                ret += [lb.BipolarThresholdingLayer(thr)]
            else:
                raise Exception("Unsupported quantization function")

        elif layerType == "BinaryInnerProduct":
            # binary inner product layer may or may not have bias field
            # additionally, it may use the l1-norm as a scaling factor
            # need access to prototxt to find out whether to use alpha
            if not layerModel.binary_inner_product_param.use_binarization:
                raise Exception("use_binarization not set in BinaryInnerProduct layer")
            useBias = layerModel.inner_product_param.bias_term
            W = layerParams[0].data
            (rows, cols) = W.shape
            useAlpha = layerModel.binary_inner_product_param.use_alpha
            # the weights here are not yet binarized - need to do that
            # access and binarize the weights as done by the bnfc layer impl
            # binarize the weight matrix:
            Wbin = np.sign(W)
            # generate fully connected layer output

            # TODO indicate 1 bit signed (bipolar)
            ret += [lb.FullyConnectedLayer(Wbin, 1, 32, 32)]
            ret[-1].in_dim = inDim
            ret[-1].kernel = 1
            # treat the produced data as "rows"-channel, 1px images
            inChans = rows
            inDim = 1
            if useAlpha:
                # add a linear layer with A=alpha B=0 after the FC layer
                alpha = np.zeros(rows, dtype=np.float32)
                beta = np.zeros(rows, dtype=np.float32)
                Wabs = np.abs(W)
                for i in range(rows):
                    alpha[i] = Wabs[i].sum() / cols
                ret += [lb.LinearLayer(alpha, beta)]
            if useBias:
                # add bias as additive linear layer
                b = layerParams[1].data
                ret += [lb.LinearLayer(np.ones((rows), dtype=np.float32), b)]
        elif layerType == "BinaryConvolution":
            if not layerModel.binary_convolution_param.use_binarization:
                raise Exception("use_binarization not set in BinaryInnerProduct layer")
            useAlpha = layerModel.binary_convolution_param.use_alpha
            useBias = layerModel.convolution_param.bias_term
            ofm = layerModel.convolution_param.num_output
            # TODO warn about non-uniform stride/pad/kernelsize
            # kernel size
            if len(layerModel.convolution_param.kernel_size) == 0:
                raise Exception("Unknown kernel size")
            else:
                k = layerModel.convolution_param.kernel_size[0]
            # stride options
            if len(layerModel.convolution_param.stride) == 0:
                s = 1
            else:
                s = layerModel.convolution_param.stride[0]
            # padding options
            if len(layerModel.convolution_param.pad) == 0:
                pad = 0
            else:
                pad = layerModel.convolution_param.pad[0]
            # size of each output feature map
            outDim = ((inDim + 2*pad - k) / s) + 1
            W = layerParams[0].data
            # binarize kernel weights and output conv layer
            orig_shape = W.shape
            Wbin = np.sign(W)

            # TODO indicate 1 bit signed (bipolar)
            ret += [lb.ConvolutionLayer(Wbin, inDim, pad, s, 1, 1, 1)]
            ret[-1].kernel = k
            ret[-1].k = k
            ret[-1].stride = s
            ret[-1].parallel = layerModel.convolution_param.group
            # compute alphas, if needed
            if useAlpha:
                Wa = W.reshape((ofm, k*k*inChans))
                (rows, cols) = Wa.shape
                # add a linear layer with A=alpha B=0 after the conv layer
                alpha = np.zeros(rows, dtype=np.float32)
                Wabs = np.abs(Wa)
                for i in range(rows):
                    alpha[i] = Wabs[i].sum() / cols
                beta = np.zeros(rows, dtype=np.float32)
                #outLayerStr += ["linear"]
                #outParams += [alpha, beta]
                ret += [lb.LinearLayer(alpha, beta)]
            # TODO support conv bias
            if useBias:
                raise Exception("BinaryConvolution bias not yet supported")
            # update data shape passed to next layer
            inChans = ofm
            inDim = outDim
        elif layerType == "Convolution":
            useBias = layerModel.convolution_param.bias_term
            ofm = layerModel.convolution_param.num_output
            # TODO warn about non-uniform stride/pad/kernelsize
            # kernel size
            if len(layerModel.convolution_param.kernel_size) == 0:
                raise Exception("Unknown kernel size")
            else:
                k = layerModel.convolution_param.kernel_size[0]
            # stride options
            if len(layerModel.convolution_param.stride) == 0:
                s = 1
            else:
                s = layerModel.convolution_param.stride[0]
            # padding options
            if len(layerModel.convolution_param.pad) == 0:
                pad = 0
            else:
                pad = layerModel.convolution_param.pad[0]
            # size of each output feature map
            outDim = ((inDim + 2*pad - k) / s) + 1
            W = layerParams[0].data
            #outParams += [W]
            #outLayerStr += ["conv:%d:%d:%d:32:32:32" % (inDim, pad, s)]
            ret += [lb.ConvolutionLayer(W, inDim, pad, s, 32, 32, 32)]
            ret[-1].kernel = k
            ret[-1].stride = s
            ret[-1].parallel = layerModel.convolution_param.group
             # TODO support conv bias
            if useBias:
                rows = ofm    
                b = layerParams[1].data
                ret += [lb.LinearLayer(np.ones((rows), dtype=np.float32), b)]
                raise Exception("Convolution bias not yet supported")
            # update data shape passed to next layer
            inChans = ofm
            inDim = outDim
        elif layerType == "InnerProduct":
            useBias = layerModel.inner_product_param.bias_term
            W = layerParams[0].data
            (rows, cols) = W.shape
            # if the previous layer was a conv layer, interleave the columns
            # to match the interleaved channel data layout
            # generate fully connected layer output
            #outLayerStr += ["fc:32:32:32"]
            #outParams += [W]
            ret += [lb.FullyConnectedLayer(W, 32, 32, 32)]
            # treat the produced data as "rows"-channel, 1px images
            ret[-1].kernel =1
            inChans = rows
            inDim = 1
            if useBias:
                # add bias as additive linear layer
                b = layerParams[1].data
                #outLayerStr += ["linear"]
                #outParams += [np.ones((rows), dtype=np.float32), b]
                ret += [lb.LinearLayer(np.ones((rows), dtype=np.float32), b)]
        elif layerType == "Pooling":
            if inDim == 1:
                continue
            if layerModel.pooling_param.pool == 0:  # max pooling
                poolFxn = "MAX"
            elif layerModel.pooling_param.pool == 1:  # average pooling
                poolFxn = "AVE"
            else:
                raise Exception("Only max and average pooling supported for now")
            k = layerModel.pooling_param.kernel_size
            s = layerModel.pooling_param.stride
            #outLayerStr += ["maxpool:%d:%d:%d:%d" % (inDim, inChans, k, s)]
            ret += [lb.PoolingLayer(inDim, inChans, k, s, poolFxn)]
            # update data shape passed to next layer
            inChans = ofm
            inDim = ((inDim - k) / s) + 1
        elif layerType == "Softmax":
            ret += [lb.SoftmaxLayer()]
            ret[-1].outsize = inChans
            ret[-1].insize = inChans
        elif layerType == "ReLU":
            ret += [lb.ReLULayer()]
        elif layerType == "LRN":
            pass
        elif layerType == "Dropout":
            pass
        else:
            raise Exception("Unrecognized or unsupported layer: %s" % layerType)

    return ret

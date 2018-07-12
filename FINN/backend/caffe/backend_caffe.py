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
import FINN.transforms.transformations as trns
import FINN.core.layers as lb
import layers_caffe as lcaffe
import caffe
import os

# Backend functions to generate Caffe prototxt + caffemodel from FINN
# models. Note that the CaffeInteger* and CaffeMultiThresholdLayer are only
# supported in Yaman's bit serial Caffe branch.

def passConvertToCaffeLayers(pipeline, qnnengine):
    "Convert layers to corresponding Caffe-equivalent implementation layers."
    inStages = pipeline
    ret = []
    default_engine_maxbits = 4
    gemmlowp_maxbits = 8
    if qnnengine == "float":
        # force all layers to be generated as vanilla Caffe layers
        default_engine_maxbits = 0
        gemmlowp_maxbits = 0
    # note that layer, in and out buf names are empty -- we'll set those later
    for L in inStages:
        if lb.isFCLayer(L):
            if (L.wbits * L.ibits) <= default_engine_maxbits:
                ret += [lcaffe.CaffeIntegerInnerProductLayer("", L, qnnengine)]
            elif L.wbits <= gemmlowp_maxbits:
                ret += [lcaffe.CaffeIntegerInnerProductLayer("", L, "gemmlowp")]
            else:
                ret += [lcaffe.CaffeInnerProductLayer("", L)]
        elif lb.isConvLayer(L):
            if (L.wbits * L.ibits) <= default_engine_maxbits:
                ret += [lcaffe.CaffeIntegerConvolutionLayer("", L, qnnengine)]
            elif L.wbits <= gemmlowp_maxbits:
                ret += [lcaffe.CaffeIntegerConvolutionLayer("", L, "gemmlowp")]
            else:
                ret += [lcaffe.CaffeConvolutionLayer("", L)]
        elif lb.isPoolingLayer(L):
            ret += [lcaffe.CaffePoolLayer("", L)]
        elif lb.isThresholdLayer(L):
            ret += [lcaffe.CaffeMultiThresholdLayer("", L)]
        elif lb.isLinearLayer(L):
            ret += [lcaffe.CaffeScaleLayer("", L)]
        elif lb.isReLULayer(L):
            ret += [lcaffe.CaffeReLULayer("")]
        elif lb.isSoftmaxLayer(L):
            ret += [lcaffe.CaffeSoftmaxLayer("")]
        elif lcaffe.isCaffeLayer(L):
            ret += [L]
        else:
            raise Exception("Unsupported layer type for Caffe backend: %s" % L.get_type())

    return (ret, 0)

def passMarkByteIO(pipeline):
    "Use byte i/o between neighboring threshold-integer op pairs."
    inStages = pipeline
    inStages.reverse()
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lcaffe.isCaffeMultiThresholdLayer(layerA) and lcaffe.isCaffeIntMatrixLayer(layerB):
            layerA.use_byte_output = True
            layerB.use_byte_input = True
            ret += [layerA, layerB]
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]

    return (ret, 0)

def prepare_pipeline(pipeline, qnnengine):
    # compute/update bitwidths
    pipeline = trns.apply_repeated(pipeline, trns.passUpdateBitwidths)
    # convert all layers to their Caffe implementation variants
    selectedConvert = lambda x: passConvertToCaffeLayers(x, qnnengine)
    pipeline = trns.apply_repeated(pipeline, selectedConvert)
    # use byte i/o between neighboring threshold-integer op pairs.
    pipeline = trns.apply_repeated(pipeline, passMarkByteIO)
    # give names to each layer -- import to be able to set parameters
    pipeline = trns.apply_repeated(pipeline, trns.passGiveUniqueNames)
    return pipeline

def synthesize(pipeline, output_dir, data_shape, data_layer="", prefix="", qnnengine="bitserial"):
    if len(data_shape) != 4:
        raise Exception("data_shape must be 4D array (batchsize, channels, h, w)")
    deployFileName = output_dir+'/%sdeploy.prototxt' % prefix
    testFileName = output_dir+'/%stest.prototxt' % prefix
    paramFileName = output_dir+'/%sweights.caffemodel' % prefix
    # create output dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # convert pipeline into synthesizable form
    pipeline = prepare_pipeline(pipeline, qnnengine)
    # generate deploy.prototxt
    model = caffe.layers.Input(shape=dict(dim=data_shape))
    # generate Caffe model from each layer
    for L in pipeline:
        model = L.codegen(model)
    # save generated model to prototxt
    with open(deployFileName, 'w') as f:
        f.write(str(caffe.to_proto(model)))
    # load generated model
    net = caffe.Net(network_file=deployFileName, phase=caffe.TEST)
    # call each layer's filler function
    for L in pipeline:
        L.paramfill(net)
    # save the resulting parameters
    net.save(paramFileName)
    # only generate test prototxt if data layer is specified
    if data_layer != "":
        # unpack the data layer as tuple
        data, label = data_layer
        # Caffe model which we'll turn into a prototxt
        model = data
        # generate Caffe model from each layer
        for L in pipeline:
            model = L.codegen(model)
        # add top-1 and top-5 accuracy layers
        model_top1 = caffe.layers.Accuracy(model, label)
        model_top5 = caffe.layers.Accuracy(model, label, accuracy_param=dict(top_k=5))
        # save generated model to prototxt
        with open(testFileName, 'w') as f:
            f.write(str(caffe.to_proto(model_top1, model_top5)))
    # return the created Caffe net object
    return net

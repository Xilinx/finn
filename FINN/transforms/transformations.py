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


import copy
import numpy as np
import FINN.core.layers as lb
import FINN.core.quantize as qnt

# Transformations are a key part of FINN. Each transformation takes in a QNN
# represented in the FINN IR (and possibly some other parameters), and returns
# a transformed variant also represented in the FINN IR.
# Each transformation returns a tuple (transformed_qnn, num_changes) where
# num_changes is the number of alterations (>=0) applied on the input.
# Thus, it is vital that a transformation returns 0 when it made no changes, OR
# when it is sure that no more calls to this transform is needed, in order to
# avoid infinite loops.
# Based on this mechanic, we use the following function to repeatedly apply the
# same transform on the graph, until everything that can be transformed has been
# transformed:
def apply_repeated(pipeline, pass_to_apply):
    "Repeatedly applies a transform until there is nothing left to change."
    ret = copy.deepcopy(pipeline)
    while True:
        (ret, numChanges) = pass_to_apply(ret)
        if numChanges == 0:
            break
    return ret

# general-purpose (device-neutral) transformations

def directlyQuantizeLayer(layer, bits):
    "Apply direct quantization to given layer, returns [quantized layer, scaling layer]"
    assert(lb.isMatrixLayer(layer))
    qlayer = copy.deepcopy(layer)
    (Wint, alpha) = qnt.quantize_matrix(qlayer.W, bits)
    qlayer.W = Wint
    qlayer.wbits = bits
    slayer = lb.LinearLayer(A = alpha, B = np.zeros(alpha.shape))
    return [qlayer, slayer]

def directlyQuantizeAllFloatWeights(pipeline, bits):
    "Quantize all float weights in network to given number of bits."
    ret = []
    pipeline_copy = copy.deepcopy(pipeline)
    for L in pipeline_copy:
        if lb.isMatrixLayer(L):
            if L.wbits == 32:
                ret += directlyQuantizeLayer(L, bits)
            else:
                ret += [L]
        else:
            ret += [L]
    return ret

# TODO change name of this transform to "streamline" to avoid
# overloading Benjamin Graham's "cromulent networks"
def makeCromulent(pipeline, reorder_maxpool=True):
    "Simplifies a QNN by absorbing linear operators into thresholds."
    ret = pipeline
#    ret = apply_repeated(ret, passRemoveOneDimMaxPool)
#    ret = apply_repeated(ret, passUpdateChannels)
    ret = apply_repeated(ret, passFwdPropagateLinear)
    ret = apply_repeated(ret, passCollapseLinear)
    ret = apply_repeated(ret, passAbsorbLinearIntoThreshold)
    if reorder_maxpool:
        ret = apply_repeated(ret, passReorderMaxPooling)
        ret = apply_repeated(ret, passAbsorbLinearIntoThreshold)
    ret = apply_repeated(ret, passUpdateBitwidths)
    ret = apply_repeated(ret, passRoundUpIntThresholds)
    return ret


def passThresholdSetter(pipeline):
    inStages = pipeline
    ret = []
    lnum = 0
    for L in inStages:
        ltypename = L.__class__.__name__
        if ltypename == "ThresholdLayer":
            pass
        ret += [L]
        lnum += 1
    return (ret, 0)

def passGiveUniqueNames(pipeline):
    "Give unique name to each layer using simple enumeration."
    inStages = pipeline
    ret = []
    lnum = 0
    for L in inStages:
        ltypename = L.__class__.__name__
        L.name = "%s_%d" % (ltypename, lnum)
        ret += [L]
        lnum += 1
    return (ret, 0)

def passFuseActivations(pipeline):
    "Replace (Matrix, Threshold) layer pairs with fused equivalents."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lb.isMatrixLayer(layerA) and lb.isThresholdLayer(layerB):
            ret += [lb.MatrixThresholdLayer("", layerA, layerB)]
            numChanges += 1
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]
    return (ret, numChanges)

def passRoundUpIntThresholds(pipeline):
    "Round up thresholds of ThresholdingLayers with integer inputs."
    inStages = pipeline
    ret = []
    for L in inStages:
        # TODO this is not a good way to check for integer input --
        # fix this once we have i/o data types specified
        if lb.isThresholdLayer(L) and L.ibits <= 16:
            L.thresholds = np.ceil(L.thresholds).astype(np.int16)
        ret += [L]
    return (ret, 0)

def passUpdateBitwidths(pipeline, inputBitWidth = 1):
    "Update the input/output bitwidths throughout the graph."
    inStages = pipeline
    numChanges = 0
    ret = []
    lastBitWidth = inputBitWidth
    for L in inStages:
        iprev = L.ibits
        oprev = L.obits
        lastBitWidth = L.updateBitwidths(lastBitWidth)
        ret += [L]
    return (ret, 0)

def passCollapseLinear(pipeline):
    "Collapse neighboring linear (non-matrix) layers into a single linear layer."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lb.isLinearLayer(layerA) and lb.isLinearLayer(layerB):
            # let layerA be Jx + K and layerB be Mx + N
            # the output is M(Jx + K) + N = MJx + MK + N
            # so the collapsed layer will be (MJ)x + (MK + N)
            scaleNew = layerA.A * layerB.A
            shiftNew = layerB.A * layerA.B + layerB.B
            # TODO emit Scalarlb.LinearLayer, or just do shape=1 for those
            ret += [lb.LinearLayer(scaleNew, shiftNew)]
            numChanges += 1
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]

    return (ret, numChanges)

def passFwdPropagateLinear(pipeline):
    "Move linear layers past matrix and pooling layers."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lb.isLinearLayer(layerA) and lb.isMatrixLayer(layerB):
            # move the scalar ax+b to after the matrix layer Wx
            # originally we have W(ax+b) = Wax + Wb
            # desired: Mx+N = a(Wx) + Wb
            # repeat a and b to make appropriately-sized vectors
            a = layerA.A
            b = layerA.B
            W = layerB.W
            matrixLayerOutSize = W.shape[0]
            scaleNew = a*np.ones(matrixLayerOutSize)
            shiftNew = np.dot(W, b*np.ones(W.shape[1]))
            ret += [layerB, lb.LinearLayer(scaleNew, shiftNew)]
            numChanges += 1
        elif lb.isLinearLayer(layerA) and lb.isPoolingLayer(layerB):
            # TODO do we need to check layerA.A < 0 and maxpooling here?
            ret += [layerB, layerA]
            numChanges += 1
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]

    return (ret, numChanges)

def passAbsorbLinearIntoThreshold(pipeline):
    "Absorb linear transformations into the following threshold layer."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lb.isLinearLayer(layerA) and lb.isThresholdLayer(layerB):
            # absorb the linear transform Ax+B into the threshold layer
            # by updating each threshold T as Tnew = (T-B)/A
            A = layerA.A
            B = layerA.B
            T = layerB.thresholds
            Tnew = np.asarray([(t-B)/A for t in T])
            layerBThresClass = layerB.__class__
            ret += [layerBThresClass(Tnew)]
            numChanges += 1
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]
    return (ret, numChanges)

def passReorderMaxPooling(pipeline):
    "Move max pooling layers past thresholding layers."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if lb.isMaxPoolingLayer(layerA) and lb.isThresholdLayer(layerB):
            # need to check that the threholding preserves max to reoder
            for t in range(len(layerB.thresholds)-1):
                if not (layerB.thresholds[t+1] >= layerB.thresholds[t]).all():
                    raise Exception("Threshold does not preserve max")
            # re-insert layers in reversed order
            ret += [layerB, layerA]
            numChanges += 1
        else:
            ret += [layerA]
            inStages.append(layerB)
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]

    return (ret, numChanges)

def passInterleaveChannels(pipeline):
    """Interleave the weight matrices of all convolutional layers, plus the first
    subsequent fully-connected layer."""
    ret = []
    numChanges = 0
    # whether the inputs to the current layer were interleaved
    last_output_interleaved = 0
    # the interleave factor for the inputs to the current layer
    last_output_interleave_factor = 0
    for L in pipeline:
        if lb.isConvLayer(L):
            # interleave the conv weight matrix
            # originally, we assume the data layout is [ofm][ifm][k][k]
            W = L.W.reshape(L.ofm, L.ifm, L.get_filter_dim(), L.get_filter_dim())
            # transpose the weight tensor to be [ofm][k][k][ifm]
            W = W.transpose(0, 2, 3, 1)
            # put back into matrix form and set layer weight matrix
            L.W = W.reshape(L.ofm, -1)
            ret += [L]
            last_output_interleaved = 1
            last_output_interleave_factor = L.ofm
        elif lb.isFCLayer(L) and last_output_interleaved == 1:
            # interleave the columns in the weight matrix of the first FC
            # layer
            r = L.W.shape[0]
            c = L.W.shape[1]
            W = L.W.reshape(r, last_output_interleave_factor, -1)
            L.W = W.transpose(0, 2, 1).reshape(r, c)
            ret += [L]
            # output is no longer interleaved
            last_output_interleaved = 0
        else:
            # copy layer as-is
            ret += [L]
    return (ret, numChanges)

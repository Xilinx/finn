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
import FINN.core.layers as lb
import FINN.backend.fpga.codegen_fpga as cg
from FINN.backend.fpga.utils import *

def isFPGAMaxPoolingLayer(L):
    "Whether given layer is an FPGA maxpooling layer."
    # TODO check for other pooling variants
    return isinstance(L, FPGAMaxPoolLayer)

def isFPGAConvLayer(L):
    "Whether given layer is an FPGA convolution layer."
    # TODO should check for the non-threshold version as well
    return isinstance(L, FPGABipolarConvThresholdLayer)

def isMultichannel(L):
    "Whether given layer uses multichannel input/output."
    return isFPGAConvLayer(L) or isFPGAMaxPoolingLayer(L)

def isFPGAMatrixLayer(L):
    "Whether given layer is an FPGA matrix layer with PE/SIMD/MMV fields."
    return isinstance(L, cg.StreamingMatrixCodegen)

def isFPGABufferLayer(L):
    "Wheter given layer is an FPGA buffer (FIFO) layer."
    return isinstance(L, FPGABufferLayer)

def isFPGAStreamingLayer(L):
    "Whether given layer is an FPGA layer using stream I/O."
    return isinstance(L, cg.StreamingComponentCodegen)

# no-activation variant of FPGABipolarMatrixThresholdLayer
class FPGABipolarMatrixLayer(lb.FullyConnectedLayer, cg.StreamingMatrixCodegen):
    def __init__(self, fcl):
        # TODO support conv layers -- or maybe as own class instead..
        if fcl.get_type() != "FullyConnectedLayer":
            raise Exception("Only fully connected layers currently supported")
        # copy all attributes from base FullyConnectedLayer
        self.__dict__.update(fcl.__dict__)
        # TODO checking width is not enough -- also need to check encoding
        # (bipolar/regular)
        if self.wbits != 1:
            raise Exception("Only binarized weights supported")
        if self.ibits > 8:
            raise Exception("Only sub-8-bit activations supported")
        # make sure weight array really is bipolar
        if not isBipolarMatrix(self.W):
            raise Exception("Non-bipolar elements found in weight matrix")
        # manually set output width to accumulator width
        self.obits = self.getAccWidth()
        self.pe = 1
        self.simd = 1
        self.mmv = 1

    # implementations for StreamingComponentCodegen base class
    def getIBits(self):
        "Bits per single input element."
        return self.ibits

    def getIGroup(self):
        "Number of elements in input group (stream)."
        return self.getSIMD()

    def getOBits(self):
        "Bits per single output element."
        return self.getAccWidth()

    def getOGroup(self):
        "Number of elements in output group (stream)."
        return self.getPE()

    def isOSigned(self):
        "Whether the output elements are signed."
        if self.isBipolarTimesBipolar():
            # pure BNN uses positive-only output
            return False
        else:
            return True

    # implementations for StreamingMatrixCodegen base class
    def getW(self):
        "Return the weight matrix."
        # note that the weight matrix is converted to bipolar encoding first,
        # since the HLS weights will be produced from this.
        if self.isBipolarTimesRegular():
            # the RPNN primitives assume inverted encoding:
            return bipolarEncoding(-self.W)
        elif self.isBipolarTimesBipolar():
            # the BNN primitives work fine with the regular encoding
            return bipolarEncoding(self.W)
        else:
            raise Exception("Don't know how to provide W for this matrix operation.")

    def getWBits(self):
        return 1

    def getAccWidth(self):
        "Returns the number of bits used for the PE internal accumulators."
        # TODO can use original obits for minimal accumulators, but needed
        # to test that this works correctly.
        return 16

class FPGABipolarMatrixThresholdLayer(lb.MatrixThresholdLayer, cg.StreamingMatrixCodegen):
    def __init__(self, mtl):
        super(cg.StreamingMatrixCodegen, self).__init__()
        # copy all attributes from base MatrixThresholdLayer
        self.__dict__.update(mtl.__dict__)
        # TODO checking width is not enough -- also need to check encoding
        # (bipolar/regular)
        self.outsize = self.mlayer.W.shape[0]
        self.insize = self.mlayer.W.shape[1]
        self.kernel = self.mlayer.get_filter_dim()
        self.stride = self.mlayer.get_stride()
        if self.wbits != 1:
            raise Exception("Only binarized weights supported")
        if self.ibits > 8:
            raise Exception("Only sub-8-bit activations supported")
        if self.mlayer.get_type() != "FullyConnectedLayer":
            raise Exception("Only fully connected layers currently supported")
        # make sure weight array really is bipolar
        if not isBipolarMatrix(self.mlayer.W):
            raise Exception("Non-bipolar elements found in weight matrix")
        self.pe = 1
        self.simd = 1
        self.mmv = 1

    def get_in_dim(self):
        return self.mlayer.get_in_dim()

    # implementations for StreamingComponentCodegen base class
    def getIBits(self):
        "Bits per single input element."
        return self.ibits

    def getIGroup(self):
        "Number of elements in input group (stream)."
        return self.getSIMD()

    def getOBits(self):
        "Bits per single output element."
        return self.obits

    def getOGroup(self):
        "Number of elements in output group (stream)."
        return self.getPE()

    def isOSigned(self):
        "Whether the output elements are signed."
        if isinstance(self.tlayer, lb.BipolarThresholdingLayer):
            # bipolar thresholding returns signed numbers
            return True
        else:
            # assume all other thresholding returns unsigned numbers
            return False

    # implementations for StreamingMatrixCodegen base class
    def getW(self):
        "Return the weight matrix."
        # note that the weight matrix is converted to bipolar encoding first,
        # since the HLS weights will be produced from this.
        if self.isBipolarTimesRegular():
            # the RPNN primitives assume inverted encoding:
            return bipolarEncoding(-self.mlayer.W)
        elif self.isBipolarTimesBipolar():
            # the BNN primitives work fine with the regular encoding
            return bipolarEncoding(self.mlayer.W)
        else:
            raise Exception("Don't know how to provide W for this matrix operation.")
    
    def getNumOps(self):
        return self.getW().size *2

    def getWBits(self):
        return 1

    def getAccWidth(self):
        "Returns the number of bits used for the PE internal accumulators."
        # TODO can use mtlayer.mlayer.obits for minimal accumulators, but needed
        # to test that this works correctly.
        return 16

    # additions and overrides for also emitting thresholds
    def getTMemCount(self):
        return self.getMatrixDims()[1] / self.getPE()

    def getTMemName(self):
        return "thres_" + self.getParamSuffix()

    def getTMemDType(self):
        nthres = self.tlayer.thresholds.shape[0]
        if self.isBipolarTimesRegular():
            # the rpnn thresholding function wants an extra threshold
            nthres += 1
        return "const ap_uint<%d>" % (self.getAccWidth() * nthres)

    def getT(self):
        "Return the thresholds for MVTU"
        if self.isBipolarTimesBipolar():
            # TODO also need to check if thresholding is bipolar here
            # ensure positive thresholds for BNNs
            return makePositiveThresholds(self.tlayer.thresholds, self.getNumInputElems())
        elif self.isBipolarTimesRegular():
            # return thresholds as-is for QNN thresholding
            return self.tlayer.thresholds
        else:
            raise Exception("Unsupported thresholding case")

    def codegen_params(self, output_dir):
        # call base codegen for weight matrix generation
        super(FPGABipolarMatrixThresholdLayer, self).codegen_params(output_dir)

        # also generate code for thresholds
        # TODO need to cater for the QNN thresholds case here
        packed_t = cg.packThresholdMatrix(self.getT(), self.getPE(), self.getAccWidth())
        thres_hls = cg.npyNDArrayToHLSArray(
            packed_t, self.getTMemDType(), self.getTMemName(), str
        )
        # emit into file
        with open(output_dir + "/" + self.getParamFileName(), "a") as f:
            f.write("\n\n\n// thresholds\n")
            f.write(thres_hls)

    def codegen_architecture(self):
        ret = ""
        if self.isBipolarTimesBipolar():
            # layer function call for BNN variant
            ret = "\nFCLayer_Batch<"
            ret += "%d, %d, " % (self.getInStreamW(), self.getOutStreamW())
            ret += "%d, %d, %d, " % (self.getSIMD(), self.getPE(), self.getAccWidth())
            ret += "%d, %d, " % self.getMatrixDims()
            ret += "%d, %d>(" % (self.getWMemCount(), self.getTMemCount())
            ret += "%s, %s, " % (self.getIBufName(), self.getOBufName())
            ret += "%s, %s, " % (self.getWMemName(), self.getTMemName())
            ret += "numReps);"
        elif self.isBipolarTimesRegular():
            # layer function call for QNN variant
            ret += "\nMatrixVector_Precision_Batch<"
            ret += "%d, %d, " % (self.getSIMD(), self.getPE())
            # the rpnn thresholding fxn needs an extra threshold
            nthres = self.tlayer.thresholds.shape[0] + 1
            bits_per_thres = self.getAccWidth() * nthres
            ret += "%d, %d, " % (self.getWBits(), bits_per_thres)
            ret += "%d, %d, " % self.getMatrixDims()
            # note that TMemCount is set to one for the dummy threshold var
            ret += "%d, %d, " % (self.getWMemCount(), self.getTMemCount())
            ret += "%d, %d, " % (self.getIBits(), self.getOBits())
            # ActivationType is set to FULL_THRESHOLDS
            # TODO -- if input is signed, ap_uint here needs to change
            ret += "%d, %s, %s>(" % (self.getAccWidth(), "FULL_THRESHOLDS", "ap_uint")
            ret += "%s, %s, " % (self.getIBufName(), self.getOBufName())
            dummy_thres_name = "dummythres%s" % self.getParamSuffix()
            ret += "%s, %s, " % (self.getWMemName(), self.getTMemName())
            ret += "numReps);"
        else:
            raise Exception("Unsupported case for code generation")
        return ret

    def codegen_declarations(self):
        ret="""
        #pragma HLS ARRAY_PARTITION variable=%s complete dim=1
        #pragma HLS ARRAY_PARTITION variable=%s complete dim=1
        """ % (self.getWMemName(), self.getTMemName())
        return ret

class FPGABufferLayer(lb.Layer):
    def __init__(self):
        self.name = ""
        # ibits/obits will be propagated with updateBitwidths
        self.ibits = 1
        self.obits = 1
        # FIFO width/depth will be determined in a transformation pass
        self.streamwidth = 1
        self.depth = 2

    def execute(self, v):
        # pass data through without any changes
        return v

    def updateBitwidths(self, inBitWidth):
        self.ibits = inBitWidth
        self.obits = inBitWidth
        return self.obits

    def codegen_params(self, output_dir):
        # no parameters for buffer layer
        pass

    def codegen_globals(self):
        return ""

    def codegen_architecture(self):
        return ""

    def codegen_declarations(self):
        ret = ""
        # instantiate stream variable
        # TODO stream type needs to be more flexible, not just ap_uint
        ret += "\nstream<ap_uint<%d> > %s;" % (self.streamwidth, self.name)
        # add stream pragma to set depth
        ret += "\n#pragma HLS stream depth=%d variable=%s" % (self.depth, self.name)
        return ret

class FPGABipolarConvThresholdLayer(lb.MatrixThresholdLayer, cg.StreamingMatrixCodegen):
    def __init__(self, mtl):
        # copy all attributes from base MatrixThresholdLayer
        self.__dict__.update(mtl.__dict__)
        # TODO checking width is not enough -- also need to check encoding
        # (bipolar/regular)
        if self.wbits != 1:
            raise Exception("Only binarized weights supported")
        if self.ibits > 8:
            raise Exception("Only sub-8-bit activations supported")
        if not lb.isConvLayer(self.mlayer):
            raise Exception("FPGABipolarConvThresholdLayer needs conv as matrix layer")
        # make sure weight array really is bipolar
        if not isBipolarMatrix(self.mlayer.W):
            raise Exception("Non-bipolar elements found in weight matrix")
        # ConvolutionMMVInputGenerator needs a fix to handle uneven stride
        if self.mlayer.k % self.mlayer.stride != 0:
            raise Exception("ConvolutionMMVInputGenerator currently needs window mod stride == 0")
        self.pe = 1
        self.simd = 1
        self.mmv = 1

    def get_in_dim(self):
        return self.mlayer.get_in_dim()
    
    def isSamePadding(self):
        "Return whether this layer uses the 'same' padding mode"
        return self.mlayer.idim == self.mlayer.odim

    def isValidPadding(self):
        "Return whether this layer uses the 'valid' paddong mode"
        return self.mlayer.pad == 0

    # TODO override getMMV/PE/SIMD to check for feasibility conditions
    # implementations for StreamingComponentCodegen base class
    def getIBits(self):
        "Bits per single input element."
        return self.ibits

    def getIGroup(self):
        "Number of elements in input group (stream)."
        return self.mlayer.ifm

    def getOBits(self):
        "Bits per single output element."
        return self.obits

    def getOGroup(self):
        "Number of elements in output group (stream)."
        return self.mlayer.ofm

    def isOSigned(self):
        "Whether the output elements are signed."
        if isinstance(self.tlayer, lb.BipolarThresholdingLayer):
            # bipolar thresholding returns signed numbers
            return True
        else:
            # assume all other thresholding returns unsigned numbers
            return False

    # override part of resource cost model for conv layers
    def bram_cost(self):
        wshape = self.getW().shape
        wmem0 = (wshape[0] * wshape[1]) / (self.getSIMD() * self.getPE())
        bram_mode_width = math.log(self.getSIMD(), 2) + 1
        bram_mode_depth = (512*32) / bram_mode_width
        wmem = self.getPE() * np.ceil(float(self.getSIMD())/bram_mode_width) * np.ceil(float(wmem0)/bram_mode_depth)
        input_gen = self.getMMV() * ((self.mlayer.k/self.mlayer.stride)+1) * np.ceil(float(self.getIBits()*self.mlayer.ifm)/32) * np.ceil(float(self.mlayer.idim*self.mlayer.stride)/512)
        return wmem + input_gen

    def layer_ops(self):
        wshape = self.getW().shape
        outpix = self.mlayer.odim * self.mlayer.odim
        return 2 * (wshape[0] * wshape[1] * outpix) # *2 due to MAC

    def getMaxSIMD(self):
        # conv layers have an extra constraint on SIMD
        return self.mlayer.ifm

    def getMaxMMV(self):
        # conv layers can do MMV
        return self.mlayer.odim

    # implementations for StreamingMatrixCodegen base class
    def getW(self):
        "Return the weight matrix."
        # note that the weight matrix is converted to bipolar encoding first,
        # since the HLS weights will be produced from this.
        if self.isBipolarTimesRegular():
            # the RPNN primitives assume inverted encoding:
            return bipolarEncoding(-self.mlayer.W)
        elif self.isBipolarTimesBipolar():
            # the BNN primitives work fine with the regular encoding
            return bipolarEncoding(self.mlayer.W)
        else:
            raise Exception("Don't know how to provide W for this matrix operation.")

    def getWBits(self):
        return 1

    def getAccWidth(self):
        "Returns the number of bits used for the PE internal accumulators."
        # TODO can use mtlayer.mlayer.obits for minimal accumulators, but needed
        # to test that this works correctly.
        return 16

    # override io stream widths, since those are set by ifm/ofm
    def getInStreamW(self):
        "Width of input stream in bits"
        return self.getIBits() * self.mlayer.ifm

    def getOutStreamW(self):
        "Width of output stream in bits"
        return self.getOBits() * self.mlayer.ofm

    # override # of input-output elems, since the defaults are for matrix matrix
    def getNumInputElems(self):
        "Number of single input elements for this layer"
        return self.mlayer.ifm * self.mlayer.idim * self.mlayer.idim

    def getNumOutputElems(self):
        "Number of single output elements for this layer"
        return self.mlayer.ofm * self.mlayer.odim * self.mlayer.odim

    # additions and overrides for also emitting thresholds
    def getTMemCount(self):
        return self.getMatrixDims()[1] / self.getPE()

    def getTMemName(self):
        return "thres_" + self.getParamSuffix()

    def getTMemDType(self):
        nthres = self.tlayer.thresholds.shape[0]
        if self.isBipolarTimesRegular():
            # the rpnn thresholding function wants an extra threshold
            nthres += 1
        return "const ap_uint<%d>" % (self.getAccWidth() * nthres)

    def getT(self):
        "Return the thresholds for MVTU"
        if self.isBipolarTimesBipolar():
            # TODO also need to check if thresholding is bipolar here
            # ensure positive thresholds for BNNs
            return makePositiveThresholds(self.tlayer.thresholds, self.getNumInputElems())
        elif self.isBipolarTimesRegular():
            # return thresholds as-is for QNN thresholding
            return self.tlayer.thresholds
        else:
            raise Exception("Unsupported thresholding case")

    def codegen_params(self, output_dir):
        # call base codegen for weight matrix generation
        super(FPGABipolarConvThresholdLayer, self).codegen_params(output_dir)

        # also generate code for thresholds
        # TODO need to cater for the QNN thresholds case here
        packed_t = cg.packThresholdMatrix(self.getT(), self.getPE(), self.getAccWidth())
        thres_hls = cg.npyNDArrayToHLSArray(
            packed_t, self.getTMemDType(), self.getTMemName(), str
        )
        # emit into file
        with open(output_dir + "/" + self.getParamFileName(), "a") as f:
            f.write("\n\n\n// thresholds\n")
            f.write(thres_hls)

    def codegen_architecture(self):
        ret = ""
        # it doesn't make sense to have more PEs than conv weight matrix rows
        assert(self.pe <= self.mlayer.ofm)
        # the following is because the dwc inside ConvolutionalLayerMMV_Valid_Batch
        # implicitly assumes simd <= ifm
        assert(self.simd <= self.mlayer.ifm)
        # the following are ConvolutionMMVInputGenerator assumptions
        assert(self.mlayer.odim % self.mmv == 0)
        assert(self.mmv <= self.mlayer.odim)
        if self.isBipolarTimesBipolar():
            # layer function call for BNN variant
            #raise Exception("BNN convs not yet implemented")
            ret += "\nConvLayerMMV_BNN_Batch<"
            ret += "%d, %d, " % (self.mlayer.k, self.mlayer.ifm)
            ret += "%d, %d, %d, %d, " % (self.mlayer.get_in_dim(), self.mlayer.ofm,  self.mlayer.get_out_dim(), self.mlayer.stride)
            ret += "%d, %d, " % (self.getSIMD(), self.getPE())
            ret += "%d, %d, %d>" % (16, self.getWMemCount(), self.getTMemCount()) #XXX Hard coded popcount width to 16.
            ret += "(%s, %s, " % (self.getIBufName(), self.getOBufName())
            ret += "%s, %s, " % (self.getWMemName(), self.getTMemName())
            ret += "numReps);"
        elif self.isBipolarTimesRegular():
            # layer function call for QNN variant
            if self.isSamePadding() or self.isValidPadding():
                if self.isSamePadding():
                    ret += "\nConvolutionalLayerMMV_Same_Batch<"
                elif self.isValidPadding():
                    ret += "\nConvolutionalLayerMMV_Valid_Batch<"
                ret += "%d, %d, " % (self.mlayer.k, self.mlayer.ifm)
                ret += "%d, %d, %d, " % (self.mlayer.idim, self.mlayer.ofm, self.mlayer.stride)
                ret += "%d, %d, " % (self.getSIMD(), self.getPE())
                ret += "%d, %d, " % (self.getWMemCount(), self.getTMemCount())
                # the rpnn thresholding fxn needs an extra threshold
                nthres = self.tlayer.thresholds.shape[0] + 1
                bits_per_thres = self.getAccWidth() * nthres
                ret += "%d, %d, %d, " % (self.getWBits(), bits_per_thres, self.getAccWidth())
                ret += "%d, %d, " % (self.getIBits(), self.getOBits())
                # ActivationType is set to FULL_THRESHOLDS
                # TODO -- if input is signed, ap_uint here needs to change
                ret += "%d, %s, %s>(" % (self.getMMV(), "FULL_THRESHOLDS", "ap_uint")
                ret += "%s, %s, " % (self.getIBufName(), self.getOBufName())
                dummy_thres_name = "dummythres%s" % self.getParamSuffix()
                ret += "%s, %s, " % (self.getWMemName(), self.getTMemName())
                ret += "numReps);"
            else:
                raise Exception("Only same and valid padding is supported for now")
        else:
            raise Exception("Unsupported case for code generation")
        return ret

    def codegen_declarations(self):
        ret="\n#pragma HLS ARRAY_PARTITION variable=%s complete dim=1" % self.getWMemName()
        ret+="\n#pragma HLS ARRAY_PARTITION variable=%s complete dim=1" % self.getTMemName()
        return ret

class FPGAMaxPoolLayer(lb.PoolingLayer, cg.StreamingComponentCodegen):
    def __init__(self, pl):
        # copy all attributes from base PoolingLayer
        self.__dict__.update(pl.__dict__)
        if self.poolFxn != "MAX":
            raise Exception("Only maxpooling supported for synthesis for now.")
        if self.ibits > 8:
            raise Exception("Only sub-8-bit activations supported")
        #if self.ibits == 1:
        #    raise Exception("BNN maxpooling not yet supported")

    # implementations for StreamingComponentCodegen base class
    def getIBits(self):
        "Bits per single input element."
        return self.ibits

    def getIGroup(self):
        "Number of elements in input group (stream)."
        return self.chans

    def getOBits(self):
        "Bits per single output element."
        return self.obits

    def getOGroup(self):
        "Number of elements in output group (stream)."
        return self.chans

    def isOSigned(self):
        "Whether the output elements are signed."
        return False

    def getNumInputElems(self):
        "Number of single input elements for this layer"
        return self.idim * self.idim * self.chans

    def getNumOutputElems(self):
        "Number of single output elements for this layer"
        return self.odim * self.odim * self.chans

    def codegen_architecture(self):
        "Generate the actual function call for this layer"
        hls_paddim_valid = (1 + (self.idim - self.k)/self.s) * self.k;
        hls_padplus_same = (1 if self.idim % self.s != 0 else 0)
        hls_paddim_same = (self.idim / self.s + hls_padplus_same) * self.k
        hls_outdim_valid = hls_paddim_valid / self.k
        hls_outdim_same = hls_paddim_same / self.k
        ret = ""
        if self.getIBits() == 1:
            ret = "\nMaxPool_BNN_Batch<"
            ret += "%d, %d, %d>(" % (self.idim, self.k, self.chans)
            ret += "%s, %s, numReps);" % (self.getIBufName(), self.getOBufName())
        elif self.odim == hls_outdim_valid:
            ret = "\nMaxPoolStride_Valid_Batch<"
            ret += "%d, %d, %d, " % (self.idim, self.k, self.s)
            ret += "%d, %d>(" % (self.chans, self.getIBits())
            ret += "%s, %s, numReps);" % (self.getIBufName(), self.getOBufName())
        elif self.odim == hls_outdim_same:
            ret = "\nMaxPoolStride_Same_Batch<"
            ret += "%d, %d, %d, " % (self.idim, self.k, self.s)
            ret += "%d, %d>(" % (self.chans, self.getIBits())
            ret += "%s, %s, numReps);" % (self.getIBufName(), self.getOBufName())
        else:
            raise Exception("Unsupported max-pooling size")

        return ret

class FPGADataWidthConverter(lb.Layer, cg.StreamingComponentCodegen):
    def __init__(self, layerIn, layerOut):
        self.ibits = layerIn.obits
        self.obits = layerOut.ibits
        self.igroup = layerIn.getOGroup()
        self.ogroup = layerOut.getIGroup()
        self.ielems = layerIn.getNumOutputElems()
        self.oelems = layerOut.getNumInputElems()

    def execute(self, v):
        raise Exception("Not implemented")

    def updateBitwidths(self, ibits):
        raise Exception("Not implemented")

    # implementations for StreamingComponentCodegen base class
    def getIBits(self):
        "Bits per single input element."
        return self.ibits

    def getIGroup(self):
        "Number of elements in input group (stream)."
        return self.igroup

    def getOBits(self):
        "Bits per single output element."
        return self.obits

    def getOGroup(self):
        "Number of elements in output group (stream)."
        return self.ogroup

    def isOSigned(self):
        "Whether the output elements are signed."
        return False

    def getNumInputElems(self):
        "Number of single input elements for this layer"
        return self.ielems

    def getNumOutputElems(self):
        "Number of single output elements for this layer"
        return self.oelems

    def codegen_architecture(self):
        "Generate the actual function call for this layer"
        assert(self.ielems % self.igroup == 0)
        assert(self.ielems % self.ogroup == 0)
        inPerExec = self.ielems / self.igroup
        ret = "\nDataWidthConverter_Batch<"
        ret += "%d, %d, %d>(" % (self.getInStreamW(), self.getOutStreamW(), inPerExec)
        ret += "%s, %s, numReps);" % (self.getIBufName(), self.getOBufName())
        return ret

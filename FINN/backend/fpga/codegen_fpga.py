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

import FINN.core.config as config
import numpy as np
from FINN.backend.fpga.utils import *
import math
# base class for streaming component code generation
# mostly things concerning input/output stream data types
class StreamingComponentCodegen():
    def __init__(self):
        self.instreamW = 0
    # subclasses must implement the following:
    def getIBits(self):
        "Bits per single input element."
        pass

    def getIGroup(self):
        "Number of elements in input group (stream)."
        pass

    def getOBits(self):
        "Bits per single output element."
        pass

    def isOSigned(self):
        "Whether the output elements are signed."
        pass

    def getOGroup(self):
        "Number of elements in output group (stream)."
        pass

    def getNumInputElems(self):
        "Number of single input elements for this layer"
        pass

    def getNumOutputElems(self):
        "Number of single output elements for this layer"
        pass

    # subclasses can implement the following if appropriate:
    def codegen_params(self, output_dir):
        "Generate parameters for this layer"
        pass

    def codegen_globals(self):
        "Generate global includes etc for this layer"
        return ""

    def codegen_declarations(self):
        "Generate local variable and stream declarations for this layer"
        return ""

    def codegen_architecture(self):
        "Generate the actual function call for this layer"
        return ""

    # subclass should also have name, inBufName, outBufName fields:
    def getParamSuffix(self):
        "Returns a string suffix for parameters."
        return self.name

    def getIBufName(self):
        return self.inBufName

    def getOBufName(self):
        return self.outBufName

    # the rest of the methods have more concrete implementations:
    def getInStreamW(self):
        "Width of input stream in bits"
        return self.getIBits() * self.getIGroup()

    def getOutStreamW(self):
        "Width of output stream in bits"
        return self.getOBits() * self.getOGroup()

    def getInputElemDType(self):
        "HLS datatype for a single input element"
        return "ap_uint<%d>" % self.getIBits()

    def getOutputElemDType(self):
        "HLS datatype for a single output element"
        return "ap_uint<%d>" % self.getOBits()

    def getInputStreamDType(self):
        "HLS datatype for a group of input elements"
        return "ap_uint<%d>" % (self.getInStreamW())

    def getOutputStreamDType(self):
        "HLS datatype for a PE group of output elements"
        return "ap_uint<%d>" % (self.getOutStreamW())

    def getInStreamDecl(self):
        return "stream<%s > & %s" % (self.getInputStreamDType(), self.getIBufName())

    def getOutStreamDecl(self):
        return "stream<%s > & %s" % (self.getOutputStreamDType(), self.getOBufName())

    def codegen_single2instream(self, singleStrmName, inStrmName, dtype="T"):
        "Generate code to convert a single-element stream into this layer's input stream."
        ret = ""
        lastStrmName = singleStrmName
        tmpName = "%s_iconvert" % self.getParamSuffix()
        ret += "\n// cast to single input element data type"
        ret += "\nstream<%s > %s;" % (self.getInputElemDType(), tmpName)
        ret += "\nCast(%s, %s, %d);" % (lastStrmName, tmpName, self.getNumInputElems())
        ret += "\n// convert to appropriate input stream width"
        ret += "\nDataWidthConverter<%d, %d, %d>(%s, %s);" % (self.getIBits(), self.getInStreamW(), self.getNumInputElems(), tmpName, inStrmName)
        return ret

    def codegen_outstream2single(self, outStrmName, singleStrmName, dtype="T"):
        "Generate code to convert this layer's output stream into a single-element dtype stream."
        # here is the logic behind the operations here:
        # 1. unpack from transport bundle: ap_uint<group> to ap_uint<single>
        # 2. cast to single actual output element type: ap_int<single> to out_dtype
        # 3. cast to dtype
        # 4. add ScaleShiftByConstant if needed for bipolar decoding etc.
        # extract single elements
        tmpName = "%s_oconvert" % self.getParamSuffix()
        ret = "\nstream<%s > %s;" % (self.getOutputElemDType(), tmpName)
        numOutputGroups = (self.getNumOutputElems() * self.getOBits()) / self.getOutStreamW()
        ret += "\nDataWidthConverter<%d, %d, %d>(%s, %s);" % (self.getOutStreamW(), self.getOBits(), numOutputGroups, outStrmName, tmpName)
        # cast to dtype
        ret += "\n// cast to single elem type"
        ret += "\nCast(%s, %s, %d);" % (tmpName, singleStrmName, self.getNumOutputElems())
        return ret

# code generator base for streaming layers that have a weight matrix
class StreamingMatrixCodegen(StreamingComponentCodegen, object):
    def __init__(self):
        super(StreamingComponentCodegen, self).__init__()


    # subclasses must implement the following:
    def getW(self):
        "Return the weight matrix."
        pass

    def getInStreamW(self):
        return self.instreamW

    def getWBits(self):
        "Returns bits per element of the weight matrix."
        pass

    def getAccWidth(self):
        "Returns the number of bits used for the PE internal accumulators."
        pass

    # subclasses should have the following member variables:
    def getPE(self):
        "Return the number of PEs."
        return self.pe

    def getSIMD(self):
        "Return the number of SIMD lanes per PE."
        return self.simd

    def getMMV(self):
        "Return the number of MMV lanes per PE."
        return self.mmv

    # the rest of the methods have more concrete implementations:
    def isBipolarTimesBipolar(self):
        # TODO checking number of bits is not sufficient to determine bipolar
        return self.getIBits() == 1 and self.getWBits() == 1

    def isBipolarTimesRegular(self):
        # TODO checking number of bits is not sufficient to determine bipolar
        return self.getIBits() > 1 and self.getWBits() == 1

    def getMatrixDims(self):
        "Returns (cols, rows) of the weight matrix"
        # TODO need to add padding here if we want to support unaligned PE/SIMD
        wshape = self.getW().shape
        return (wshape[1], wshape[0])

    def getWMemCount(self):
        return np.prod(self.getMatrixDims()) / (self.getPE() * self.getSIMD())

    def getWMemName(self):
        return "weights_" + self.getParamSuffix()

    def getWMemDType(self):
        return "const ap_uint<%d>" % self.getSIMD()

    def getParamFileName(self):
        return "params_" + self.getParamSuffix() + ".h"

    # resource cost model functions
    def bram_cost(self):
        wshape = self.getW().shape
        wmem0 = (wshape[0] * wshape[1]) / (self.getSIMD() * self.getPE())
        bram_mode_width = math.log(self.getSIMD(), 2) + 1
        bram_mode_depth = (512*32) / bram_mode_width
        wmem = self.getPE() * np.ceil(float(self.getSIMD())/bram_mode_width) * np.ceil(float(wmem0)/bram_mode_depth)
        return wmem

    def lut_cost(self):
        return config.prec_op_cost[self.getIBits() * self.getWBits()] * self.ops_per_cycle()

    def layer_ops(self):
        wshape = self.getW().shape
        return 2 * (wshape[0] * wshape[1]) # *2 due to MAC

    def ops_per_cycle(self):
        return self.getSIMD() * self.getPE() * self.getMMV() * 2 # times two due to MAC

    def layer_cycles(self):
        return self.layer_ops() / self.ops_per_cycle()

    def getMaxSIMD(self):
        wshape = self.getW().shape
        return wshape[1]

    def canIncreaseSIMD(self):
        return self.getSIMD() < self.getMaxSIMD()

    def getMaxPE(self):
        wshape = self.getW().shape
        return wshape[0]

    def canIncreasePE(self):
        return self.getPE() < self.getMaxPE()

    def getMaxMMV(self):
        # matrix layers (FC) can't do MMV by default
        # conv layer will override this
        return 1

    def canIncreaseMMV(self):
        return self.getMMV() < self.getMaxMMV()

    # implementation of StreamingComponentCodegen members
    def getNumInputElems(self):
        "Number of single input elements for this layer"
        return self.getW().shape[1]

    def getNumOutputElems(self):
        "Number of single output elements for this layer"
        return self.getW().shape[0]

    def codegen_params(self, output_dir):
        # sanity check: make sure PE/SIMD divides matrix size
        assert(self.getW().shape[0] % self.getPE() == 0)
        assert(self.getW().shape[1] % self.getSIMD() == 0)
        # generate code for weights
        packed_w = packWeightMatrix(self.getW(), self.getPE(), self.getSIMD())
        weights_hls = npyNDArrayToHLSArray(
            packed_w, self.getWMemDType(), self.getWMemName(),
            lambda x: binstr2hls(x, bitwidth=self.getSIMD(), reverse=True)
        )
        # emit into file
        with open(output_dir + "/" + self.getParamFileName(), "w") as f:
            f.write("\n\n\n//weights\n")
            f.write(weights_hls)

    def codegen_globals(self):
        return "\n#include \"%s\"" % self.getParamFileName()

    def codegen_declarations(self):
        ret="""
        #pragma HLS ARRAY_PARTITION variable=%s complete dim=1
        """ % (self.getWMemName())
        return ret

    def codegen_architecture(self):
        ret = ""
        if self.isBipolarTimesBipolar():
            # layer function call for BNN variant
            ret = "\nFCLayer_NoActivation_Batch<"
            ret += "%d, %d, " % (self.getInStreamW(), self.getOutStreamW())
            ret += "%d, %d, %d, " % (self.getSIMD(), self.getPE(), self.getAccWidth())
            ret += "%d, %d, " % self.getMatrixDims()
            ret += "%d>(" % (self.getWMemCount())
            ret += "%s, %s, " % (self.getIBufName(), self.getOBufName())
            ret += "%s, " % (self.getWMemName())
            ret += "numReps);"
        elif self.isBipolarTimesRegular():
            # layer function call for QNN variant
            ret += "\nMatrixVector_Precision_NoActivation_Batch<"
            ret += "%d, %d, " % (self.getSIMD(), self.getPE())
            ret += "%d, " % (self.getWBits())
            ret += "%d, %d, " % self.getMatrixDims()
            ret += "%d, " % (self.getWMemCount())
            ret += "%d, %d, " % (self.getIBits(), self.getOBits())
            # ActivationType is set to 0 -- no activation
            # TODO -- if input is signed, ap_uint here needs to change
            ret += "%d, %d, %s>(" % (self.getAccWidth(), 0, "ap_uint")
            ret += "%s, %s, " % (self.getIBufName(), self.getOBufName())
            dummy_thres_name = "dummythres%s" % self.getParamSuffix()
            ret += "%s, " % (self.getWMemName())
            ret += "numReps);"
        else:
            raise Exception("Unsupported case for code generation")
        return ret

    def codegen_single2instream(self, singleStrmName, inStrmName, dtype="T"):
        "Generate code to convert a single-element stream into this layer's input stream."
        ret = ""
        lastStrmName = singleStrmName
        # first convert into bipolar encoding, if needed
        # TODO checking ibits==1 is not enough to determine bipolar encoding
        if self.getIBits() == 1:
            encStrmName = "%s_bipolarenc" % self.getParamSuffix()
            ret += "\n// convert to bipolar encoding, mapping {-1, 1} -> {0, 1}"
            ret += "\nstream<%s> %s;" % (dtype, encStrmName)
            # 0.5x + 0.5 takes us from -1, +1 to 0, 1
            ret += "\nScaleShiftByConstant(%s, %s, %ff, %ff, %d);" % (singleStrmName, encStrmName, 0.5, 0.5, self.getNumInputElems())
            lastStrmName = encStrmName
        tmpName = "%s_iconvert" % self.getParamSuffix()
        ret += "\n// cast to single input element data type"
        ret += "\nstream<%s > %s;" % (self.getInputElemDType(), tmpName)
        ret += "\nCast(%s, %s, %d);" % (lastStrmName, tmpName, self.getNumInputElems())
        ret += "\n// convert to appropriate input stream width"
        ret += "\nDataWidthConverter<%d, %d, %d>(%s, %s);" % (self.getIBits(), self.getInStreamW(), self.getNumInputElems(), tmpName, inStrmName)
        return ret

    def codegen_outstream2single(self, outStrmName, singleStrmName, dtype="T"):
        "Generate code to convert this layer's output stream into a single-element stream."
        # here is the logic behind the operations here:
        # 1. unpack from transport bundle: ap_uint<group> to ap_uint<single>
        # 2. cast to single actual output element type: ap_int<single> to out_dtype
        # 3. cast to dtype
        # 4. add ScaleShiftByConstant if needed for bipolar decoding etc.

        # 1. extract single elements
        tmpName = "%s_oconvert" % self.getParamSuffix()
        ret = "\nstream<%s > %s;" % (self.getOutputElemDType(), tmpName)
        numOutputGroups = (self.getNumOutputElems() * self.getOBits()) / self.getOutStreamW()
        ret += "\nDataWidthConverter<%d, %d, %d>(%s, %s);" % (self.getOutStreamW(), self.getOBits(), numOutputGroups, outStrmName, tmpName)
        if self.getOBits() == 1:
            # TODO properly check for bipolar encoding on output
            # cast to dtype
            decStrmName = "%s_bipolardec" % self.getParamSuffix()
            ret += "\n// cast to dtype"
            ret += "\nstream<%s> %s;" % (dtype, decStrmName)
            ret += "\nCast(%s, %s, %d);" % (tmpName, decStrmName, self.getNumOutputElems())
            ret += "\n// decode bipolars, mapping {0, 1} -> {-1, +1}"
            # convert from bipolar to regular encoding
            # 2x -1 takes us from 0, 1 to -1, +1
            ret += "\nScaleShiftByConstant(%s, %s, %ff, %ff, %d);" % (decStrmName, singleStrmName, 2, -1, self.getNumOutputElems())
        elif self.isBipolarTimesBipolar():
            adjStrmName = "%s_negadj" % self.getParamSuffix()
            ret += "\n// cast to dtype"
            ret += "\nstream<%s> %s;" % (dtype, adjStrmName)
            # cast to float
            ret += "\nCast(%s, %s, %d);" % (tmpName, adjStrmName, self.getNumOutputElems())
            # compensate for lack of negative contributions in HLS bipolar implementation
            ret += "\n// compensate for lack of negative contributions in HLS bipolar impl."
            adj = -self.getNumInputElems()
            ret += "\nScaleShiftByConstant(%s, %s, %ff, %ff, %d);" % (adjStrmName, singleStrmName, 2, adj, self.getNumOutputElems())
        elif self.isBipolarTimesRegular() and self.isOSigned():
            # typecast to signed int, which is the output type
            intStrmName = "%s_signedint" % self.getParamSuffix()
            ret += "\n// cast to signed integers"
            ret += "\nstream<ap_int<%d> > %s;" % (self.getOBits(), intStrmName)
            ret += "\nCast(%s, %s, %d);" % (tmpName, intStrmName, self.getNumOutputElems())
            # cast to dtype
            ret += "\n// cast to dtype"
            ret += "\nCast(%s, %s, %d);" % (intStrmName, singleStrmName, self.getNumOutputElems())
        else:
            # just cast to dtype
            ret += "\n// cast to dtype"
            ret += "\nCast(%s, %s, %d);" % (tmpName, singleStrmName, self.getNumOutputElems())
        return ret

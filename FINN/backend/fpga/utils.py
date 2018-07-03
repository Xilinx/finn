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

# various utility functions for FPGA code generation

def isBipolarMatrix(W):
    # ensure W only contains -1 and +1 elements
    return np.logical_or(W == -1, W == +1).all()

def bipolarEncoding(W):
    # return the weight matrix using the bipolar encoding, where
    # -1 is represented by 0, and 1 is represented by 1. we can use
    # (x + 1) / 2 to map x {-1, +1} to x {0, 1}.
    return ((W+1) / 2).astype(np.int8)

def makePositiveThresholds(T, fanin):
    # the bipolar-times-bipolar implementation in FINN only returns the
    # popcount (e.g sum of +1s, and not the -1s). to compensate for this,
    # we adjust each threshold as Tnew = (Told+fanin)/2. this follows from
    # why? let p be the number of +1 bits and n be the number of -1 bits.
    # we know p+n=fanin, so we can express the actual sum p-n as p-(fanin-p)
    # = 2p-fanin. using this for the new threshold value, 2*Tnew-fanin=Told,
    # so Tnew=(Told+fanin)/2
    # NOTE: we cast back to float, do the add-and-divide, than take ceil() and
    # cast back to integer. ceil is the right way for the way we threshold --
    # just doing this on integers would round down and cause different results.
    Tnew = (T.astype(np.float32)+fanin) / 2
    # why the -1? this is due to a small mismatch between how the HLS library
    # implements thresholding (with >) versus how the ThresholdingLayer here does
    # it (with >=). by decreasing the threshold seen by HLS by 1, we get the
    # same behavior.
    Tnew = np.ceil(Tnew).astype(np.int16) - 1
    # ensure no negative thresholds, replacing those by fanin (no-activation)
    Tnew = np.asarray(map(lambda x: x if x >= 0 else fanin, Tnew.flatten()))
    return Tnew.reshape(T.shape)

def npyNDArrayToHLSArray(ndarray, hls_type_str, hls_var_name, elem2str):
    "Return C++ code representation of a numpy ndarray."
    ndims = len(ndarray.shape)
    # add type string and variable name
    # e.g. "const ap_uint<64>" "weightMem0"
    ret = "%s %s" % (hls_type_str, hls_var_name)
    # add dimensions
    for d in range(ndims):
        ret += "[%d]" % ndarray.shape[d]
    orig_printops = np.get_printoptions()
    np.set_printoptions(threshold=np.nan)
    strarr = np.array2string(ndarray, separator=", ", formatter={'all': elem2str})
    np.set_printoptions(**orig_printops)
    strarr = strarr.replace("[", "{").replace("]", "}")
    ret = ret + " = \n" + strarr + ";"
    return ret

def bin2str(bin_list):
    "Return a binary string corresponding to a list of binary integers."
    return reduce(lambda x,y: x+y, map(str, bin_list), "")

def binstr2hls(binstr, bitwidth, reverse=True):
    "Returns an HLS ap_uint<bitwidth> initializer for the given binary string."
    if reverse:
        binstr = binstr[::-1]
    #return "ap_uint<%d>(\"%s\", 2)" % (bitwidth, binstr)
    hexstr = hex(int(binstr, 2))
    if bitwidth <= 64:
        return hexstr
    else:
        # strip the 0x and L at the end
        hexstr = hexstr.replace("L", "").replace("0x", "")
        return "ap_uint<%d>(\"%s\", 16)" % (bitwidth, hexstr)

def packWeightMatrix(W, PE, SIMD):
    "Pack the weight matrix W into specified (PE, SIMD) size."
    # interleave rows between PEs using reshape + transpose
    Wr = W.reshape(-1, PE, W.shape[1]).transpose((1, 0, 2))
    Wr = Wr.reshape(PE, -1, SIMD)
    Wrb = np.apply_along_axis(bin2str, 2, Wr)
    return Wrb

def int2binstr(val, num_bits):
    "Returns a binary two's complement representation of val in given number of bits."
    ret = ""
    if val < 0:
        pos_offs = val + (1 << (num_bits-1))
        ret = bin(abs(pos_offs))[2:]
        ret = "1" + ("0" * (num_bits-len(ret)-1)) + ret
    else:
        ret = bin(abs(val))[2:] # convert abs to binary string, strip 0b
        ret = "0" * (num_bits - len(ret)) + ret
    return ret

def packMultiThresholds(thres, acc_bits):
    ret = []
    if thres.shape[0] * acc_bits > 64:
        raise Exception("Too many threshold bits to pack, need a more generic implementation")
    makeHex = lambda x: hex(int(int2binstr(x, acc_bits), 2))
    i = 0
    for t in thres:
        ret += "(%s << %d)" % (makeHex(t), acc_bits * i)
        i += 1
    return "|".join(ret);

def packThresholdMatrix(T, PE, acc_bits):
    "Pack the thresholds matrix T into specified (PE) size."
    if len(T.shape) != 2:
        raise Exception("Threshold array must be 2D: [thres_levels, channels]")
    nthres = T.shape[0]
    nchans = T.shape[1]
    if nchans % PE != 0:
        raise Exception("Number of channels not divisible by PE count.")
    if T.dtype != np.int16:
        raise Exception("Non-int16 thresholding not yet supported.")
    if nthres > 1:
        # move thres levels dimension innermost
        T = T.transpose()
        Tnew = []
        for ch in T:
            joined_thres = "".join(map(lambda x: int2binstr(x, acc_bits), ch))
            joined_thres = "ap_uint<%d>(\"%s\", 2)" % ((nthres+1) * acc_bits, joined_thres)
            Tnew += [joined_thres]

        T = np.asarray(Tnew)
    # interleave rows between PEs using reshape + transpose
    Tr = T.reshape(-1, PE).transpose()
    return Tr

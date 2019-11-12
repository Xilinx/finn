#   Copyright (c) 2019, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
import numpy as np
import os
import sys
import random 
import subprocess

outFileWeights = open("memdata.h" , "wt")
outFileConfig = open("config.h" , "wt")

kernel_dim = 3 
stride = 1
input_precision = 4
ifm_channels = 1
ofm_channels = 1
ifm_dimension = 8
ofm_dimension = 6

activation_precision = 16
expand = 1
simd = 1
pe = 1
w_precision = 1
mmv=2

tile = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

outFileConfig.write("#define KERNEL_DIM %d \n" % kernel_dim)
outFileConfig.write("#define SIMD1 %d \n" % simd)
outFileConfig.write("#define PE1 %d \n" % pe)
outFileConfig.write("#define MMV1 %d \n" % mmv)
outFileConfig.write("#define WIDTH %d \n" % w_precision)

outFileConfig.write("#define IFM_Channels1 %d \n" % ifm_channels)
outFileConfig.write("#define OFM_Channels1 %d \n" % ofm_channels)
outFileConfig.write("#define IFMDim1 %d \n" % ifm_dimension)
outFileConfig.write("#define OFMDim1 %d \n" % ofm_dimension)
outFileConfig.write("#define STRIDE %d \n" % stride)
outFileConfig.write("#define INPUT_PRECISION %d \n" % input_precision)
outFileConfig.write("#define TILE1 %d \n" % tile)

outFileConfig.write("#define ACTIVATION_PRECISION %d \n" % activation_precision)

outFileConfig.close()
outFileWeights.write("#ifndef PARAMS_HPP\n")
outFileWeights.write("#define PARAMS_HPP\n")

outFileWeights.write("namespace PARAM{ \n")
if (w_precision == 1):
	outFileWeights.write("static BinaryWeights<%d,%d,%d> weights= {\n{\n" %(simd,pe,tile))
else:
	outFileWeights.write("static FixedPointWeights<%d,ap_int<%d>,%d,%d> weights= {\n{\n" %(simd,w_precision,pe,tile))

for p in range(pe):
	outFileWeights.write("{ \n")
	for t in range(tile):
		width = simd*w_precision;
		val = random.randint(0, 1<<width-1)
		outFileWeights.write("%s" % hex(val))
		if t!=tile-1:
			outFileWeights.write(",\n")
	outFileWeights.write("} \n")
	if p!=pe-1:
		outFileWeights.write(",")


outFileWeights.write("}\n};\n } \n")
outFileWeights.write("#endif \n")
outFileWeights.close()





/* 
 Copyright (c) 2018, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the <organization> nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define AP_INT_MAX_W 4096
#include <hls_stream.h>
#include "rpnn-library.h"
#include "ap_int.h"
using namespace hls;
#include "dorefanet-pruned-config.h"

#include "params.h"

constexpr const unsigned int pad(unsigned int in, unsigned int padTo) {
    return (in % padTo == 0) ? in : (in + padTo - (in % padTo));
}


constexpr const unsigned int myMax(unsigned int v1, unsigned int v2) {
    return v1 >= v2 ? v1 : v2;
}


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}

const unsigned int inBits = 224*224*3*8; // 18816 64-bit words
const unsigned int inBitsPadded = 512*2352; 
const unsigned int inBytesPadded = inBitsPadded / 8; 

const unsigned int outBits = 6*6*256 * Precision;
const unsigned int outBitsPadded = 6*6*256 * Precision; 

const unsigned int outBytesPadded = outBitsPadded/8;
const unsigned int DataWidth=512;


void DoCompute(ap_uint<512> * in, ap_uint<512> * out, unsigned int numReps) {
#pragma HLS DATAFLOW
	
	stream<ap_uint<DataWidth> > memInStrm("memInStrm");

	stream<ap_uint<64> > inter0_1("inter0_1");
	stream<ap_uint<192> > inter0_big("inter0_big");
	stream<ap_uint<L0_SIMD*8> > inter0_input("inter0_input");

	stream<ap_uint<L0_OFMC * Precision> > inter1("inter1");
	stream<ap_uint<L1_IFMC * Precision> > inter1_upper("inter1_uppper"), inter1_lower("inter1_lower");
	stream<ap_uint<L1_OFMC * Precision> > inter2_upper("inter2_upper"), inter2_lower("inter2_lower");
	stream<ap_uint<L3_IFMC * Precision> > inter2("inter2"), inter3("inter3");
	stream<ap_uint<L3_OFMC * Precision> > inter4("inter4"), inter5("inter5");

	stream<ap_uint<L4_IFMC * Precision> > inter5_upper("inter5_upper"), inter5_lower("inter5_lower");
	stream<ap_uint<L4_OFMC * Precision> > inter6_upper("inter6_upper"), inter6_lower("inter6_lower");
	stream<ap_uint<L6_OFMC * Precision> > inter7_upper("inter7_upper"), inter7_lower("inter7_lower");
	stream<ap_uint<L6_OFMC * 2 * Precision> > inter7("inter7");
	stream<ap_uint<L6_OFMC * 2 * Precision> > inter7_0("inter7_0");

#pragma HLS STREAM variable=memInStrm depth=256

#pragma HLS STREAM variable=inter0_1 depth=8
#pragma HLS STREAM variable=inter0_big depth=8
#pragma HLS STREAM variable=inter0_input depth=8
#pragma HLS STREAM variable=inter1 depth=8
#pragma HLS STREAM variable=inter1_upper depth=2048
#pragma HLS STREAM variable=inter1_lower depth=2048
#pragma HLS STREAM variable=inter2 depth=256
#pragma HLS STREAM variable=inter2_upper depth=8
#pragma HLS STREAM variable=inter2_lower depth=8
#pragma HLS STREAM variable=inter3 depth=256
#pragma HLS STREAM variable=inter4 depth=32
#pragma HLS STREAM variable=inter5 depth=8
#pragma HLS STREAM variable=inter5_upper depth=32
#pragma HLS STREAM variable=inter5_lower depth=32
#pragma HLS STREAM variable=inter6_upper depth=2048
#pragma HLS STREAM variable=inter6_lower depth=2048
#pragma HLS STREAM variable=inter7 depth=8
#pragma HLS STREAM variable=inter7_0 depth=8
#pragma HLS STREAM variable=inter7_upper depth=8
#pragma HLS STREAM variable=inter7_lower depth=8	
	
	Mem2Stream_Batch<DataWidth, inBytesPadded>(in, memInStrm, numReps);

    DataWidthConverter_Batch<DataWidth, 64, inBitsPadded / DataWidth>(memInStrm, inter0_1, numReps);

    DataWidthConverter_Batch<64, 192, inBitsPadded/64>(inter0_1, inter0_big, numReps);

    DataWidthConverter_Batch<192, L0_SIMD*8, inBitsPadded/192>(inter0_big, inter0_input, numReps);
	
	// Conv0
	ConvolutionalLayerMMV_Valid_Batch_dsp<
		L0_KERNELDIM, L0_IFMC, 224, L0_OFMC, 4,
		L0_SIMD, L0_PE, L0_WMEM, L0_TMEM,
		8, 20 * 4, 20, 8, 2, L0_MMV, FULL_THRESHOLDS, ap_int >(inter0_input, inter1, weightMem0, threshMem0, numReps);	
		
	Splitter_Batch<L0_OFMC*Precision, L1_IFMC*Precision, 54*54>(inter1, inter1_upper, inter1_lower, numReps);

	// Conv1
	ConvolutionalLayerMMV_Same_Batch<
		L1_KERNELDIM, L1_IFMC, 54, L1_OFMC, 1,
		L1_SIMD, L1_PE, L1_WMEM, L1_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L1_MMV, FULL_THRESHOLDS>(inter1_upper, inter2_upper, weightMem1, threshMem1, numReps);	
	ConvolutionalLayerMMV_Same_Batch<
		L2_KERNELDIM, L2_IFMC, 54, L2_OFMC, 1,
		L2_SIMD, L2_PE, L2_WMEM, L2_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L2_MMV, FULL_THRESHOLDS>(inter1_lower, inter2_lower, weightMem2, threshMem2, numReps);

	Merger_Batch<L2_OFMC*Precision, L3_IFMC*Precision, 54*54>(inter2_lower, inter2_upper, inter2, numReps);

	// Pool1
	MaxPoolStride_Same_Batch<54, 3, 2, L3_IFMC, Precision>(inter2, inter3, numReps);

	// Conv2
	ConvolutionalLayerMMV_Same_Batch<
		L3_KERNELDIM, L3_IFMC, 27, L3_OFMC, 1,
		L3_SIMD, L3_PE, L3_WMEM, L3_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L3_MMV, FULL_THRESHOLDS>(inter3, inter4, weightMem3, threshMem3, numReps);	

	// Pool2	
	MaxPoolStride_Same_Batch<27, 3, 2, L3_OFMC, Precision>(inter4, inter5, numReps);

	Splitter_Batch<L3_OFMC * Precision, L4_IFMC * Precision, 14*14>(inter5, inter5_upper, inter5_lower, numReps);

	// Conv3
    ConvolutionalLayerMMV_Same_Batch<
		L4_KERNELDIM, L4_IFMC, 14, L4_OFMC, 1,
		L4_SIMD, L4_PE, L4_WMEM, L4_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L4_MMV, FULL_THRESHOLDS>(inter5_upper, inter6_upper, weightMem4, threshMem4, numReps);
    ConvolutionalLayerMMV_Same_Batch<
		L5_KERNELDIM, L5_IFMC, 14, L5_OFMC, 1,
		L5_SIMD, L5_PE, L5_WMEM, L5_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L5_MMV, FULL_THRESHOLDS>(inter5_lower, inter6_lower, weightMem5, threshMem5, numReps);
	
	// Conv4	
	ConvolutionalLayerMMV_Same_Batch<
		L6_KERNELDIM, L6_IFMC, 14, L6_OFMC, 1,
		L6_SIMD, L6_PE, L6_WMEM, L6_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L6_MMV, FULL_THRESHOLDS>(inter6_upper, inter7_upper, weightMem6, threshMem6, numReps);
	ConvolutionalLayerMMV_Same_Batch<
		L7_KERNELDIM, L7_IFMC, 14, L7_OFMC, 1,
		L7_SIMD, L7_PE, L7_WMEM, L7_TMEM,
		1, ThresholdPrecision * 4, 16, 2, 2, L7_MMV, FULL_THRESHOLDS>(inter6_lower, inter7_lower, weightMem7, threshMem7, numReps);

	Merger_Batch<L7_OFMC * Precision, L7_OFMC * 2 * Precision, 14*14>(inter7_lower, inter7_upper, inter7, numReps);

	// Pool4
	MaxPoolStride_Valid_Batch<14, 3, 2, L7_OFMC * 2, Precision>(inter7, inter7_0, numReps);

	Stream2Mem_Batch<DataWidth, outBytesPadded>(inter7_0, out, numReps);	
}


#ifdef __SDX__
	extern "C" {
#endif
void opencldesign_wrapper(ap_uint<512> * in, ap_uint<512> * out, bool doInit, unsigned int numReps) {
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=gmem depth=4096
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=gmem depth=36
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weightMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshMem7 complete dim=1


if (!doInit) {
    DoCompute(in, out, numReps);
  }
	
}
#ifdef __SDX__
	}
#endif

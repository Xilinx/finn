
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

#define AP_INT_MAX_W 2048
#include "bnn-library.h"
#include "config.h"

#include "memdata-0.h"
#include "memdata-1.h"
#include "memdata-2.h"
#include "memdata-3.h"
#include "memdata-4.h"
#include "memdata-5.h"
#include "memdata-6.h"
#include "memdata-7.h"
#include "memdata-8.h"

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}


void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	stream<ap_uint<64> > inter0("DoCompute.inter0");
	stream<ap_uint<192> > inter0_1("DoCompute.inter0_1");
	stream<ap_uint<24> > inter0_2("DoCompute.inter0_2");
	stream<ap_uint<128> > inter1("DoCompute.inter1");
	stream<ap_uint<128> > inter2("DoCompute.inter2");
	stream<ap_uint<128> > inter3("DoCompute.inter3");
	stream<ap_uint<256> > inter4("DoCompute.inter4");
	stream<ap_uint<256> > inter5("DoCompute.inter5");
	stream<ap_uint<256> > inter6("DoCompute.inter6");
	stream<ap_uint<512> > inter7("DoCompute.inter7");
	stream<ap_uint<512> > inter8("DoCompute.inter8");
  stream<ap_uint<512> > inter9("DoCompute.inter9");
  // TODO recalculate depths for FC streams
#pragma HLS STREAM variable=inter9 depth=512
  stream<ap_uint<L6_PE> > inter10("DoCompute.inter10");
#pragma HLS STREAM variable=inter10 depth=512
	stream<ap_uint<L7_PE> > inter11("DoCompute.inter11");
#pragma HLS STREAM variable=inter11 depth=3
	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");



	const unsigned int inBits = 32*32*3*8;
	//const unsigned int inBitsPadded = paddedSize(inBits, 64);
	const unsigned int outBits = 12*16;
#define STRIDE 1
#define PAD 1
	Mem2Stream_Batch<64, inBits/8>(in, inter0, numReps);
    DataWidthConverter_Batch<64, 192, (32*32*3*8) / 64>(inter0, inter0_1, numReps);
    DataWidthConverter_Batch<192, 24, (32*32*3*8) / 192>(inter0_1, inter0_2, numReps);
	ConvLayerMMV_Fxd_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, STRIDE, 8, 1, L0_SIMD, L0_PE, 24, 16, L0_WMEM, L0_TMEM, PAD, L0_MMV, 3>(inter0_2, inter1, weightMem0, thresMem0, numReps);
	ConvLayerMMV_BNN_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, STRIDE, L1_SIMD, L1_PE, 16, L1_WMEM, L1_TMEM, PAD, L1_MMV, 3>(inter1, inter2, weightMem1, thresMem1, numReps);
	MaxPool_BNN_Batch<L1_OFM_DIM, 2, L1_OFM_CH>(inter2, inter3, numReps);
	ConvLayerMMV_BNN_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, STRIDE, L2_SIMD, L2_PE, 16, L2_WMEM, L2_TMEM, PAD, L2_MMV>(inter3, inter4, weightMem2, thresMem2, numReps);
	ConvLayerMMV_BNN_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, STRIDE, L3_SIMD, L3_PE, 16, L3_WMEM, L3_TMEM, PAD, L3_MMV>(inter4, inter5, weightMem3, thresMem3, numReps);
	MaxPool_BNN_Batch<L3_OFM_DIM, 2, L3_OFM_CH>(inter5, inter6, numReps);
	ConvLayerMMV_BNN_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, STRIDE, L4_SIMD, L4_PE, 16, L4_WMEM, L4_TMEM, PAD, L4_MMV>(inter6, inter7, weightMem4, thresMem4, numReps);
	ConvLayerMMV_BNN_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM, STRIDE, L5_SIMD, L5_PE, 16, L5_WMEM, L5_TMEM, PAD, L5_MMV>(inter7, inter8, weightMem5, thresMem5, numReps);
    MaxPool_BNN_Batch<L5_OFM_DIM, 2, L5_OFM_CH>(inter8, inter9, numReps);

  // fully connected layers
    FCLayer_Batch<512, L6_PE, L6_SIMD, L6_PE, 16, L6_MW, L6_MH, L6_WMEM, L6_TMEM>(inter9, inter10, weightMem6, thresMem6, numReps);
    FCLayer_Batch<L6_PE, L7_PE, L7_SIMD, L7_PE, 16, L7_MW, L7_MH, L7_WMEM, L7_TMEM>(inter10, inter11, weightMem7, thresMem7, numReps);

    FCLayer_NoActivation_Batch<L7_PE, 64, L8_SIMD, L8_PE, 16, L8_MW, L8_MH, L8_WMEM>(inter11, memOutStrm, weightMem8, numReps);

	Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
}

#ifdef __SDX__
	extern "C" {
#endif
void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weightMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem8 complete dim=1

// specifically assign resource types for some PE arrays
#pragma HLS RESOURCE variable=weightMem1 core=RAM_1P_LUTRAM

	if (!doInit) {
		DoCompute(in, out, numReps);
	}
}
#ifdef __SDX__
	}
#endif


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

#include "bnn-library.h"
#include "config.h"

#include "memdata-0.h"
#include "memdata-1.h"
#include "memdata-2.h"
#include "memdata-3.h"


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}


void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
	stream<ap_uint<64> > memInStrm;
	stream<ap_uint<L0_PE> > inter0;
	stream<ap_uint<L1_PE> > inter1;
	stream<ap_uint<L2_PE> > inter2;
	stream<ap_uint<16> > memOutStrmSmall;
	stream<ap_uint<64> > memOutStrm;
	
#pragma HLS DATAFLOW

#pragma HLS stream depth=1024 variable=memInStrm     // write: 1, read: 1 but mask memory latency
#pragma HLS stream depth=2 variable=inter0          // write: 4/13, read: 1/4 - bottleneck
#pragma HLS stream depth=2 variable=inter1          // write: 1/4, read: 1/4 - minimal FIFO
#pragma HLS stream depth=2 variable=inter2          // write: 1/4, read: 1/4 - minimal FIFO
#pragma HLS stream depth=2 variable=memOutStrmSmall // II=1 - minimal FIFO
#pragma HLS stream depth=1024 variable=memOutStrm    // mask memory latency

	const unsigned int inBits = 28*28;
	const unsigned int inBitsPadded = 832; // paddedSizeHW(inBits, 64)
	const unsigned int inBytesPadded = inBitsPadded/8;
	const unsigned int outBits = 64;
	const unsigned int outBitsPadded = 64; // paddedSizeHW(outBits, 64)
	const unsigned int outBytesPadded = outBitsPadded/8;
	const unsigned int inWordsPerImg = inBitsPadded / 64;
	const unsigned int outWordsPerImg = outBitsPadded / 64;

	Mem2Stream_Batch<64, inBytesPadded>(in, memInStrm, numReps);
  FCLayer_Batch<64, L0_PE, L0_SIMD, L0_PE, 16, L0_MW, L0_MH, L0_WMEM, L0_TMEM>(
		  memInStrm, inter0, weightMem0, thresMem0, numReps);

  FCLayer_Batch<L0_PE, L1_PE, L1_SIMD, L1_PE, 16, L1_MW, L1_MH, L1_WMEM, L1_TMEM>(
		  inter0, inter1, weightMem1, thresMem1, numReps);

  FCLayer_Batch<L1_PE, L2_PE, L2_SIMD, L2_PE, 16, L2_MW, L2_MH, L2_WMEM, L2_TMEM>(
		  inter1, inter2, weightMem2, thresMem2, numReps);

  FCLayer_Batch<L2_PE, 16, L3_SIMD, L3_PE, 16, L3_MW, L3_MH, L3_WMEM, L3_TMEM>(
		  inter2, memOutStrmSmall, weightMem3, thresMem3, numReps);
  Cast<ap_uint<16>, ap_uint<64> >(memOutStrmSmall, memOutStrm, numReps);
  Stream2Mem_Batch<64, outBytesPadded>(memOutStrm, out, numReps);
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

	if (!doInit) {
		DoCompute(in, out, numReps);
	}
}
#ifdef __SDX__
	}
#endif

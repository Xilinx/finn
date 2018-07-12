#pragma once

#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
using namespace hls;
using namespace std;

#define CASSERT_DATAFLOW(x) ;

// essentially small DMA generators, moving data between mem-mapped arrays and streams
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream(ap_uint<DataWidth> * in, stream<ap_uint<DataWidth> > & out) {
	CASSERT_DATAFLOW(DataWidth % 8 == 0);
	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++) {
		ap_uint<DataWidth> e = in[i];
		out.write(e);
	}

}

template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch_external_wmem(ap_uint<DataWidth> * in,
        stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
    unsigned int rep = 0;
    while (rep != numReps) {
        Mem2Stream<DataWidth, numBytes>(&in[0], out);
        rep += 1;
    }
}

template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out) {
	CASSERT_DATAFLOW(DataWidth % 8 == 0);

	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++) {
		ap_uint<DataWidth> e = in.read();
		out[i] = e;
	}
}

// call different statically-sized variants of Mem2Stream and Stream2Mem without burst 
// This is done since the donut used in CAPI-based design has small FIFO shared among 
// write and read buffer, causing stall in the offload if FIFO is busy constantly reading
// (can be avoided by sizing input stream as "big enough" FIFO)
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch(ap_uint<DataWidth> * in,
		hls::stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
	const unsigned int indsPerRep = numBytes / (DataWidth / 8);
	unsigned int rep = 0;
	while (rep != numReps) {
		Mem2Stream<DataWidth, numBytes>(&in[rep * indsPerRep], out);
		rep += 1;
	}
}

template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem_Batch(hls::stream<ap_uint<DataWidth> > & in,
		ap_uint<DataWidth> * out, const unsigned int numReps) {
	const unsigned int indsPerRep = numBytes / (DataWidth / 8);
	unsigned int rep = 0;
	while (rep != numReps) {
		Stream2Mem<DataWidth, numBytes>(in, &out[rep * indsPerRep]);
		rep += 1;
	}
}
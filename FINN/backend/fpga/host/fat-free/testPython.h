#pragma once

#include "ap_int.h"

template<typename T>
T * getArrayData(numeric::array &arr){
	return (T *) ((PyArrayObject *) arr.ptr())->data;
}

// Defined in network
//void opencldesign_wrapper(ap_uint<512> * in, ap_uint<512> * out, bool doInit, unsigned int numReps);
void DoCompute(hls::stream<ap_uint<256> > & in, hls::stream<ap_uint<256> > & out);

void getMemory(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd);
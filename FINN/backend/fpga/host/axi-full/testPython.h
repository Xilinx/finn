#pragma once

#include "ap_int.h"

template<typename T>
T * getArrayData(numeric::array &arr){
	return (T *) ((PyArrayObject *) arr.ptr())->data;
}

// Defined in network
#ifdef __SDX__
	extern "C" {
#endif
void opencldesign_wrapper(ap_uint<512> * in, ap_uint<512> * out, bool doInit, 
	unsigned int numReps);
#ifdef __SDX__
	}
#endif

void getMemory(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd);

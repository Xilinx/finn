#pragma once

#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
#include "utils.h"
using namespace hls;
using namespace std;

#define CASSERT_DATAFLOW(x) ;

template<
	unsigned int ImgDim, 
	unsigned int KernelDim, 
	unsigned int Stride, 
	unsigned int NumChannels,
	unsigned int Precision, 
	unsigned int IsPaddedSame=0>
void MaxPool_InputGenerator_Batch(stream<ap_uint<NumChannels * Precision> > &in, 
		stream<ap_uint<NumChannels * Precision> > & out, 
		const unsigned int numReps){
	
	// Number of rows and cols that overlap among two consecutives windowing of the input
	constexpr unsigned int overlap = KernelDim - Stride;

	// Number of output windows
	// Note that all the operations are among unsigned int (i.e. divisions are floored)
	constexpr unsigned int outputWindows = 1 + (ImgDim - KernelDim) / Stride;
	
	// Number of output elements per dimension
	constexpr unsigned int outputElements = outputWindows * KernelDim;

	// Buffer to store data to reuse (i.e. output at a different time than when they are read)
	ap_uint<NumChannels * Precision> windowBuffer[ImgDim][ImgDim];
	
	unsigned int rowCount, colCount;
	unsigned int inX, inY;
	ap_uint<NumChannels * Precision> dataIn;
	ap_uint<NumChannels * Precision> dataOut;
		
	for(unsigned int rep=0; rep<numReps; rep++)
	{
		rowCount = 0;
		inY = 0;
		// Cycle on Output Data to Generate
		for(unsigned int outY = 0; outY < outputElements; outY++){
			for(unsigned int outX = 0; outX < outputElements; outX++){
	#pragma HLS PIPELINE II=1				
				if(outX==0){
					colCount = 0;
					inX = 0;
				}
				
				unsigned int increaseY = 0;

				// Normal Condition, Reading from input
				if(colCount < KernelDim && rowCount < KernelDim){
					dataIn = in.read();
					windowBuffer[inY][inX] = dataIn;
					dataOut = dataIn;
					inX += 1;

					// Shifting only when I am consuming from input
					increaseY = 1;
				}

				// Repeat Col Condition
				else if(colCount >= KernelDim && rowCount < KernelDim){
					dataOut = windowBuffer[inY][inX - overlap + colCount - KernelDim];
				}

				// Repeat Row only Condition
				else if(rowCount >= KernelDim && colCount < KernelDim){
					dataOut = windowBuffer[inY - overlap + rowCount - KernelDim][inX];
					inX += 1;
				}

				// Repead Row and Col Condition
				else if(rowCount >= KernelDim && colCount >= KernelDim){
					dataOut = windowBuffer[inY - overlap + rowCount - KernelDim][inX - overlap + colCount - KernelDim];
				}

				// TODO: This logic here may refrain from synthesizing at 200MHz
				colCount += 1;

				if(colCount == KernelDim + overlap){
					colCount = overlap;
				}
				
				// Increment count of number of processed output block rows
				if(outX == outputElements - 1){
					if(rowCount + 1 == KernelDim + overlap)
						rowCount = overlap;
					else
						rowCount += 1;
					
					if(increaseY)
						inY+=1;
				}

				// Output current Data
				out.write(dataOut);
			}
		}
	}
}

template<unsigned int ImgDim, 
unsigned int PoolDim, 
unsigned int NumChannels, 
unsigned int Precision>
void MaxPool_ReducedPrecision_Batch(stream<ap_uint<NumChannels * Precision> > & in,
		stream<ap_uint<NumChannels * Precision> > & out, 
		const unsigned int numReps) {
	CASSERT_DATAFLOW(ImgDim % PoolDim == 0);

	// need buffer space for a single maxpooled row of the image
	ap_uint<Precision> buf[ImgDim / PoolDim][NumChannels];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=2

	for (unsigned int rep=0; rep < numReps; rep++)
	{
		for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
			for(unsigned int ch = 0; ch<NumChannels; ch++){
	#pragma HLS UNROLL
			  buf[i][ch] = 0;
			}
		}

		ap_uint<NumChannels * Precision> inputData;	
		ap_uint<NumChannels * Precision> outputData;


		for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
			for (unsigned int ky = 0; ky < PoolDim; ky++) {
				for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
			
					// Change to comparator	
					for (unsigned int kx = 0; kx < PoolDim; kx++) {
	#pragma HLS PIPELINE II=1
						inputData = in.read();
						for(unsigned int ch = 0; ch<NumChannels; ch++){
	#pragma HLS UNROLL						
							unsigned int lowBit = ch * Precision;
							unsigned int highBit = (ch+1) * Precision -1;
							ap_uint<Precision> channeldata = inputData(highBit, lowBit);
							
							ap_uint<Precision> oldMax = buf[xp][ch];
							
							if(channeldata > oldMax){
								buf[xp][ch] = channeldata;
							}
						}
					}
				}
			}

			for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
				for(unsigned int ch = 0; ch < NumChannels; ch++){
	#pragma HLS UNROLL
					
					unsigned int lowBit = ch * Precision;
					unsigned int highBit = (ch+1) * Precision -1;	
					outputData(highBit, lowBit) = buf[outpix][ch];

					// get buffer ready for next use
					buf[outpix][ch] = 0;
				}
				out.write(outputData);
			}
		}
	}
}

template<	unsigned int ImgDim, 
			unsigned int KernelDim, 
			unsigned int Stride, 
			unsigned int NumChannels,
			unsigned int Precision>
void MaxPoolStride_Same_Batch(stream<ap_uint<NumChannels * Precision> > & in, 
		stream<ap_uint<NumChannels * Precision> > & out, const unsigned int numReps){
#pragma HLS INLINE

	stream<ap_uint<NumChannels * Precision> > paddingOut, resizeOut;

	// Number of output windows
	constexpr unsigned int outputWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);

	// Output dimensions of the resize stage
	constexpr unsigned int resizeOutputDim = KernelDim + Stride * (outputWindows - 1);

	// Number of output elements per dimension (of padder + resize components)
	constexpr unsigned int ImgDimOut = outputWindows * KernelDim;

	SameResize_Batch<ImgDim, KernelDim, Stride, NumChannels, Precision>(in, paddingOut, numReps);
	MaxPool_InputGenerator_Batch<resizeOutputDim, KernelDim, Stride, NumChannels, Precision, 1>(paddingOut, resizeOut, numReps);
	MaxPool_ReducedPrecision_Batch<ImgDimOut, KernelDim, NumChannels, Precision>(resizeOut, out, numReps);

}


template<	unsigned int ImgDim, 
			unsigned int KernelDim, 
			unsigned int Stride, 
			unsigned int NumChannels,
			unsigned int Precision>
void MaxPoolStride_Valid_Batch(stream<ap_uint<NumChannels * Precision> > & in, 
		stream<ap_uint<NumChannels * Precision> > & out, const unsigned int numReps){
#pragma HLS INLINE

	stream<ap_uint<NumChannels * Precision> > paddingOut, resizeOut;

	// Number of output windows
	constexpr unsigned int outputWindows = 1 + (ImgDim - KernelDim)/Stride;

	// Number of output elements per dimension (of padder + resize components)
	constexpr unsigned int ImgDimOut = outputWindows * KernelDim;

	ValidResize_Batch<ImgDim, KernelDim, Stride, NumChannels, Precision>(in, paddingOut, numReps);
	MaxPool_InputGenerator_Batch<ImgDim, KernelDim, Stride, NumChannels, Precision>(paddingOut, resizeOut, numReps);
	MaxPool_ReducedPrecision_Batch<ImgDimOut, KernelDim, NumChannels, Precision>(resizeOut, out, numReps);
}


// TODOs:
// - double buffer to sustain full throughput
// - add a StreamPadder to handle ImgDim % PoolDim != 0 cases
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void MaxPool_BNN(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out) {
	CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
	// need buffer space for a single maxpooled row of the image
	ap_uint<NumChannels> buf[ImgDim / PoolDim];
	for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
#pragma HLS UNROLL
	  buf[i] = 0;
	}

	for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
		for (unsigned int ky = 0; ky < PoolDim; ky++) {
			for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
#pragma HLS PIPELINE II=1
				ap_uint<NumChannels> acc = 0;
				for (unsigned int kx = 0; kx < PoolDim; kx++) {
					acc = acc | in.read();
				}
				// pool with old value in row buffer
				buf[xp] |= acc;
			}
		}

		for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
#pragma HLS PIPELINE II=1
			out.write(buf[outpix]);
			// get buffer ready for next use
			buf[outpix] = 0;
		}
	}

}

// calling 1-image maxpool in a loop works well enough for now
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void MaxPool_BNN_Batch(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out, unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		MaxPool_BNN<ImgDim, PoolDim, NumChannels>(in, out);
	}
}
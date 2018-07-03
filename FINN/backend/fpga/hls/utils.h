#pragma once

#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
using namespace hls;
using namespace std;

#define CASSERT_DATAFLOW(x) ;

template <typename T>
void ScaleShiftByConstant(stream<T> & in, stream<T> & out, T scale, T shift, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write(scale * in.read() + shift);
  }
}

template <typename T, typename F>
void ScaleShiftByConstant(stream<T> & in, stream<T> & out, F scale, F shift, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write(scale * in.read() + shift);
  }
}

template<typename InT, typename OutT>
void Cast(stream<InT> & in, stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}

template<typename InT, typename OutT>
void BitwiseConvert(stream<InT> & in, stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    InT inElem = in.read();
    OutT outElem = *(reinterpret_cast<OutT*>(&inElem));
    out.write(outElem);
  }
}

// only let the first X elements of a stream to pass through, the remainder
// are consumed from input but not re-emitted from the output
// useful for getting rid of e.g. padding words
template<unsigned int DataWidth,		// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal // total number of words (NumTotal-NumAllowed swallowed)
>
void Limiter(stream<ap_uint<DataWidth> > & in,
		stream<ap_uint<DataWidth> > & out) {
	CASSERT_DATAFLOW(NumTotal >= NumAllowed);
	unsigned int numLeft = NumAllowed;
	for (unsigned int i = 0; i < NumTotal; i++) {
		ap_uint<DataWidth> e = in.read();
		if (numLeft > 0) {
			out.write(e);
			numLeft--;
		}
	}
}

// simple batched function variants with just a loop around the base function call
// this works well when the base function is simple enough, such that HLS is able to
// merge the loops into a single iteration space and pipeline everything with II=1
template<unsigned int DataWidth,		// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal // total number of words (NumTotal-NumAllowed swallowed)
>
void Limiter_Batch(stream<ap_uint<DataWidth> > & in,
		stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {

	for (unsigned int rep = 0; rep < numReps; rep++) {
		Limiter<DataWidth, NumAllowed, NumTotal>(in, out);
	}
}

template<unsigned int DataWidth, unsigned int NumNormal, unsigned int NumExtra>
void Filler(stream<ap_uint<DataWidth> > & in, stream<ap_uint<DataWidth> > & out) {
        for(unsigned int i = 0; i<NumNormal; i++) {
                ap_uint<DataWidth> e = in.read();
                out.write(e);
        }

        for (unsigned int i = 0; i < NumExtra; i++) {
                out.write(0);
        }
}

// simple batched function with just a loop around the base function call
template<unsigned int DataWidth, unsigned int NumNormal, unsigned int NumExtra>
void Filler_Batch(stream<ap_uint<DataWidth> > & in, stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		Filler<DataWidth, NumNormal, NumExtra>(in, out);
	}
}

template<unsigned int DataWidth, unsigned int Number_streams>
void Interleaver(stream<ap_uint<DataWidth> > & in, stream<ap_uint<DataWidth> > & out) {
        for(unsigned int i = 0; i<Number_streams; i++) {
                ap_uint<DataWidth> e = in.read();
                out.write(e);
				e=0;
				out.write(e);
        }
}




// Reshape input stream to output by adding padding values on top, bottom left and rigth
template<
	unsigned int ImgDim,
	unsigned int KernelDim,
	unsigned int NumChannels,
	unsigned int Precision,
	unsigned int Pad>
void Padding(stream<ap_uint<NumChannels * Precision> > &in,
		stream<ap_uint<NumChannels * Precision> > & out){


	ap_uint<NumChannels* Precision> outData, inData;

	for(unsigned int y = 0; y<ImgDim+2*Pad; y++){
		for(unsigned int x=0; x < ImgDim+2*Pad; x++){
#pragma HLS PIPELINE II=1

			// Padding Rows
			if(y < Pad || y >= (ImgDim+ Pad)){
				outData = 0;
			}
			// Padding Cols
			else if(x < Pad || x >= (ImgDim+ Pad)){
				outData = 0;
			}
			// No Padding
			else{
				inData = in.read();
				outData = inData;
			}

			out.write(outData);
		}
	}
}



template<
	unsigned int ImgDim,
	unsigned int KernelDim,
	unsigned int NumChannels,
	unsigned int Precision,
	unsigned int Pad>
void Padding_Batch(stream<ap_uint<NumChannels* Precision> > &in,
		stream<ap_uint<NumChannels* Precision> > &out,
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		Padding<ImgDim, KernelDim, NumChannels, Precision, Pad>(in, out);
	}

}


// convert a stream of bits of given word width to another width
// the greater width must be evenly divisable by the smaller width
template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void DataWidthConverter(stream<ap_uint<InWidth> > & in,
		stream<ap_uint<OutWidth> > & out) {
	if (InWidth > OutWidth) {
		// emit multiple output words per input word read
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		for (unsigned int i = 0; i < NumInWords; i++) {
			ap_uint<InWidth> ei = in.read();
			for (unsigned int o = 0; o < outPerIn; o++) {
				ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
				out.write(eo);
				ei = ei >> OutWidth;
			}
		}
	} else if (InWidth == OutWidth) {
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords; i++) {
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	} else { // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		const unsigned int inPerOut = OutWidth / InWidth;
		for (unsigned int o = 0; o < NumInWords / inPerOut; o++) {
			ap_uint<OutWidth> eo = 0;
			for (unsigned int i = 0; i < inPerOut; i++) {
				ap_uint<InWidth> ei = in.read();
				eo = eo >> InWidth;
				eo(OutWidth - 1, OutWidth - InWidth) = ei;
			}
			out.write(eo);
		}
	}
}

// Reshape input stream to output only useful data when padding is VALID:
// Might drop lines and columns at right and bottom
template<
	unsigned int ImgDim,
	unsigned int KernelDim,
	unsigned int Stride,
	unsigned int NumChannels,
	unsigned int Precision>
void ValidResize(stream<ap_uint<NumChannels * Precision> > &in,
		stream<ap_uint<NumChannels * Precision> > & out){

	// Cols and Rows to drop when Padding is Valid
	// Note that all the operations are among unsigned int (i.e. divisions are floored)
	constexpr unsigned int drop = ImgDim - (KernelDim + (ImgDim - KernelDim)/Stride * Stride);

	// Last valid Row/Col of the input data, everything else past this value has to be dropped
	constexpr unsigned int dropAt = ImgDim - drop;

	for(unsigned int i=0; i<dropAt; i++){
		for(unsigned int j=0; j<ImgDim; j++){
#pragma HLS PIPELINE II=1
			ap_uint<NumChannels * Precision> data = in.read();

			if(j < dropAt)
				out.write(data);
		}
	}

	// Consuming last lines to drop
	for(unsigned int i = 0; i<drop; i++){
		for(unsigned int j=0; j<ImgDim; j++){
#pragma HLS PIPELINE II=1
			in.read();
		}
	}
}


template<
	unsigned int ImgDim,
	unsigned int KernelDim,
	unsigned int Stride,
	unsigned int NumChannels,
	unsigned int Precision>
void ValidResize_Batch(stream<ap_uint<NumChannels * Precision> > &in,
		stream<ap_uint<NumChannels * Precision> > & out,
		const unsigned int numReps){
	for (unsigned int rep = 0; rep < numReps; rep++) {
		ValidResize<ImgDim, KernelDim, Stride, NumChannels, Precision>(in, out);
	}
}



// Reshape input stream to output only useful data when padding is same:
// Might add 0s at left, right, upper, lower side of the input
// Pad with 0
template<	unsigned int ImgDim,
			unsigned int KernelDim,
			unsigned int Stride,
			unsigned int NumChannels,
			unsigned int Precision>
void SameResize(stream<ap_uint<NumChannels* Precision> > &in,
		stream<ap_uint<NumChannels* Precision> > &out){

	// Number of "same" windows over the input data
	constexpr unsigned int SameWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);

	// Number of elements to generate as output per dimension
	constexpr unsigned int OutputDim = KernelDim + Stride * (SameWindows - 1);

	// Padding
	constexpr unsigned int Padding = OutputDim - ImgDim;

	// Padding Up and Left
	constexpr unsigned int PaddingUp = Padding/2;
	constexpr unsigned int PaddingLeft = Padding/2;

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;

	ap_uint<NumChannels* Precision> outData, inData;

	for(unsigned int y = 0; y<OutputDim; y++){
		for(unsigned int x=0; x < OutputDim; x++){
#pragma HLS PIPELINE II=1

			// Padding Rows
			if(y < PaddingUp || y >= (OutputDim - PaddingDown)){
				outData = 0;
			}
			// Padding Cols
			else if(x < PaddingLeft || x >= (OutputDim - PaddingRight)){
				outData = 0;
			}
			// No Padding
			else{
				inData = in.read();
				outData = inData;
			}

			out.write(outData);
		}
	}
}


template<	unsigned int ImgDim,
			unsigned int KernelDim,
			unsigned int Stride,
			unsigned int NumChannels,
			unsigned int Precision>
void SameResize_Batch(stream<ap_uint<NumChannels* Precision> > &in,
		stream<ap_uint<NumChannels* Precision> > &out,
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		SameResize<ImgDim, KernelDim, Stride, NumChannels, Precision>(in, out);
	}

}


// Split the input channels in two different streams
template<unsigned int DataWidthIn,		// input stream width
		unsigned int DataWidthOut,	// output stream width
		unsigned int NumTotal // total number of words
>
void Splitter(stream<ap_uint<DataWidthIn> > & in,
		stream<ap_uint<DataWidthOut> > & out1, stream<ap_uint<DataWidthOut> > & out2) {

	CASSERT_DATAFLOW(DataWidthIn == DataWidthOut * 2);

	for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidthIn> e = in.read();

		ap_uint<DataWidthOut> e1 = e(DataWidthOut-1, 0);
		ap_uint<DataWidthOut> e2 = e(DataWidthIn-1, DataWidthOut);

		out1.write(e1);
		out2.write(e2);

	}
}


// simple batched function with just a loop around the base function call
template<unsigned int DataWidthIn,		// input stream width
		unsigned int DataWidthOut,	// output stream width
		unsigned int NumTotal // total number of words
>
void Splitter_Batch(stream<ap_uint<DataWidthIn> > & in,
		stream<ap_uint<DataWidthOut> > & out1, stream<ap_uint<DataWidthOut> > & out2,
		const unsigned int numReps) {

	for(unsigned int rep=0; rep<numReps; rep++){
		Splitter<DataWidthIn, DataWidthOut, NumTotal>(in, out1, out2);
	}
}

// Replicate the input channels to two different streams
template<unsigned int DataWidthIn,		// input stream width
		unsigned int NumTotal // total number of words
>
void Replicate(stream<ap_uint<DataWidthIn> > & in,
		stream<ap_uint<DataWidthIn> > & out1, stream<ap_uint<DataWidthIn> > & out2) {

	for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidthIn> e = in.read();

		ap_uint<DataWidthIn> e1 = e;
		ap_uint<DataWidthIn> e2 = e;

		out1.write(e1);
		out2.write(e2);
	}
}

// Merge two input streams into a single output one
template<unsigned int DataWidthIn,		// input stream width
		unsigned int DataWidthOut,	// output stream width
		unsigned int NumTotal // total number of words
>
void Merger(stream<ap_uint<DataWidthIn> > & in1, stream<ap_uint<DataWidthIn> > & in2,
		stream<ap_uint<DataWidthOut> > & out) {

	CASSERT_DATAFLOW(DataWidthOut == DataWidthIn * 2);

	for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidthIn> e1 = in1.read();
		ap_uint<DataWidthIn> e2 = in2.read();

		ap_uint<DataWidthOut> e;
		e(DataWidthOut - 1, DataWidthIn) = e1;
		e(DataWidthIn - 1, 0) = e2;

		out.write(e);
	}
}

// simple batched function with just a loop around the base function call
template<unsigned int DataWidthIn,		// input stream width
		unsigned int DataWidthOut,	// output stream width
		unsigned int NumTotal // total number of words
>
void Merger_Batch(stream<ap_uint<DataWidthIn> > & in1, stream<ap_uint<DataWidthIn> > & in2,
		stream<ap_uint<DataWidthOut> > & out,
		const unsigned int numReps) {
	for(unsigned int rep=0; rep<numReps; rep++){
		Merger<DataWidthIn, DataWidthOut, NumTotal>(in1, in2, out);
	}
}


// data width converter uses a common iteration space (one big loop) to achieve II=1
template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void DataWidthConverter_Batch(stream<ap_uint<InWidth> > & in,
		stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
	if (InWidth > OutWidth) {
		// emit multiple output words per input word read
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		ap_uint<InWidth> ei = 0;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
			out.write(eo);
			// shift input to get new output word for next iteration
			ei = ei >> OutWidth;
			// increment written output count
			o++;
			// wraparound indices to recreate the nested loop structure
			if (o == outPerIn) {
				o = 0;
			}
		}
	} else if (InWidth == OutWidth) {
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	} else { // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		const unsigned int inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		unsigned int i = 0;
		ap_uint<OutWidth> eo = 0;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			ap_uint<InWidth> ei = in.read();
			eo = eo >> InWidth;
			eo(OutWidth - 1, OutWidth - InWidth) = ei;
			// increment read input count
			i++;
			// wraparound logic to recreate nested loop functionality
			if (i == inPerOut) {
				i = 0;
				out.write(eo);
			}
		}
	}
}

/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file stream-tools.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to adapt stream size, 
 *  remove unnecessary streams (padding) and casting
 *
 ******************************************************************************/

#ifndef STREAMTOOLS_H
#define STREAMTOOLS_H


/**
 * \brief   Stream limiter - limits the number of stream packets
 *
 * The block only let the first NumAllowed elements of a stream to pass through, the remainder
 * (NumTotal-NumAllowed) are consumed from input but not re-emitted from the output. 
 * Useful to remove padding 
 *
 * \tparam     DataWidth    Width, in number of bits, of the input and output stream
 * \tparam     NumAllowed   Number of words to pass through
 * \tparam     NumTotal     Total number of words (NumAllowed+NumDropped)
 *
 * \param      in           Input stream
 * \param      out          Output stream
 *
 */
template<unsigned int DataWidth,    
		unsigned int NumAllowed, 	
		unsigned int NumTotal       
>
void StreamLimiter(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(NumTotal >= NumAllowed);
  unsigned int numLeft = NumAllowed;
  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
    if (numLeft > 0) {
      out.write(e);
      numLeft--;
    }
  }
}

/**
 * \brief   Stream limiter batch - limits the number of stream packets multiple times
 *
 * The block only let the first NumAllowed elements of a stream to pass through, the remainder
 * (NumTotal-NumAllowed) are consumed from input but not re-emitted from the output. 
 * Useful to remove padding on multiple images (numReps)
 *
 * \tparam     DataWidth    Width, in number of bits, of the input and output stream
 * \tparam     NumAllowed   Number of words to pass through
 * \tparam     NumTotal     Total number of words (NumAllowed+NumDropped)
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the StreamLimiter function has to be called
 *
 */
template<unsigned int DataWidth,	
		unsigned int NumAllowed, 	
		unsigned int NumTotal       
>
void StreamLimiter_Batch(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamLimiter<DataWidth, NumAllowed, NumTotal>(in, out);
  }
}

/**
 * \brief   Stream Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding to the input with zeroes in case the sliding window is
 * centered on border pixels 
 *
 * \tparam     ImgDim          Size of the input feature map
 * \tparam     KernelDim       Size of the sliding window
 * \tparam     Stride          Stride of the sliding window
 * \tparam     NumChannels     Amount of channels of the input feature map
 * \tparam     In_t            Input datatype
 * \tparam     PaddingStyle    Type of padding that will be applied
 * 
 * \param      in              Input stream
 * \param      out             Output stream
 *
 */
template<	unsigned int ImgDim, 
			unsigned int KernelDim, 
			unsigned int Stride, 
			unsigned int NumChannels,
			typename In_t,
      unsigned int PaddingStyle=2>
void SameResize(stream<ap_uint<NumChannels* In_t::width> > &in, 
		stream<ap_uint<NumChannels* In_t::width> > &out){

	// Number of "same" windows over the input data
	constexpr unsigned int SameWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);
	
	// Number of elements to generate as output per dimension
	constexpr unsigned int OutputDim = KernelDim + Stride * (SameWindows - 1);

	// Padding
	constexpr unsigned int Padding = OutputDim - ImgDim;

	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;

	ap_uint<NumChannels* In_t::width> outData, inData;

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

/**
 * \brief   Stream Padding - Padds the input of multiple frames with zeroes
 *          for when the sliding window is centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels 
 *
 * \tparam     ImgDim          Size of the input feature map
 * \tparam     KernelDim       Size of the sliding window
 * \tparam     Stride          Stride of the sliding window
 * \tparam     NumChannels     Amount of channels of the input feature map
 * \tparam     In_t            Input datatype
 * \tparam     PaddingStyle    Type of padding that will be applied
 * 
 * \param      in              Input stream
 * \param      out             Output stream
 * \param      numReps         Amount of frames / images
 *
 */
template<	unsigned int ImgDim, 
			unsigned int KernelDim, 
			unsigned int Stride, 
			unsigned int NumChannels,
			typename In_t,
      unsigned int PaddingStyle=2>
void SameResize_Batch(stream<ap_uint<NumChannels* In_t::width> > &in, 
		stream<ap_uint<NumChannels* In_t::width> > &out, 
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		SameResize<ImgDim, KernelDim, Stride, NumChannels, In_t, PaddingStyle>(in, out);
	}

}


/**
 * \brief   Stream cast - Casts the input stream to a different datatype (OutT)
 *
 * Used to upscale or downscale a stream, enabling loss of information for downscaling or 
 * 0 padding for upscaling 
 *
 * \tparam     InT          Width, in number of bits, of the input and output stream
 * \tparam     OutT         Number of words to pass through
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the StreamLimiter function has to be called
 *
 */
template<typename InT, typename OutT>
void StreamingCast(hls::stream<InT> & in, hls::stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}


/**
 * \brief   Stream Data Width Converter - Converts the width of the input stream in the output stream
 *
 * Used to upscale or downscale a stream, without any loss of data in the procedure. 
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 * For upscaling (InWidth < OutWidth), OutWidth has to be a multiple of InWidth.
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream 
 * \tparam     NumInWords   Number of input words to process
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template<unsigned int InWidth,		
		unsigned int OutWidth,		
		unsigned int NumInWords		
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
		hls::stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
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
      if (o == 0) {
        ei = in.read();
	  }
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

/**
 * \brief   Stream Duplicator - Reads in a stream and writes the data into two identical streams
 *
 * Used to generate the inputs to the bypass and convolutional branches in Resnet-50
 *
 * \tparam     DataWidth    Width, in number of bits, of the streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out1         Output stream I
 * \param      out2         Output stream II
 *
 */
template<unsigned int DataWidth,
		unsigned int NumTotal
>
void DuplicateStreams(stream<ap_uint<DataWidth> > & in, stream<ap_uint<DataWidth> > & out1,
		stream<ap_uint<DataWidth> > & out2) {
	
	for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1		
		ap_uint<DataWidth> e = in.read();
		
		out1.write(e);
		out2.write(e);
	}
}

/**
 * \brief   Batch Stream Duplicator - Reads in a stream multiple times and writes the data into two identical streams
 *
 * Used to generate the inputs to the bypass and convolutional branches in Resnet-50 when dealing with multiple 'frames'
 *
 * \tparam     DataWidth    Width, in number of bits, of the streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out1         Output stream I
 * \param      out2         Output stream II
 * \param      numReps      Number of frames / images
 *
 */
template<unsigned int DataWidth,
		unsigned int NumTotal
>
void DuplicateStreams_Batch(stream<ap_uint<DataWidth> > & in, stream<ap_uint<DataWidth> > & out1,
		stream<ap_uint<DataWidth> > & out2, const unsigned int numReps) {	
	for (unsigned int image = 0; image < numReps; image++) {
		DuplicateStreams<DataWidth, NumTotal>(in, out1, out2);
	}
}

/**
 * \brief   Element-Wise Addition - Reads in data elements from two streams and writes the sum of these elements to an output
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output 
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 *
 */

template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal, 
          int offset = 0>
void AddStreams(stream<ap_uint<NumChannels * In1_t::width>> &in1, stream<ap_uint<NumChannels * In2_t::width>> &in2,
                stream<ap_uint<NumChannels * Out_t::width>> &out) {

  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<NumChannels * In1_t::width> e1 = in1.read();
    ap_uint<NumChannels * In2_t::width> e2 = in2.read();
    ap_uint<NumChannels * Out_t::width> e;
    for (unsigned int j = 0; j < NumChannels; j++) {
#pragma HLS UNROLL
      In1_t op1 = e1((j + 1) * In1_t::width - 1, j * In1_t::width);
      In2_t op2 = e2((j + 1) * In2_t::width - 1, j * In2_t::width);
      Out_t sum = op1 + op2 + offset;
      e((j + 1) * Out_t::width - 1, j * Out_t::width) = sum;
    }
    out.write(e);
  }
}


/**
 * \brief   
 *
 * Used to implement point-wise addition in Resnet-50 for multiple images
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output 
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_Batch(stream<ap_uint<NumChannels * In1_t::width>> &in1, stream<ap_uint<NumChannels * In2_t::width>> &in2,
                stream<ap_uint<NumChannels * Out_t::width>> &out, const unsigned int numReps) {
  for (unsigned int image = 0; image < numReps; image++) {
    AddStreams<NumChannels, In1_t, In2_t, Out_t, NumTotal, offset>(in1, in2, out);
  }
}

/**
 * \brief   Addition Layer - Reads in two streams and writes the sum of these streams to an output
 *
 * Used to merge the outputs of the bypass and convolutional branches in Resnet-50
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output  * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     PECount      Amount of processing elements working in parallel 
 * \tparam     offset       Offset value for the accumulation 
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          unsigned int PECount, 
          int offset = 0>
void AddStreamsLayer_Batch(stream<ap_uint<NumChannels * In1_t::width>> &in1, stream<ap_uint<NumChannels * In2_t::width>> &in2,
                           stream<ap_uint<NumChannels * Out_t::width>> &out, const unsigned int numReps) {
#pragma HLS INLINE
  CASSERT_DATAFLOW(NumChannels % PECount == 0);
  stream<ap_uint<PECount * In1_t::width>> in_folded1;
  stream<ap_uint<PECount * In2_t::width>> in_folded2;
  stream<ap_uint<PECount * Out_t::width>> out_folded;
  StreamingDataWidthConverter_Batch<NumChannels * In1_t::width, PECount * In1_t::width, NumTotal>(in1, in_folded1, numReps);
  StreamingDataWidthConverter_Batch<NumChannels * In2_t::width, PECount * In2_t::width, NumTotal>(in2, in_folded2, numReps);
  AddStreams_Batch<PECount, In1_t, In2_t, Out_t, NumTotal *(NumChannels / PECount),offset>(in_folded1, in_folded2, out_folded, numReps);
  StreamingDataWidthConverter_Batch<PECount * Out_t::width, NumChannels * Out_t::width, NumTotal *(NumChannels / PECount)>(out_folded, out, numReps);
}


/**
 * \brief   Stream Multi Chan Data Width Converter - Converts the width of the input stream in the output stream, working on multiple parallel streams
 *
 * Used to upscale or downscale a stream, without any loss of data in the procedure. 
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 * For upscaling (InWidth < OutWidth), OutWidth has to be a multiple of InWidth.
 * This version works on the MMV structure, with multiple parallel streams
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream 
 * \tparam     NumInWords   Number of input words to process
 * \tparam     NumVecs      Number of parallel vectors MMV
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords,		// number of input words to process
		unsigned int NumVecs
>
void MultiChanDataWidthConverter_Batch(
	stream<MultiChanData<NumVecs, InWidth> > & in,
	stream<MultiChanData<NumVecs, OutWidth> > & out,
	const unsigned int numReps) {
	if (InWidth > OutWidth) {
		// emit multiple output words per input word read
        CASSERT_DATAFLOW((InWidth % OutWidth) == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		MultiChanData<NumVecs, InWidth> ei;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			MultiChanData<NumVecs, OutWidth> eo;
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = (ei.data[v])(OutWidth - 1, 0);
				// shift input to get new output word for next iteration
				ei.data[v] = ei.data[v] >> OutWidth;
			}
			out.write(eo);
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
			MultiChanData<NumVecs, InWidth> e = in.read();
			MultiChanData<NumVecs, OutWidth> eo;
			// we don't support typecasting between templated types, so explicitly
			// transfer vector-by-vector here
			for(unsigned int v=0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = e.data[v];
			}
			out.write(eo);
		}
	} else { // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW((OutWidth % InWidth) == 0);
		const unsigned int inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		unsigned int i = 0;
		MultiChanData<NumVecs, OutWidth> eo;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			MultiChanData<NumVecs, InWidth> ei = in.read();
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = eo.data[v] >> InWidth;
				(eo.data[v])(OutWidth - 1, OutWidth - InWidth) = ei.data[v];
			}
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


/**
 * \brief   Flatten Multi Chan Data - Converts the parallel input stream in a flatten output stream
 *
 * Used to pach a flattened stream into a structure with multiple parallel streams
 *
 * \tparam     NumChannels  Number of channels flattened in the input stream
 * \tparam     DataWidth    Width, in number of bits, of each stream
 *
 * \param      in           Input parallel stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template <unsigned int NumChannels, unsigned int DataWidth>
void FlattenMultiChanData(
	stream<MultiChanData<NumChannels, DataWidth> > & in,
	stream<ap_uint<NumChannels*DataWidth> > & out,
	const unsigned int numReps
) {
	for(unsigned int r = 0; r < numReps; r++) {
#pragma HLS PIPELINE II=1
		MultiChanData<NumChannels, DataWidth> e = in.read();
		ap_uint<NumChannels*DataWidth> o = 0;
		for(unsigned int v = 0; v < NumChannels; v++) {
#pragma HLS UNROLL
			o(DataWidth*(v+1)-1, DataWidth*v) = e.data[v];
		}
		out.write(o);
	}
}

/**
 * \brief   Pack Multi Chan Data - Converts the flatten input stream into a parallel output stream
 *
 * Used to pach a flattened stream into a structure with multiple parallel streams
 *
 * \tparam     NumChannels  Number of channels flattened in the input stream
 * \tparam     DataWidth    Width, in number of bits, of each stream
 *
 * \param      in           Input stream
 * \param      out          Output parallel stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template <unsigned int NumChannels, unsigned int DataWidth>
void PackMultiChanData(
	stream<ap_uint<NumChannels*DataWidth> > & in,
	stream<MultiChanData<NumChannels, DataWidth> > & out,
	const unsigned int numReps
) {
	for(unsigned int r = 0; r < numReps; r++) {
#pragma HLS PIPELINE II=1
		ap_uint<NumChannels*DataWidth> e = in.read();
		MultiChanData<NumChannels, DataWidth> o;
		for(unsigned int v = 0; v < NumChannels; v++) {
#pragma HLS UNROLL
			o.data[v] = e(DataWidth*(v+1)-1, DataWidth*v);
		}
		out.write(o);
	}
}


template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedInputStream {
  hls::stream<ap_uint<OW>>  m_target;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
    StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<OW> >&() {
    return  m_target;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedInputStream<W, W, N> {

  hls::stream<ap_uint<W>> &m_source;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<W> >&  source, unsigned const  reps) : m_source(source) {}
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_source;
  }
};


template<unsigned IW, unsigned OW, unsigned N>
class WidthAdjustedOutputStream {
  hls::stream<ap_uint<IW>>  m_buffer;
  hls::stream<ap_uint<OW>> &m_target;
  unsigned const  m_reps;
  
 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<OW> >&  target, unsigned const  reps) : m_target(target), m_reps(reps) {}
  ~WidthAdjustedOutputStream() {
    StreamingDataWidthConverter_Batch<IW, OW, N>(m_buffer, m_target, m_reps);
  }

 public:
  operator hls::stream<ap_uint<IW> >&() {
    return  m_buffer;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedOutputStream<W, W, N> {
  hls::stream<ap_uint<W>> &m_target;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<W> >&  target, unsigned const  reps)
    : m_target(target) {}
  ~WidthAdjustedOutputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_target;
  }
};

#endif

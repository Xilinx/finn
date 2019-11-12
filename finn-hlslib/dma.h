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
 *  \file dma.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of functions to access memory mapped values into 
 *  streams. 
 *
 *****************************************************************************/
#ifndef DMA_HPP
#define DMA_HPP


#include <ap_int.h>
#include <hls_stream.h>

/*!
 * \brief DMA block accessing AXI4 memory and output HLS streams
 *
 * 
 * \tparam DataWidth Width, in number of bits, of the AXI4 memory pointer and the output HLS stream
 * \tparam numBytes Number of bytes to be read from the memory
 *
 * \param in Input memory pointer
 * \param out Output HLS stream
 */
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out);

/*!
 * \brief DMA block accessing AXI4 memory and output HLS streams multiple times
 * 
 * It basically calls Mem2Stream function multiple times, possibly with bigger sizes so to increase
 * the burst size
 * 
 * \tparam DataWidth Width, in number of bits, of the AXI4 memory pointer and the output HLS stream
 * \tparam numBytes Number of bytes to be read from the memory
 *
 * \param in Input memory pointer
 * \param out Output HLS stream
 * \param numReps Number of times the Stream2Mem function has to be called
 */
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out, const unsigned int numReps);

/*!
 * \brief DMA block writing HLS streams content in AXI4 pointed memory
 *
 * 
 * \tparam DataWidth Width, in number of bits, of the AXI4 memory pointer and the output HLS stream
 * \tparam numBytes Number of bytes to be read from the memory
 *
 * \param in Input HLS stream
 * \param out Output memory pointer
 */
template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out);

/*!
 * \brief DMA block that accesses the external memory and outputs HLS streams multiple times
 *
 * It basically calls Mem2Stream function multiple times, possibly with bigger sizes so to increase
 * the burst size
 *
 * \tparam DataWidth Width, in number of bits, of the AXI4 memory pointer and the output HLS stream
 * \tparam numBytes Number of bytes to be read from the memory
 *
 * \param in Input pointer to external memory
 * \param out Output the generated HLS sream
 * \param numReps Number of times the Mem2Stream function has to be called
 */
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch_external_wmem(ap_uint<DataWidth> * in,
        stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
    unsigned int rep = 0;
    while (rep != numReps) {
        Mem2Stream<DataWidth, numBytes>(&in[0], out);
        rep += 1;
    }
}

/*!
 * \brief DMA block writing HLS streams content in AXI4 pointed memory multiple times
 * 
 * It basically calls Stream2Mem function multiple times, possibly with bigger sizes so to increase
 * the burst size
 * 
 * \tparam DataWidth Width, in number of bits, of the AXI4 memory pointer and the output HLS stream
 * \tparam numBytes Number of bytes to be read from the memory
 *
 * \param in Input HLS stream
 * \param out Output memory pointer
 * \param numReps Number of times the Stream2Mem function has to be called
 */
template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem_Batch(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out, const unsigned int numReps);

template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(DataWidth % 8 == 0);
  const unsigned int numWords = numBytes / (DataWidth / 8);
  CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in[i];
    out.write(e);
  }
}


template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out) {
  CASSERT_DATAFLOW(DataWidth % 8 == 0);
  const unsigned int numWords = numBytes / (DataWidth / 8);
  CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
	out[i] = e;
  }
}

template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
  const unsigned int indsPerRep = numBytes / (DataWidth / 8);
  unsigned int rep = 0;
  // make sure Mem2Stream does not get inlined here
  // we lose burst inference otherwise
  while (rep != numReps) {
    unsigned int repsLeft = numReps - rep;
    if ((repsLeft & 0xF) == 0) {
      // repsLeft divisable by 16, read 16 images
      Mem2Stream<DataWidth, numBytes * 16>(&in[rep * indsPerRep], out);
      rep += 16;
    } else {
      // fallback, read single image
      Mem2Stream<DataWidth, numBytes>(&in[rep * indsPerRep], out);
      rep += 1;
    }
  }
}


template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem_Batch(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out, const unsigned int numReps) {
  const unsigned int indsPerRep = numBytes / (DataWidth / 8);
  unsigned int rep = 0;
  // make sure Stream2Mem does not get inlined here
  // we lose burst inference otherwise
  while (rep != numReps) {
    unsigned int repsLeft = numReps - rep;
    if ((repsLeft & 0xF) == 0) {
      // repsLeft divisable by 16, write 16 images
      Stream2Mem<DataWidth, numBytes * 16>(in, &out[rep * indsPerRep]);
      rep += 16;
    } else {
      // fallback, write single image
      Stream2Mem<DataWidth, numBytes>(in, &out[rep * indsPerRep]);
      rep += 1;
    }
  }
}

#endif

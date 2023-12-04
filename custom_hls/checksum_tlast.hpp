/******************************************************************************
 *  Copyright (c) 2022, Xilinx, Inc.
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
 * @brief	Checksum over stream-carried data frames.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 *******************************************************************************/
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

/**
 * Computes a checksum over a forwarded stream assumed to carry frames of
 * N words further subdivided into K subwords.
 *	- Subword slicing can be customized typically by using a lambda.
 *	  The provided DefaultSubwordSlicer assumes an `ap_(u)int`-like word
 *	  type with a member `width` and a range-based slicing operator. It
 *	  further assumes a little-endian arrangement of subwords within words
 *	  for the canonical subword stream order.
 *	- Subwords wider than 23 bits are folded using bitwise XOR across
 *	  slices of 23 bits starting from the LSB.
 *	- The folded subword values are weighted according to their position
 *	  in the stream relative to the start of frame by a periodic weight
 *	  sequence 1, 2, 3, ...
 *	- The weighted folded subword values are reduced to a checksum by an
 *	  accumulation module 2^24.
 *	- A checksum is emitted for each completed frame. It is the concatenation
 *	  of an 8-bit (modulo 256) frame counter and the 24-bit frame checksum.
 */
template<typename T, unsigned K> class DefaultSubwordSlicer {
	static_assert(T::width%K == 0, "Word size must be subword multiple.");
	static constexpr unsigned  W = T::width/K;
public:
	ap_uint<W> operator()(T const &x, unsigned const  j) const {
#pragma HLS inline
		return  x((j+1)*W-1, j*W);
	}
};

template<
  unsigned K,	// subword count per data word
  typename T,	// type of stream-carried data words
  typename F = DefaultSubwordSlicer<T, K>	// f(T(), j) to extract subwords
>
void checksum_tlast(
	hls::stream<hls::axis<T, 0, 0, 0>> &src,
	hls::stream<hls::axis<T, 0, 0, 0>> &dst,
	ap_uint<32>    &chk,
	ap_uint<1>     drain,	// drain data after checksuming without forward to `dst`
	F&& f = F()
) {
#pragma HLS pipeline II=1 style=flp

	// Frame counter & output
	static ap_uint< 8>  ocnt = 0;
	static ap_uint<24>  ochk = 0;
#pragma HLS reset variable=ocnt
#pragma HLS reset variable=ochk

	// State of ongoing computation
	static ap_uint< 2>  coeff[3] = { 1, 2, 3 };
	static ap_uint<24>  s = 0;
#pragma HLS reset variable=coeff
#pragma HLS reset variable=s

	if(!src.empty()) {
		auto const  x = src.read();

		// Pass-thru copy
		if(!drain)  dst.write(x);

		// Actual checksum update
		for(unsigned  j = 0; j < K; j++) {
#pragma HLS unroll
			auto const   v0 = f(x.data, j);
			constexpr unsigned  W = 1 + (decltype(v0)::width-1)/23;
			ap_uint<K*23>  v = v0;
			ap_uint<  23>  w = 0;
			for(unsigned  k = 0; k < W; k++) {
				w ^= v(23*k+22, 23*k);
			}
			s += (coeff[j%3][1]? (w, ap_uint<1>(0)) : ap_uint<24>(0)) + (coeff[j%3][0]? w : ap_uint<23>(0));
		}

		// Re-align coefficients
		for(unsigned  j = 0; j < 3; j++) {
#pragma HLS unroll
			ap_uint<3> const  cc = coeff[j] + ap_uint<3>(K%3);
			coeff[j] = cc(1, 0) + cc[2];
		}

		if(x.last) {
			// Copy result to output register
			ocnt++;
			ochk = s;

			// Reset compute state
			for(unsigned  j = 0; j < 3; j++) {
#pragma HLS unroll
				coeff[j] = j+1;
			}
			s = 0;
		}
	}
	chk = (ocnt, ochk);
}

#define CHECKSUM_TLAST_TOP_(WORD_SIZE, ITEMS_PER_WORD) \
	void checksum_tlast_ ## WORD_SIZE ## _ ## ITEMS_PER_WORD ( \
		hls::stream<hls::axis<ap_uint<WORD_SIZE>, 0, 0, 0>> &src, \
		hls::stream<hls::axis<ap_uint<WORD_SIZE>, 0, 0, 0>> &dst, \
		ap_uint<32> &chk, \
		ap_uint< 1>  drain \
	) { \
	_Pragma("HLS interface port=src axis") \
	_Pragma("HLS interface port=dst axis") \
	_Pragma("HLS interface port=chk s_axilite") \
	_Pragma("HLS interface port=drain s_axilite") \
	_Pragma("HLS interface port=return ap_ctrl_none") \
	_Pragma("HLS dataflow disable_start_propagation") \
		checksum_tlast<ITEMS_PER_WORD>(src, dst, chk, drain); \
	}
#define CHECKSUM_TLAST_TOP(WORD_SIZE, ITEMS_PER_WORD) \
	CHECKSUM_TLAST_TOP_(WORD_SIZE, ITEMS_PER_WORD)

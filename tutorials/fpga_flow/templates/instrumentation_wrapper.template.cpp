/******************************************************************************
 *  Copyright (c) 2023, Xilinx, Inc.
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
 *******************************************************************************
 * @brief	Instrumentation wrapper module for FINN IP characterization.
 * @author	Thomas B. Preusser <thomas.preusser@amd.com>
 * @details
 *	Instrumentation wrapper intercepting the feature map input to and
 *	the feature map output from a FINN IP to measure processing latency and
 *	initiation interval in terms of clock cycles. The most recent readings
 *	are exposed via AXI-light.
 *	This wrapper can run the FINN IP detached from an external data source
 *	and sink by feeding LFSR-generated data and sinking the output without
 *	backpressure.
 *	This module is currently not integrated with the FINN compiler. It must
 *	be instantiated and integrated with the rest of the system in a manual
 *	process.
 *
 * @param PENDING	maximum number of feature maps in the FINN dataflow pipeline
 * @param ILEN		number of input transactions per IFM
 * @param OLEN		number of output transactions per OFM
 * @param TI		type of input payload vector
 * @param TO		type of output payload vector
 *******************************************************************************/

#include <hls_stream.h>
#include <ap_int.h>

// Example Module Configuration
constexpr unsigned  PENDING = @PENDING@;
constexpr unsigned  ILEN    = @ILEN@;
constexpr unsigned  OLEN    = @OLEN@;
using  TI = @TI@;
using  TO = @TO@;

//---------------------------------------------------------------------------
// Utility Functions
static constexpr unsigned clog2(unsigned  x)  { return x<2? 0 : 1+clog2((x+1)/2); }

template<typename  T>
static void move(
	hls::stream<T> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read());
}

//---------------------------------------------------------------------------
// Instrumentation Core
template<
	unsigned  PENDING,
	unsigned  ILEN,
	unsigned  OLEN,
	typename  TI,
	typename  TO
>
void instrument(
	hls::stream<TI> &finnix,
	hls::stream<TO> &finnox,
	ap_uint<32>  cfg,   	// [0] - 0:thru, 1:lfsr;     [1] - sink output; [31:16]	- LFSR seed
	ap_uint<32> &status,	// [0] - timestamp overflow; [1] - timestamp underflow
	ap_uint<32> &latency,
	ap_uint<32> &interval
) {
#pragma HLS pipeline II=1 style=flp

	// Timestamp Management State
	using clock_t = ap_uint<32>;
	static clock_t  cnt_clk = 0;
#pragma HLS reset variable=cnt_clk
	hls::stream<clock_t>  timestamps;
#pragma HLS stream variable=timestamps depth=PENDING
	static bool  timestamp_ovf = false;
	static bool  timestamp_unf = false;
#pragma HLS reset variable=timestamp_ovf
#pragma HLS reset variable=timestamp_unf

	// Input Feed & Generation
	constexpr unsigned  LFSR_WIDTH = (TI::width+15)/16 * 16;
	static ap_uint<clog2(ILEN)+1>  icnt = 0;
	static bool                 thru;
	static ap_uint<LFSR_WIDTH>  lfsr;
#pragma HLS reset variable=icnt
#pragma HLS reset variable=thru off
#pragma HLS reset variable=lfsr off
	if(!finnix.full()) {
		bool  wr = false;
		TI    val;

		if(icnt == 0) {
			// Start of new feature map
			thru = !cfg[0];
			for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
#pragma HLS unroll
				lfsr(15+i, i) = cfg(31, 16) ^ (i>>4)*33331;
			}
		}
		else {
			// Advance LFSR
			for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
#pragma HLS unroll
				lfsr(15+i, i) = (lfsr(15+i, i) >> 1) ^ ap_uint<16>(lfsr[i]? 0 : 0x8805);
			}
		}

		// Input Selection
		if(!thru)  {
			wr  = true;
			val = lfsr;
		}

		// Input Feed
		if(wr) {
			finnix.write_nb(val);
			if(icnt == 0)  timestamp_ovf |= !timestamps.write_nb(cnt_clk);
			icnt = icnt == ILEN-1? decltype(icnt)(0) : decltype(icnt)(icnt + 1);
		}
	}

	// Output Tracking
	static ap_uint<clog2(OLEN)+1>  ocnt = 0;
	static bool  sink;
#pragma HLS reset variable=ocnt
#pragma HLS reset variable=sink off
	static clock_t  ts1 = 0;	// last output timestamp
	static clock_t  last_latency = 0;
	static clock_t  last_interval = 0;
#pragma HLS reset variable=ts1
#pragma HLS reset variable=last_latency
#pragma HLS reset variable=last_interval

	if(!finnox.empty()) {
		// Start of new feature map
		if(ocnt == 0)  sink = (cfg & 2) != 0;

		if(sink) {
			TO  val;
			finnox.read_nb(val);
			if(ocnt != OLEN-1)  ocnt++;
			else {
				clock_t  ts0;
				if(!timestamps.read_nb(ts0))  timestamp_unf = true;
				else {
					last_latency  = cnt_clk - ts0;
					last_interval = cnt_clk - ts1;
					ts1 = ts0;
				}
				ocnt = 0;
			}
		}
	}

	// Advance Timestamp Counter
	cnt_clk++;

	// Copy Status Outputs
	status[0] = timestamp_ovf;
	status[1] = timestamp_unf;
	latency = last_latency;
	interval = last_interval;

} // instrument()

void instrumentation_wrapper(
	hls::stream<TI> &finnix,
	hls::stream<TO> &finnox,
	ap_uint<32>  cfg,
	ap_uint<32> &status,
	ap_uint<32> &latency,
	ap_uint<32> &interval
) {
#pragma HLS interface axis port=finnix
#pragma HLS interface axis port=finnox
#pragma HLS interface s_axilite bundle=ctrl port=cfg
#pragma HLS interface s_axilite bundle=ctrl port=status
#pragma HLS interface s_axilite bundle=ctrl port=latency
#pragma HLS interface s_axilite bundle=ctrl port=interval
#pragma HLS interface ap_ctrl_none port=return

#pragma HLS dataflow disable_start_propagation
	static hls::stream<TI>  finnix0;
	static hls::stream<TO>  finnox0;
#pragma HLS stream variable=finnix0 depth=2
#pragma HLS stream variable=finnox0 depth=2

	// AXI-Stream -> FIFO
	move(finnox, finnox0);

	// Main
	instrument<PENDING, ILEN, OLEN>(finnix0, finnox0, cfg, status, latency, interval);

	// FIFO -> AXI-Stream
	move(finnix0, finnix);

} // instrumentation_wrapper
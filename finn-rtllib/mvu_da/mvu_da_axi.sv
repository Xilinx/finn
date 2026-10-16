/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	AXI-stream flow control bracket around a free-running, fixed-latency
 *			compute core (distributed-arithmetic MVU with embedded weights).
 * @details
 *	The core is a pure II=1 pipeline without valid, enable or reset: every
 *	beat accepted on the input stream is presented on core_inp and its result
 *	appears on core_out exactly CORE_LATENCY clock edges later. Validity is
 *	tracked in a shift register next to the core, and results are caught in
 *	an SRL output queue that absorbs the beats still in flight when the
 *	consumer applies back-pressure. This mirrors the output bracket of
 *	mvu_vvu_axi.sv and keeps every control signal off the arithmetic datapath.
 *****************************************************************************/

module mvu_da_axi #(
	int unsigned  INPUT_STREAM_WIDTH,	// MW * ACTIVATION_WIDTH
	int unsigned  OUTPUT_STREAM_WIDTH,	// MH * ACCU_WIDTH
	int unsigned  CORE_LATENCY,			// clock edges from core_inp to core_out, >= 1
	bit           SIGNED_OUTPUT = 1,	// sign-extend the padding bits of the output beat
	// Beats accepted before the lock takes effect can still land in the queue:
	// the lock engages one cycle after the queue becomes non-empty, so one
	// entry is queued at lock time and CORE_LATENCY more beats (accepted up to
	// the lock edge) follow. The testbench confirms CORE_LATENCY+1 as the exact
	// bound; one more entry is kept as margin, SRLs are cheap.
	int unsigned  MAX_IN_FLIGHT = CORE_LATENCY + 2,

	// Safely deducible parameters
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata,
	input	logic  s_axis_input_tvalid,
	output	logic  s_axis_input_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready,

	// Free-running compute core (registered inside the core)
	output	logic [INPUT_STREAM_WIDTH-1:0]   core_inp,
	input	logic [OUTPUT_STREAM_WIDTH-1:0]  core_out
);

	initial begin
		if(CORE_LATENCY < 1) begin
			$error("%m: CORE_LATENCY must be at least 1.");
			$finish;
		end
	end

	uwire  rst = !ap_rst_n;

	//- Input Side ----------------------------------------------------------
	uwire  idle;
	assign	s_axis_input_tready = !idle;
	uwire  ivld = s_axis_input_tvalid && !idle;
	assign	core_inp = s_axis_input_tdata[INPUT_STREAM_WIDTH-1:0];

	//- Valid Pipeline alongside the Core -----------------------------------
	logic [CORE_LATENCY-1:0]  VldPipe = '0;
	if(CORE_LATENCY == 1) begin : genVldSingle
		always_ff @(posedge ap_clk) begin
			if(rst)  VldPipe <= '0;
			else     VldPipe <= ivld;
		end
	end : genVldSingle
	else begin : genVldShift
		always_ff @(posedge ap_clk) begin
			if(rst)  VldPipe <= '0;
			else     VldPipe <= { VldPipe[CORE_LATENCY-2:0], ivld };
		end
	end : genVldShift
	uwire  ovld = VldPipe[CORE_LATENCY-1];
	uwire [OUTPUT_STREAM_WIDTH-1:0]  odat = core_out;

	//- Output Queue (identical to mvu_vvu_axi.sv blkOutput) ---------------
	if(1) begin : blkOutput
		typedef logic [OUTPUT_STREAM_WIDTH-1:0]  output_t;

		logic signed [$clog2(MAX_IN_FLIGHT+1):0]  OPtr = '1;	// -1 | 0, 1, ..., MAX_IN_FLIGHT
		(* SHREG_EXTRACT = "YES" *)
		output_t  OBuf[0:MAX_IN_FLIGHT];
		logic     OVld  =  0;
		output_t  OReg  = 'x;
		logic     OLock =  0;	// Lock upon backpressure (second entry into queue)

		// Catch every output into (SRL) Output Queue
		always_ff @(posedge ap_clk) begin
			if(ovld)  OBuf <= { odat, OBuf[0:MAX_IN_FLIGHT-1] };
		end

		always_ff @(posedge ap_clk) begin
			if(rst) begin
				OPtr  <= '1;
				OVld  <=  0;
				OReg  <= 'x;
				OLock <=  0;
			end
			else begin
				automatic logic  push = ovld;
				automatic logic  pop  = (m_axis_output_tready || !OVld) && !OPtr[$left(OPtr)];
				assert(pop || !push || (OPtr < $signed(MAX_IN_FLIGHT))) else begin
					$error("%m: Overflowing output queue.");
				end
				OPtr <= OPtr + $signed(push == pop? 0 : push? 1 : -1);

				if(OPtr[$left(OPtr)])                   OLock <= 0;
				else if(OVld && !m_axis_output_tready)  OLock <= 1;

				if(m_axis_output_tready || !OVld) begin
					OVld <= !OPtr[$left(OPtr)];
					OReg <= OBuf[OPtr[$left(OPtr)-1:0]];
				end
			end
		end
		assign	idle = OLock;

		assign	m_axis_output_tvalid = OVld;
		if(OUTPUT_STREAM_WIDTH_BA > OUTPUT_STREAM_WIDTH) begin : genPad
			assign	m_axis_output_tdata = {
				{(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){SIGNED_OUTPUT? OReg[OUTPUT_STREAM_WIDTH-1] : 1'b0}},
				OReg
			};
		end : genPad
		else begin : genNoPad
			assign	m_axis_output_tdata = OReg;
		end : genNoPad

	end : blkOutput

endmodule : mvu_da_axi

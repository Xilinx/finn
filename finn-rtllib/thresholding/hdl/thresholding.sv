/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	Pipelined thresholding by binary search.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 * @description
 *  Produces the N-bit count of those among 2^N-1 thresholds that are not
 *  larger than the corresponding input:
 *     y = Σ(T_i <= x)
 *  The result is computed by binary search. The runtime-configurable
 *  thresholds must be written in ascending order:
 *     i < j => T_i < T_j
 *  The design supports channel folding allowing each input to be processed
 *  with respect to a selectable set of thresholds. The corresponding
 *  threshold configuration relies on a channel address prefix. Inputs are
 *  accompanied by a channel selector.
 *****************************************************************************/
module thresholding #(
	int unsigned  N,  // output precision
	int unsigned  M,  // input/threshold precision
	int unsigned  C,  // number of channels

	int  BIAS = 0,  // offsetting the output [0, 2^N-1) -> [-BIAS, 2^N-1 - BIAS)

	localparam int unsigned  C_BITS = C < 2? 1 : $clog2(C),
	localparam int unsigned  O_BITS = BIAS <= 0?
		/* unsigned */ $clog2(2**N-BIAS) :
		/* signed */ 1+$clog2(BIAS >= 2**(N-1)? BIAS : 2**N-BIAS)
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Threshold Configuration
	input	logic  twe,
	input	logic [$clog2(C)+N-1:0]  twa,
	input	logic [          M-1:0]  twd,

	// Clock Enable for Stream Processing
	input	logic  en,

	// Input Stream
	input	logic  ivld,
	input	logic        [C_BITS-1:0]  icnl,	// Ignored for C == 1
	input	logic signed [M     -1:0]  idat,

	// Output Stream
	output	logic  ovld,
	output	logic [C_BITS-1:0]  ocnl,
	output	logic [O_BITS-1:0]  odat
);

	// Pipeline Links & Feed
	typedef struct packed {
		logic                      vld;	// Valid data identification
		logic        [C_BITS-1:0]  cnl;	// Channel
		logic signed [M     -1:0]  val;	// Original input value
		logic        [0:N-1]       res;	// Assembling result with valid prefix [0:stage] after stage #stage
	} pipe_t;
	uwire pipe_t  pipe[0:N];
	assign	pipe[0] = pipe_t'{ vld: ivld, cnl: icnl, val: idat, res: {N{1'bx}} };	// Feed original input

	// Stages: 0, 1, ..., N-1
	uwire [0:N-1]  tws = (twa[N-1:0]+1) & ~twa[N-1:0];   // Write Select per stage by address suffix
	for(genvar  stage = 0; stage < N; stage++) begin : genStages

		// Threshold Memory
		uwire signed [M-1:0]  thresh;
		if(1) begin : blkUpdate

			// Write control: local select from global address
			uwire  we = twe && tws[stage];
			if((C == 1) && (stage == 0)) begin
				logic signed [M-1:0]  Thresh = 'x;
				always_ff @(posedge clk) begin
					if(rst)      Thresh <= 'x;
					else if(we)  Thresh <= twd;
				end
				assign  thresh = Thresh;
			end
			else begin
				logic signed [M-1:0]  Threshs[C * 2**stage];
				uwire [$clog2(C)+stage-1:0]  wa = twa[$left(twa):N-stage];
				uwire [$clog2(C)+stage-1:0]  ra;
				if(C > 1)  assign  ra[stage+:C_BITS] = pipe[stage].cnl;
				if(stage)  assign  ra[stage-1:0]     = pipe[stage].res[0:stage-1];

				// Write
				always_ff @(posedge clk) begin
					if(we)  Threshs[wa] <= twd;
				end

				// Read
				logic signed [M-1:0]  RdReg;
				always_ff @(posedge clk) begin
					if(en)  RdReg <= Threshs[ra];
				end
				assign	thresh = RdReg;
			end

		end : blkUpdate

		// Pipeline regs simply copying the input
		pipe_t  State = '{ vld: 0, cnl: 'x, val: 'x, res: 'x };
		always_ff @(posedge clk) begin
			if(rst)      State <= '{ vld: 0, cnl: 'x, val: 'x, res: 'x };
			else if(en)  State <= pipe[stage];
		end

		// Assemble pipeline data
		logic [0:N-1]  res;
		always_comb begin
			res        = State.res;
			res[stage] = thresh <= State.val;	// Patch in next result bit
		end
		assign	pipe[stage+1] = '{
			vld: State.vld,
			cnl: State.cnl,
			val: State.val,
			res: res
		};

	end : genStages

	// Output
	assign	ovld = pipe[N].vld;
	assign	ocnl = pipe[N].cnl;
	assign	odat = pipe[N].res - BIAS;

endmodule : thresholding

/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
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
 * @brief	Pipelined multi-input adder using LUT-based compressors.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module add_multi import mvu_pkg::*; #(
	int unsigned  N,     // Number of Addends
	int unsigned  DEPTH, // Pipeline Depth
	int unsigned  ARG_WIDTH,
	int  ARG_LO = 0,
	int  ARG_HI = 0,
	bit  RESET_ZERO = 1,
	bit  TARGET = 0,   // 0 = Versal; 1 = 7-series/UltraScale (compressor path only)

	localparam int unsigned  SUM_WIDTH = sumwidth(N, ARG_WIDTH, ARG_LO, ARG_HI)
)(
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	input	logic [ARG_WIDTH-1:0]  arg[N],
	output	logic [SUM_WIDTH-1:0]  sum
);

//---------------------------------------------------------------------------
// Compressor path: unsigned low reduction via parametric add_multi_sv, whose
// LUT-compressor schedule is built at elaboration (add_multi_sched.svh).
// add_multi_sv is free-running (no en); en=0 holds inputs upstream.

	// Re-derive NP (output width) and PIPE (latency) from the same schedule.
	localparam int unsigned  SCHED_N          = N;
	localparam int unsigned  SCHED_ARG_WIDTH  = ARG_WIDTH;
	localparam bit           SCHED_SIGNED     = 1'b0;
	localparam int unsigned  SCHED_COMB_DEPTH = 1;
	`include "add_multi_sched.svh"

	// Eligible only when there is pipeline budget to hide the compressor latency.
	localparam bit  USE_COMP = !RESET_ZERO && (ARG_LO >= 0) && (DEPTH >= PIPE);

	if(USE_COMP) begin : genComp
		uwire [N-1:0][ARG_WIDTH-1:0]  a;
		for(genvar  i = 0; i < N; i++)  assign  a[i] = arg[i];

		uwire [NP-1:0]  out;
		add_multi_sv #(
			.N(N), .ARG_WIDTH(ARG_WIDTH), .TARGET(TARGET), .SIGNED(1'b0), .COMB_DEPTH(1)
		) comp_inst (
			.clk, .en(1'b1), .a, .vld(), .p(out)
		);
		initial assert(SUM_WIDTH >= NP) else
			$warning("add_multi: compressor output width %0d > SUM_WIDTH %0d", NP, SUM_WIDTH);

		// Pad the remaining pipeline depth (en-gated).
		localparam int unsigned  SUM_DELAY = DEPTH - PIPE;
		if(SUM_DELAY == 0)  assign  sum = out;
		else begin : genDelay
			logic [SUM_WIDTH-1:0]  SumZ[SUM_DELAY] = '{ default: '0 };
			always_ff @(posedge clk) begin
				if(rst)  SumZ <= '{ default: '0 };
				else if(en) begin
					for(int unsigned  i = 0; i < SUM_DELAY-1; i++)  SumZ[i] <= SumZ[i+1];
					SumZ[SUM_DELAY-1] <= out;
				end
			end
			assign	sum = SumZ[0];
		end : genDelay
	end : genComp

//- Generic Behavioral Addition ---------
	else begin : genGeneric

	localparam int unsigned  L = $clog2(N);  // Tree levels

	logic [SUM_WIDTH-1:0]  sum0;
	if(L < 1) begin : genPassThrough
		assign	sum0 = arg[0];
	end : genPassThrough
	else begin : genTree
		localparam int unsigned  D = L < DEPTH? L : DEPTH;  // Pipeline stages absorbed by tree

		// Compute the count of decendents for all nodes in the reduction trees.
		typedef int unsigned  leaf_load_t[2*N-1];
		function leaf_load_t init_leaf_loads();
			automatic leaf_load_t  res;
			for(int unsigned  i = 2*N-1; i-- > N-1;)  res[i] = 1;
			for(int unsigned  i =   N-1; i-- >   0;)  res[i] = res[2*i+1] + res[2*i+2];
			return  res;
		endfunction : init_leaf_loads
		localparam leaf_load_t   LEAF_LOAD = init_leaf_loads();

		// Adder Tree
		localparam bit  SIGNED = ARG_LO < 0;
		uwire signed [SUM_WIDTH-1:0]  tree[2*N-1];

		for(genvar  i = 0; i < 2*N-1; i++) begin
			localparam int unsigned  SW = sumwidth(LEAF_LOAD[i], ARG_WIDTH, ARG_LO, ARG_HI);
			uwire [SW-1:0]  s;

			if(N-1 <= i) begin : genLeave
				assign	s = arg[i - (N-1)];
			end : genLeave
			else begin : genReduce
				localparam int unsigned  LIDX = 2*i+1;
				localparam int unsigned  RIDX = 2*i+2;
				localparam int unsigned  LW = sumwidth(LEAF_LOAD[LIDX], ARG_WIDTH, ARG_LO, ARG_HI);
				localparam int unsigned  RW = sumwidth(LEAF_LOAD[RIDX], ARG_WIDTH, ARG_LO, ARG_HI);

				uwire [LW-1:0]  l = tree[LIDX];
				uwire [RW-1:0]  r = tree[RIDX];
				if(!SIGNED)  assign  s = l + r;
				else begin
					uwire signed [SW-1:0]  s0 = $signed(l) + $signed(r);
					assign	s = s0;
				end
			end : genReduce

			localparam int unsigned  TREE_LEVEL = $clog2(i+2) - 1;
			localparam int unsigned  STEP = L - TREE_LEVEL;
			localparam bit  REG = (TREE_LEVEL < L) && ((STEP*D / L) < ((STEP+1)*D / L));
			if(REG) begin : genReg
				localparam logic [SW-1:0]  S_RESET = {(SW){RESET_ZERO? 1'b0 : 1'bx}};
				logic [SW-1:0]  S = S_RESET;
				always_ff @(posedge clk) begin
					if(rst)      S <= S_RESET;
					else if(en)  S <= s;
				end
				assign  tree[i] = $signed({ SIGNED && S[SW-1], S} );
			end : genReg
			else  assign  tree[i] = $signed({ SIGNED && s[SW-1], s} );
		end

		assign	sum0 = tree[0];

	end : genTree

	// Delay Output if requested DEPTH exceeds Tree Height
	if(DEPTH <= L)  assign  sum = sum0;
	else begin : genDelay
		localparam int unsigned  DELAY = DEPTH - L;
		logic [SUM_WIDTH-1:0]  SumZ[DELAY] = '{ default: '0 };
		always_ff @(posedge clk) begin
			if(rst)  SumZ <= '{ default: '0 };
			else if(en) begin
				for(int unsigned  i = 0; i < DELAY-1; i++)  SumZ[i] <= SumZ[i+1];
				SumZ[DELAY-1] <= sum0;
			end
		end
		assign	sum = SumZ[0];
	end : genDelay

	end : genGeneric

endmodule : add_multi

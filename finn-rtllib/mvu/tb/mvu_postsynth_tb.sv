/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Minimal self-checking testbench for post-synthesis simulation of mvu_vvu_axi (DSP58).
 * @details
 *	Run with mvu_postsynth.sh (synthesis + xsim), it compiles this testbench against the netlist.
 *	Prints "TEST PASSED" or "TEST FAILED".
 *****************************************************************************/
`timescale 1ns/1ps

module mvu_postsynth_tb;

	// Small INT8-core MVU: PE=2 lanes, SIMD=8 (3 DSP58 per chain), 4 input vectors
	localparam int unsigned  PE = 2, SIMD = 8, MW = 32, MH = 4, NV = 4;
	localparam int unsigned  AW = 6, WW = 4, ACCU = 18;
	// Handshake: idle cycles after each input vector, valid/ready probabilities in percent
	localparam int unsigned  GAP = 40, P_VALID = 60, P_READY = 50;

	localparam int unsigned  SF = MW/SIMD, NF = MH/PE;
	localparam int unsigned  ABITS = (SIMD*AW + 7)/8*8;
	localparam int unsigned  WBITS = (PE*SIMD*WW + 7)/8*8;
	localparam int unsigned  OBITS = (PE*ACCU + 7)/8*8;

	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst_n = 0;

	//-----------------------------------------------------------------------
	// Stimulus and reference output (integer matrix product)
	int signed  A[NV][MW];
	int signed  W[MH][MW];
	logic [ABITS-1:0]  act[NV*SF];
	logic [WBITS-1:0]  wgt[NV*NF*SF];
	logic [OBITS-1:0]  exp_beat[NV*NF];

	initial begin
		automatic int signed  sum;
		for(int v = 0; v < NV; v++)  for(int k = 0; k < MW; k++)  A[v][k] = $urandom_range(0, 2**AW-1) - 2**(AW-1);
		for(int h = 0; h < MH; h++)  for(int k = 0; k < MW; k++)  W[h][k] = $urandom_range(0, 2**WW-2) - (2**(WW-1)-1);  // narrow: -7..7

		foreach(act[i])       act[i] = '0;
		foreach(wgt[i])       wgt[i] = '0;
		foreach(exp_beat[i])  exp_beat[i] = '0;
		for(int v = 0; v < NV; v++) begin
			for(int sf = 0; sf < SF; sf++)  for(int s = 0; s < SIMD; s++)
				act[v*SF + sf][s*AW +: AW] = AW'(A[v][sf*SIMD + s]);
			for(int nf = 0; nf < NF; nf++) begin
				for(int sf = 0; sf < SF; sf++)  for(int pe = 0; pe < PE; pe++)  for(int s = 0; s < SIMD; s++)
					wgt[(v*NF + nf)*SF + sf][(pe*SIMD + s)*WW +: WW] = WW'(W[nf*PE + pe][sf*SIMD + s]);
				for(int pe = 0; pe < PE; pe++) begin
					sum = 0;
					for(int k = 0; k < MW; k++)  sum += A[v][k] * W[nf*PE + pe][k];
					exp_beat[v*NF + nf][pe*ACCU +: ACCU] = ACCU'(sum);
				end
			end
		end
		#300ns @(posedge clk) rst_n <= 1;
	end

	//-----------------------------------------------------------------------
	// DUT
	logic [ABITS-1:0]  s_dat = '1;
	logic [WBITS-1:0]  w_dat = '1;
	logic  s_vld = 0, w_vld = 0, m_rdy = 0;
	uwire  s_rdy, w_rdy, m_vld;
	uwire [OBITS-1:0]  m_dat;

	// Synthesized netlist: the module has no parameters (see the generics in mvu_postsynth.sh)
	mvu_vvu_axi  dut (
		.ap_clk(clk), .ap_clk2x(1'b0), .ap_rst_n(rst_n),
		.s_axis_weights_tdata(w_dat), .s_axis_weights_tvalid(w_vld), .s_axis_weights_tready(w_rdy),
		.s_axis_input_tdata(s_dat), .s_axis_input_tvalid(s_vld), .s_axis_input_tready(s_rdy),
		.m_axis_output_tdata(m_dat), .m_axis_output_tvalid(m_vld), .m_axis_output_tready(m_rdy)
	);

	//-----------------------------------------------------------------------
	// Stream drivers: beat i is offered with probability P_VALID, held until accepted.
	// Garbage is driven whenever valid is low. The activation stream idles for GAP cycles after each
	// input vector. The weight stream keeps offering data during that gap.
	function automatic logic [ABITS-1:0]  garbage_a();
		for(int b = 0; b < ABITS; b += 8)  garbage_a[b +: 8] = $urandom;
	endfunction
	function automatic logic [WBITS-1:0]  garbage_w();
		for(int b = 0; b < WBITS; b += 8)  garbage_w[b +: 8] = $urandom;
	endfunction

	int unsigned  ai = 0, wi = 0, gap = 0;
	always @(posedge clk) if(rst_n) begin
		if(gap != 0)  gap--;
		if(s_vld && s_rdy) begin
			ai++;
			if(ai % SF == 0)  gap = GAP;
		end
		if(w_vld && w_rdy)  wi++;

		if(!s_vld || s_rdy) begin
			if(ai < NV*SF && gap == 0 && $urandom_range(99) < P_VALID) begin
				s_vld <= 1;  s_dat <= act[ai];
			end
			else begin
				s_vld <= 0;  s_dat <= garbage_a();
			end
		end
		if(!w_vld || w_rdy) begin
			if(wi < NV*NF*SF && $urandom_range(99) < P_VALID) begin
				w_vld <= 1;  w_dat <= wgt[wi];
			end
			else begin
				w_vld <= 0;  w_dat <= garbage_w();
			end
		end
		m_rdy <= $urandom_range(99) < P_READY;
	end

	//-----------------------------------------------------------------------
	// Output checker
	int unsigned  oi = 0, errors = 0;
	always @(posedge clk) if(rst_n && m_vld && m_rdy) begin
		if(m_dat[PE*ACCU-1:0] !== exp_beat[oi][PE*ACCU-1:0]) begin
			errors++;
			$display("MISMATCH output #%0d (vector %0d, rows %0d..%0d): got %h expected %h",
				oi, oi/NF, (oi%NF)*PE, (oi%NF)*PE + PE-1, m_dat[PE*ACCU-1:0], exp_beat[oi][PE*ACCU-1:0]);
		end
		oi++;
		if(oi == NV*NF) begin
			if(errors == 0)  $display("TEST PASSED: %0d output beats match the matrix product", oi);
			else             $display("TEST FAILED: %0d of %0d output beats differ from the matrix product", errors, oi);
			#50ns $finish;
		end
	end

	initial begin
		#1ms;
		$display("TEST FAILED: timeout after %0d of %0d output beats", oi, NV*NF);
		$finish;
	end

endmodule : mvu_postsynth_tb

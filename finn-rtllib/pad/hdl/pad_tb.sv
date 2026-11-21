/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Self-checking testbench for the generic padding core.
 *
 * @description
 *	Runs a set of geometries (2D, 1D with token tables, single fold, no leading
 *	padding) for both EAGER settings in two modes:
 *	  - continuous:	Always ready output.
 *	  - random:	Input bubbles, output back-pressure, and idle gaps before each frame.
 *	Every output is compared with the expected padded frame.
 ***************************************************************************/

module pad_test #(
	int unsigned  XIMG,	// image columns
	int unsigned  YIMG,	// image rows
	int unsigned  PAD_W,
	int unsigned  PAD_E,
	int unsigned  PAD_N,
	int unsigned  PAD_S,
	int unsigned  NUM_CHANNELS,
	int unsigned  SIMD,
	int unsigned  ELEM_BITS,
	bit           USE_TOKENS,	// tokens on the W/E edges rather than zeros, requires PAD_N = PAD_S = 0
	bit           EAGER,
	bit           RANDOM,
	int unsigned  FRAMES = 3
)(
	input	logic  clk,
	input	logic  rst,
	output	logic  done,
	output	int unsigned  errors
);
`default_nettype none

	localparam int unsigned  SF = NUM_CHANNELS/SIMD;
	localparam int unsigned  XON  = PAD_W;
	localparam int unsigned  XOFF = PAD_W + XIMG;
	localparam int unsigned  XEND = PAD_W + XIMG + PAD_E - 1;
	localparam int unsigned  YON  = PAD_N;
	localparam int unsigned  YOFF = PAD_N + YIMG;
	localparam int unsigned  YEND = PAD_N + YIMG + PAD_S - 1;
	localparam int unsigned  XBITS = $clog2(XEND+2);
	localparam int unsigned  YBITS = $clog2(YEND+2);

	localparam int unsigned  FRAME_FOLDS = (XEND+1) * (YEND+1) * SF;
	localparam int unsigned  LEAD_FOLDS  = (PAD_N*(XEND+1) + PAD_W) * SF;
	localparam int unsigned  NEXT_FOLDS  = EAGER? LEAD_FOLDS : 0;	// emitted ahead of the next frame
	localparam int unsigned  GAP = 3*FRAME_FOLDS + 100;	// idle cycles that suffice to settle

	typedef logic [ELEM_BITS-1:0]  elem_t;
	typedef logic [SIMD-1:0][ELEM_BITS-1:0]  fold_t;

	// Stimulus and Expectation
	function automatic elem_t datum(int unsigned  f, int unsigned  y, int unsigned  x, int unsigned  ch);
		automatic elem_t  r = elem_t'((f*131 + y*31 + x*7 + ch*3) % 251 + 1);
		if(ELEM_BITS >= 32)  r |= elem_t'(32'h40000000);
		return  r;
	endfunction : datum

	function automatic elem_t token(int unsigned  east, int unsigned  t, int unsigned  ch);
		return  elem_t'(32'hC0DE0001 + (east << 12) + (t << 6) + ch);
	endfunction : token

	localparam int unsigned  TOKENS_W = USE_TOKENS? PAD_W : 0;
	localparam int unsigned  TOKENS_E = USE_TOKENS? PAD_E : 0;
	localparam int unsigned  TOKENS_W_DEPTH = (TOKENS_W > 0)? TOKENS_W : 1;
	localparam int unsigned  TOKENS_E_DEPTH = (TOKENS_E > 0)? TOKENS_E : 1;
	typedef logic [TOKENS_W_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_BITS-1:0]  tokens_w_t;
	typedef logic [TOKENS_E_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_BITS-1:0]  tokens_e_t;
	function automatic tokens_w_t init_tokens_w();
		automatic tokens_w_t  r = '0;
		for(int unsigned  t = 0; t < TOKENS_W; t++)
			for(int unsigned  ch = 0; ch < NUM_CHANNELS; ch++)
				r[t][ch] = token(0, t, ch);
		return  r;
	endfunction : init_tokens_w
	function automatic tokens_e_t init_tokens_e();
		automatic tokens_e_t  r = '0;
		for(int unsigned  t = 0; t < TOKENS_E; t++)
			for(int unsigned  ch = 0; ch < NUM_CHANNELS; ch++)
				r[t][ch] = token(1, t, ch);
		return  r;
	endfunction : init_tokens_e

	// Fold of a position of the padded frame
	function automatic fold_t expected(int unsigned  f, int unsigned  y, int unsigned  x, int unsigned  s);
		automatic fold_t  r;
		for(int unsigned  i = 0; i < SIMD; i++) begin
			automatic int unsigned  ch = s*SIMD + i;
			if((XON <= x) && (x < XOFF) && (YON <= y) && (y < YOFF))  r[i] = datum(f, y-YON, x-XON, ch);
			else if(USE_TOKENS && (x < XON))    r[i] = token(0, x, ch);
			else if(USE_TOKENS && (x >= XOFF))  r[i] = token(1, x-XOFF, ch);
			else  r[i] = '0;
		end
		return  r;
	endfunction : expected

	fold_t  exp_q[$];	// all expected output folds in order, including the leading padding of the next frame
	initial begin
		for(int unsigned  f = 0; f <= FRAMES; f++) begin
			for(int unsigned  y = 0; y <= YEND; y++)
				for(int unsigned  x = 0; x <= XEND; x++)
					for(int unsigned  s = 0; s < SF; s++)
						exp_q.push_back(expected(f, y, x, s));
		end
	end

	//=== DUT ===============================================================
	uwire   irdy;
	logic   ivld = 0;
	fold_t  idat = 'x;
	logic   ordy = 0;
	uwire   ovld;
	uwire fold_t  odat;

	pad #(
		.NUM_CHANNELS(NUM_CHANNELS), .SIMD(SIMD), .ELEM_BITS(ELEM_BITS),
		.XCOUNTER_BITS(XBITS), .YCOUNTER_BITS(YBITS),
		.EAGER(EAGER),
		.TOKENS_W(TOKENS_W), .TOKENS_E(TOKENS_E),
		.TOKENS_W_DATA(init_tokens_w()), .TOKENS_E_DATA(init_tokens_e())
	) dut (
		.clk, .rst,
		.xon(XBITS'(XON)), .xoff(XBITS'(XOFF)), .xend(XBITS'(XEND)),
		.yon(YBITS'(YON)), .yoff(YBITS'(YOFF)), .yend(YBITS'(YEND)),
		.irdy, .ivld, .idat,
		.ordy, .ovld, .odat
	);

	//=== Input Feed ========================================================
	int unsigned  OutCount = 0;	// output folds accepted so far

	task automatic check_emitted(input string  what, input int unsigned  expect_count);
		if(OutCount != expect_count) begin
			$error("%m: %s: %0d folds emitted, expected %0d.", what, OutCount, expect_count);
			errors++;
		end
	endtask : check_emitted

	logic  Fed = 0;	// all input delivered
	initial begin
		errors = 0;
		@(posedge clk iff !rst);
		for(int unsigned  f = 0; f < FRAMES; f++) begin
			if(RANDOM) begin
				// Nothing but (eager) leading padding may appear without input.
				repeat(GAP) @(posedge clk);
				check_emitted($sformatf("frame %0d without input", f), f*FRAME_FOLDS + NEXT_FOLDS);
			end
			for(int unsigned  y = YON; y < YOFF; y++)
				for(int unsigned  x = XON; x < XOFF; x++)
					for(int unsigned  s = 0; s < SF; s++) begin
						ivld <= 1;
						idat <= expected(f, y, x, s);
						@(posedge clk iff irdy);
						if(RANDOM && ($urandom()%4 == 0)) begin
							ivld <= 0;
							idat <= 'x;
							repeat($urandom()%4) @(posedge clk);
						end
					end
			ivld <= 0;
			idat <= 'x;
		end
		Fed = 1;
	end

	//=== Output Sink and Checking ==========================================
	int unsigned  FirstCycle = 0;
	int unsigned  LastCycle  = 0;
	int unsigned  Cycle      = 0;
	initial begin
		done = 0;
		forever @(posedge clk) begin
			Cycle <= Cycle + 1;
			ordy <= !RANDOM || ($urandom()%4 != 0);
		end
	end

	initial begin
		forever @(posedge clk iff !rst && ovld && ordy) begin
			if(OutCount >= exp_q.size()) begin
				$error("%m: Unexpected output fold.");
				errors++;
			end
			else if(odat !== exp_q[OutCount]) begin
				$error("%m: Output %0d: %h, expected %h.", OutCount, odat, exp_q[OutCount]);
				errors++;
			end
			if(OutCount == 0)  FirstCycle = Cycle;
			LastCycle = Cycle;
			OutCount++;
		end
	end

	// Final checks once everything is delivered and the output settled
	initial begin
		@(posedge clk iff Fed);
		repeat(GAP) @(posedge clk);
		check_emitted("end of test", FRAMES*FRAME_FOLDS + NEXT_FOLDS);
		if(!RANDOM && (LastCycle - FirstCycle + 1 != OutCount)) begin
			$error("%m: Continuous input does not yield continuous output: %0d folds in %0d cycles.", OutCount, LastCycle - FirstCycle + 1);
			errors++;
		end
		done = 1;
	end

`default_nettype wire
endmodule : pad_test


module pad_tb;
`default_nettype none

	// Clock and reset
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(4) @(posedge clk);
		rst <= 0;
	end

	// Test configurations: X/Y image size, W/E/N/S padding, channels, SIMD, element bits, tokens
	localparam int unsigned  NUM_CONFIGS = 7;
	typedef struct {
		int unsigned  xi, yi, pw, pe, pn, ps, nch, simd, bits;
		bit  tokens;
	} cfg_t;
	localparam cfg_t  CFGS[NUM_CONFIGS] = '{
		'{ xi: 5, yi: 4, pw: 2, pe: 1, pn: 1, ps: 2, nch: 4, simd: 2, bits:  8, tokens: 0 },  // 2D, all edges
		'{ xi: 4, yi: 3, pw: 0, pe: 2, pn: 0, ps: 1, nch: 6, simd: 3, bits:  4, tokens: 0 },  // 2D, no leading padding
		'{ xi: 3, yi: 5, pw: 1, pe: 1, pn: 2, ps: 0, nch: 2, simd: 2, bits:  8, tokens: 0 },  // 2D, single fold, no south
		'{ xi: 6, yi: 1, pw: 2, pe: 1, pn: 0, ps: 0, nch: 6, simd: 2, bits: 32, tokens: 1 },  // 1D, tokens, float patterns
		'{ xi: 5, yi: 1, pw: 0, pe: 3, pn: 0, ps: 0, nch: 4, simd: 4, bits: 32, tokens: 1 },  // 1D, tokens, right only
		'{ xi: 4, yi: 1, pw: 1, pe: 0, pn: 0, ps: 0, nch: 3, simd: 1, bits: 16, tokens: 1 },  // 1D, tokens, left only
		'{ xi: 2, yi: 2, pw: 1, pe: 1, pn: 1, ps: 1, nch: 8, simd: 2, bits: 12, tokens: 0 }   // 2D, minimal image
	};

	localparam int unsigned  NUM_TESTS = NUM_CONFIGS * 2 * 2;
	uwire [NUM_TESTS-1:0]  done;
	int unsigned  test_errors[NUM_TESTS];
	for(genvar  c = 0; c < NUM_CONFIGS; c++) begin : genCfg
		for(genvar  e = 0; e < 2; e++) begin : genEager
			for(genvar  r = 0; r < 2; r++) begin : genRandom
				localparam int unsigned  IDX = 4*c + 2*e + r;
				pad_test #(
					.XIMG(CFGS[c].xi), .YIMG(CFGS[c].yi),
					.PAD_W(CFGS[c].pw), .PAD_E(CFGS[c].pe), .PAD_N(CFGS[c].pn), .PAD_S(CFGS[c].ps),
					.NUM_CHANNELS(CFGS[c].nch), .SIMD(CFGS[c].simd), .ELEM_BITS(CFGS[c].bits),
					.USE_TOKENS(CFGS[c].tokens), .EAGER(e), .RANDOM(r)
				) test (
					.clk, .rst, .done(done[IDX]), .errors(test_errors[IDX])
				);
			end : genRandom
		end : genEager
	end : genCfg

	// Verdict
	initial begin
		automatic int unsigned  total = 0;
		@(posedge clk iff &done);
		for(int unsigned  i = 0; i < NUM_TESTS; i++)  total += test_errors[i];
		if(total == 0)  $display("PASS: all %0d padding tests passed.", NUM_TESTS);
		else            $display("FAIL: %0d errors.", total);
		$finish;
	end

	// Watchdog
	initial begin
		#20ms;
		$error("Timeout, tests did not complete (%b).", done);
		$display("FAIL: timeout.");
		$finish;
	end

`default_nettype wire
endmodule : pad_tb

/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Generic padding of folded feature maps and token sequences.
 *
 * @description
 *	Pads frames of (yend+1) x (xend+1) positions. Image positions
 *	(xon <= x < xoff, yon <= y < yoff) consume input, all others are padding
 *	(zeros, or tokens on the W/E edges). A 1D sequence has yon = 0, yoff = 1, yend = 0.
 *	EAGER = 1 emits leading padding without waiting for input, EAGER = 0 waits
 *	for the first input of a frame. TOKENS_W/TOKENS_E > 0 enable token tables and
 *	require xon = TOKENS_W, xend + 1 - xoff = TOKENS_E and no padding rows.
 ***************************************************************************/

module pad #(
	int unsigned  NUM_CHANNELS,
	int unsigned  SIMD,
	int unsigned  ELEM_BITS,
	int unsigned  XCOUNTER_BITS,
	int unsigned  YCOUNTER_BITS,
	bit           EAGER = 1,
	int unsigned  TOKENS_W = 0,
	int unsigned  TOKENS_E = 0,

	localparam int unsigned  TOKENS_W_DEPTH = (TOKENS_W > 0)? TOKENS_W : 1,
	localparam int unsigned  TOKENS_E_DEPTH = (TOKENS_E > 0)? TOKENS_E : 1,
	parameter logic [TOKENS_W_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_BITS-1:0]  TOKENS_W_DATA = '0,
	parameter logic [TOKENS_E_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_BITS-1:0]  TOKENS_E_DATA = '0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Geometry
	input	logic [XCOUNTER_BITS-1:0]  xon,
	input	logic [XCOUNTER_BITS-1:0]  xoff,
	input	logic [XCOUNTER_BITS-1:0]  xend,
	input	logic [YCOUNTER_BITS-1:0]  yon,
	input	logic [YCOUNTER_BITS-1:0]  yoff,
	input	logic [YCOUNTER_BITS-1:0]  yend,

	// Input Stream
	output	logic  irdy,
	input	logic  ivld,
	input	logic [SIMD-1:0][ELEM_BITS-1:0]  idat,

	// Output Stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [SIMD-1:0][ELEM_BITS-1:0]  odat
);
`default_nettype none

	//=== Parameter Validation ==============================================
	initial begin
		if((NUM_CHANNELS < 1) || (SIMD < 1) || (NUM_CHANNELS % SIMD != 0)) begin
			$error("%m: Channel count must be a positive multiple of SIMD.");
			$finish;
		end
		if((XCOUNTER_BITS < 1) || (YCOUNTER_BITS < 1)) begin
			$error("%m: Counters must have at least one bit.");
			$finish;
		end
	end

	typedef logic [SIMD-1:0][ELEM_BITS-1:0]  fold_t;
	localparam int unsigned  SF = NUM_CHANNELS/SIMD;

	//=== Position Counters =================================================
	// The cascaded counters S (fold), X (column) and Y (row) advance on each
	// issued output fold.
	uwire  sen;

	// S-Counter: SF-2, SF-3, ..., 1, 0, -1 (last)
	typedef logic [$clog2(SF-1):0]  scount_t;
	scount_t  SCount = SF-2;
	uwire  xen = sen && SCount[$left(SCount)];
	always_ff @(posedge clk) begin
		if(rst || xen)  SCount <= SF-2;
		else if(sen)    SCount <= SCount - 1;
	end

	// X-Counter: 0, 1, ..., xend
	typedef logic [XCOUNTER_BITS-1:0]  xcount_t;
	xcount_t  XCount = 0;
	uwire  yen = xen && (XCount == xend);
	always_ff @(posedge clk) begin
		if(rst || yen)  XCount <= 0;
		else if(xen)    XCount <= XCount + 1;
	end
	uwire  xfwd = (xon <= XCount) && (XCount < xoff);

	// Y-Counter: 0, 1, ..., yend
	typedef logic [YCOUNTER_BITS-1:0]  ycount_t;
	ycount_t  YCount = 0;
	uwire  frame_end = yen && (YCount == yend);
	always_ff @(posedge clk) begin
		if(rst || frame_end)  YCount <= 0;
		else if(yen)          YCount <= YCount + 1;
	end
	uwire  yfwd = (yon <= YCount) && (YCount < yoff);

	uwire  fwd = xfwd && yfwd; // Does the current position hold image data?

	//=== Leading Padding Policy ============================================
	// Padding may be issued without input (pad_ok) unless it is the leading
	// padding of a frame and EAGER is not selected.
	uwire  pad_ok;
	if(EAGER) begin : genEager
		assign	pad_ok = 1;
	end : genEager
	else begin : genLazy
		logic  Lead = 1;	// no position of the frame issued yet
		always_ff @(posedge clk) begin
			if(rst)       Lead <= 1;
			else if(sen)  Lead <= frame_end;
		end
		assign	pad_ok = !Lead;
	end : genLazy

	//=== Pad Data ==========================================================
	uwire fold_t  pad_dat;
	if(TOKENS_W + TOKENS_E == 0) begin : genZeroPad
		assign	pad_dat = '0;
	end : genZeroPad
	else begin : genTokenPad
		localparam int unsigned  MAX_TOKENS = (TOKENS_W > TOKENS_E)? TOKENS_W : TOKENS_E;
		localparam int unsigned  PIDX_BITS = (MAX_TOKENS * SF > 1)? $clog2(MAX_TOKENS * SF) : 1;

		// Static geometry is required: Check in simulation.
		always_ff @(posedge clk) begin
			if(!rst) begin
				assert(xon == TOKENS_W) else $error("%m: xon does not match token table.");
				assert(int'(xend) + 1 - int'(xoff) == TOKENS_E) else $error("%m: xend/xoff do not match token table.");
				assert((yon == 0) && (int'(yoff) == int'(yend) + 1)) else $error("%m: Token tables do not support padding rows.");
			end
		end

		// Padding left of the image columns is west, all other is east.
		uwire  east = !(XCount < xon);

		// Pad fold index within the current horizontal edge. It restarts with
		// the image positions and with each row.
		logic [PIDX_BITS:0]  PIdx = 0;
		always_ff @(posedge clk) begin
			if(rst)  PIdx <= 0;
			else if(sen)  PIdx <= (fwd || yen)? 0 : PIdx + 1;
		end

		typedef fold_t  pad_folds_t[2**(1 + PIDX_BITS)];
		function pad_folds_t INIT_PAD_FOLDS();
			automatic pad_folds_t  f = '{default: '0};
			for(int unsigned  t = 0; t < TOKENS_W; t++)
				for(int unsigned  s = 0; s < SF; s++)
					f[{1'b0, PIDX_BITS'(t*SF + s)}] = TOKENS_W_DATA[t][s*SIMD +: SIMD];
			for(int unsigned  t = 0; t < TOKENS_E; t++)
				for(int unsigned  s = 0; s < SF; s++)
					f[{1'b1, PIDX_BITS'(t*SF + s)}] = TOKENS_E_DATA[t][s*SIMD +: SIMD];
			return  f;
		endfunction : INIT_PAD_FOLDS
		localparam pad_folds_t  PAD_FOLDS = INIT_PAD_FOLDS();

		assign	pad_dat = yfwd? PAD_FOLDS[{east, PIdx[PIDX_BITS-1:0]}] : '0;
	end : genTokenPad

	//=== Input Sidestep Register ===========================================
	// Accepts input irrespective of the output state and holds it back
	// through padding positions.
	fold_t  ADat = 'x;
	logic   AVld = 0;

	uwire  in_avail = ivld || AVld;
	uwire  bload;
	always_ff @(posedge clk) begin
		if(rst) begin
			ADat <= 'x;
			AVld <= 0;
		end
		else begin
			AVld <= in_avail && (!bload || !fwd);
			if(!AVld)  ADat <= idat;
		end
	end
	assign	irdy = !AVld;

	//=== Output Register ===================================================
	fold_t  BDat = 'x;
	logic   BVld = 0;

	assign	bload = ordy || !BVld;
	assign	sen = bload && (in_avail || (!fwd && pad_ok));
	always_ff @(posedge clk) begin
		if(rst) begin
			BDat <= 'x;
			BVld <= 0;
		end
		else if(bload) begin
			BVld <= sen;
			if(sen)  BDat <= !fwd? pad_dat : AVld? ADat : idat;
		end
	end
	assign	ovld = BVld;
	assign	odat = BDat;

`default_nettype wire
endmodule : pad

/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Generic padding of folded feature maps and token sequences.
 *
 * @description
 *	Pads frames of IH x IW image positions by PAD_N/PAD_S rows and PAD_W/PAD_E
 *	columns. Every position carries C channels in CF = C/PE folds of PE
 *	elements. Image positions consume input, all others are padding: zeros or,
 *	with TOKENS, the token tables TOKENS_W (PAD_W tokens) and TOKENS_E (PAD_E
 *	tokens) on the W/E edges, which requires PAD_N = PAD_S = 0. A 1D sequence
 *	has IH = 1. The geometry is static; the counter widths follow from it.
 *	Without any padding, the input passes through as is.
 ***************************************************************************/

module pad #(
	int unsigned  C,	// channels
	int unsigned  PE = 1,	// elements per fold, must divide C
	int unsigned  W,	// element bit width

	int unsigned  IH = 1,	// image height (rows)
	int unsigned  IW,	// image width (columns)
	int unsigned  PAD_N = 0,	// padding rows above the image
	int unsigned  PAD_S = 0,	// padding rows below the image
	int unsigned  PAD_W = 0,	// padding columns left of the image
	int unsigned  PAD_E = 0,	// padding columns right of the image

	bit  EAGER  = 1,	// emit leading padding without waiting for the first input of a frame
	bit  TOKENS = 0,	// W/E padding from TOKENS_W/TOKENS_E rather than zeros, requires PAD_N = PAD_S = 0

	localparam int unsigned  TOKENS_W_DEPTH = (PAD_W > 0)? PAD_W : 1,
	localparam int unsigned  TOKENS_E_DEPTH = (PAD_E > 0)? PAD_E : 1,
	parameter logic [TOKENS_W_DEPTH-1:0][C-1:0][W-1:0]  TOKENS_W = '0,	// west tokens, leftmost first
	parameter logic [TOKENS_E_DEPTH-1:0][C-1:0][W-1:0]  TOKENS_E = '0	// east tokens, leftmost first
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	output	logic  irdy,
	input	logic  ivld,
	input	logic [PE-1:0][W-1:0]  idat,

	// Output Stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [PE-1:0][W-1:0]  odat
);
`default_nettype none

	//=== Parameter Validation ==============================================
	initial begin
		if(W < 1) begin
			$error("%m: Elements must have at least one bit.");
			$finish;
		end
		if((C < 1) || (PE < 1) || (C % PE != 0)) begin
			$error("%m: PE must divide the channel count C.");
			$finish;
		end
		if((IW < 1) || (IH < 1)) begin
			$error("%m: Image must have at least one row and one column.");
			$finish;
		end
		if(TOKENS && (PAD_N + PAD_S > 0)) begin
			$error("%m: Token tables do not support padding rows.");
			$finish;
		end
	end

	if(PAD_W + PAD_E + PAD_N + PAD_S == 0) begin : genTrivial
		assign	irdy = ordy;
		assign	ovld = ivld;
		assign	odat = idat;
	end : genTrivial
	else begin : genPad

		// Frame geometry: columns [XON, XOFF) and rows [YON, YOFF) are image data.
		localparam int unsigned  XON  = PAD_W;
		localparam int unsigned  XOFF = PAD_W + IW;
		localparam int unsigned  XEND = PAD_W + IW + PAD_E - 1;
		localparam int unsigned  YON  = PAD_N;
		localparam int unsigned  YOFF = PAD_N + IH;
		localparam int unsigned  YEND = PAD_N + IH + PAD_S - 1;
		localparam int unsigned  XBITS = (XEND > 0)? $clog2(XEND+1) : 1;
		localparam int unsigned  YBITS = (YEND > 0)? $clog2(YEND+1) : 1;

		typedef logic [PE-1:0][W-1:0]  fold_t;
		localparam int unsigned  CF = C/PE;

		//=== Position Counters =================================================
		// The cascaded counters S (fold), X (column) and Y (row) advance on each
		// issued output fold.
		uwire  sen;

		// S-Counter: CF-2, CF-3, ..., 1, 0, -1 (last)
		typedef logic [$clog2(CF-1):0]  scount_t;
		scount_t  SCount = CF-2;
		uwire  xen = sen && SCount[$left(SCount)];
		always_ff @(posedge clk) begin
			if(rst || xen)  SCount <= CF-2;
			else if(sen)    SCount <= SCount - 1;
		end

		// X-Counter: 0, 1, ..., XEND
		typedef logic [XBITS-1:0]  xcount_t;
		xcount_t  XCount = 0;
		uwire  yen = xen && (XCount == XEND);
		always_ff @(posedge clk) begin
			if(rst || yen)  XCount <= 0;
			else if(xen)    XCount <= XCount + 1;
		end
		uwire  xfwd = (XON <= XCount) && (XCount < XOFF);

		// Y-Counter: 0, 1, ..., YEND
		typedef logic [YBITS-1:0]  ycount_t;
		ycount_t  YCount = 0;
		uwire  frame_end = yen && (YCount == YEND);
		always_ff @(posedge clk) begin
			if(rst || frame_end)  YCount <= 0;
			else if(yen)          YCount <= YCount + 1;
		end
		uwire  yfwd = (YON <= YCount) && (YCount < YOFF);

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
		if(!TOKENS || (PAD_W + PAD_E == 0)) begin : genZeroPad
			assign	pad_dat = '0;
		end : genZeroPad
		else begin : genTokenPad
			localparam int unsigned  MAX_TOKENS = (PAD_W > PAD_E)? PAD_W : PAD_E;
			localparam int unsigned  PIDX_BITS = (MAX_TOKENS * CF > 1)? $clog2(MAX_TOKENS * CF) : 1;

			// Padding left of the image columns is west, all other is east.
			uwire  east = !(XCount < XON);

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
				for(int unsigned  t = 0; t < PAD_W; t++)
					for(int unsigned  s = 0; s < CF; s++)
						f[{1'b0, PIDX_BITS'(t*CF + s)}] = TOKENS_W[t][s*PE +: PE];
				for(int unsigned  t = 0; t < PAD_E; t++)
					for(int unsigned  s = 0; s < CF; s++)
						f[{1'b1, PIDX_BITS'(t*CF + s)}] = TOKENS_E[t][s*PE +: PE];
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

	end : genPad

`default_nettype wire
endmodule : pad

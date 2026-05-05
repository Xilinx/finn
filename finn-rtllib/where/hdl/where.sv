/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	ONNX Where stream operator with multidirectional broadcasting.
 *
 * @description
 *	The three input tensors are consumed once per frame into local word
 *	memories. The output tensor is then emitted in row-major folded order.
 *	This frame-buffered schedule supports full ONNX multidirectional
 *	broadcasting, including reuse across non-contiguous output positions.
 ***************************************************************************/

`default_nettype none

module where_broadcast #(
	int unsigned  DATA_WIDTH = 32,
	int unsigned  PE = 1,
	int unsigned  NDIMS = 2,
	int unsigned  COND_NDIMS = NDIMS,
	int unsigned  X_NDIMS = NDIMS,
	int unsigned  Y_NDIMS = NDIMS,

	parameter int unsigned  OUT_SHAPE[NDIMS]       = '{ default: 1 },
	parameter int unsigned  COND_SHAPE[COND_NDIMS] = '{ default: 1 },
	parameter int unsigned  X_SHAPE[X_NDIMS]       = '{ default: 1 },
	parameter int unsigned  Y_SHAPE[Y_NDIMS]       = '{ default: 1 },

	localparam int unsigned  OUTER_DIMS = (NDIMS > 1)? NDIMS-1 : 1,
	localparam int unsigned  COND_PE = (COND_SHAPE[COND_NDIMS-1] == 1)? 1 : PE,
	localparam int unsigned  X_PE = (X_SHAPE[X_NDIMS-1] == 1)? 1 : PE,
	localparam int unsigned  Y_PE = (Y_SHAPE[Y_NDIMS-1] == 1)? 1 : PE
)(
	// Global Control
	input	wire logic  clk,
	input	wire logic  rst,

	// Condition Stream - folded according to COND_SHAPE
	input	wire logic [COND_PE-1:0]  cdat,
	input	wire logic  cvld,
	output	logic  crdy,

	// X Stream - folded according to X_SHAPE
	input	wire logic [X_PE-1:0][DATA_WIDTH-1:0]  xdat,
	input	wire logic  xvld,
	output	logic  xrdy,

	// Y Stream - folded according to Y_SHAPE
	input	wire logic [Y_PE-1:0][DATA_WIDTH-1:0]  ydat,
	input	wire logic  yvld,
	output	logic  yrdy,

	// Output Stream - folded according to OUT_SHAPE and PE
	output	logic [PE-1:0][DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	wire logic  ordy
);

	typedef int unsigned  outer_idx_t[OUTER_DIMS];
	typedef logic [COND_PE-1:0]  cond_word_t;
	typedef logic [X_PE-1:0][DATA_WIDTH-1:0]  x_word_t;
	typedef logic [Y_PE-1:0][DATA_WIDTH-1:0]  y_word_t;
	typedef logic [PE-1:0][DATA_WIDTH-1:0]  out_word_t;

	function automatic int unsigned out_outer_elems();
		automatic int unsigned  r = 1;
		for(int unsigned  i = 0; i+1 < NDIMS; i++)
			r *= OUT_SHAPE[i];
		return  r;
	endfunction : out_outer_elems

	function automatic int unsigned cond_word_count();
		automatic int unsigned  r = 1;
		for(int unsigned  i = 0; i+1 < COND_NDIMS; i++)
			r *= COND_SHAPE[i];
		if(COND_SHAPE[COND_NDIMS-1] != 1)
			r *= COND_SHAPE[COND_NDIMS-1] / PE;
		return  r;
	endfunction : cond_word_count

	function automatic int unsigned x_word_count();
		automatic int unsigned  r = 1;
		for(int unsigned  i = 0; i+1 < X_NDIMS; i++)
			r *= X_SHAPE[i];
		if(X_SHAPE[X_NDIMS-1] != 1)
			r *= X_SHAPE[X_NDIMS-1] / PE;
		return  r;
	endfunction : x_word_count

	function automatic int unsigned y_word_count();
		automatic int unsigned  r = 1;
		for(int unsigned  i = 0; i+1 < Y_NDIMS; i++)
			r *= Y_SHAPE[i];
		if(Y_SHAPE[Y_NDIMS-1] != 1)
			r *= Y_SHAPE[Y_NDIMS-1] / PE;
		return  r;
	endfunction : y_word_count

	function automatic int unsigned out_word_count();
		return  out_outer_elems() * (OUT_SHAPE[NDIMS-1] / PE);
	endfunction : out_word_count

	function automatic int unsigned cond_dim(input int unsigned axis);
		automatic int signed  source_axis = int'(axis) + int'(COND_NDIMS) - int'(NDIMS);
		if(source_axis < 0)
			return  1;
		return  COND_SHAPE[source_axis];
	endfunction : cond_dim

	function automatic int unsigned x_dim(input int unsigned axis);
		automatic int signed  source_axis = int'(axis) + int'(X_NDIMS) - int'(NDIMS);
		if(source_axis < 0)
			return  1;
		return  X_SHAPE[source_axis];
	endfunction : x_dim

	function automatic int unsigned y_dim(input int unsigned axis);
		automatic int signed  source_axis = int'(axis) + int'(Y_NDIMS) - int'(NDIMS);
		if(source_axis < 0)
			return  1;
		return  Y_SHAPE[source_axis];
	endfunction : y_dim

	function automatic int unsigned cond_word_addr(
		input outer_idx_t  out_idx,
		input int unsigned out_fold
	);
		automatic int unsigned  r = 0;
		for(int unsigned  i = 0; i+1 < NDIMS; i++) begin
			automatic int signed  source_axis = int'(i) + int'(COND_NDIMS) - int'(NDIMS);
			if(source_axis >= 0) begin
				r *= COND_SHAPE[source_axis];
				if(COND_SHAPE[source_axis] != 1)  r += out_idx[i];
			end
		end
		if(COND_SHAPE[COND_NDIMS-1] != 1)
			r = r * (COND_SHAPE[COND_NDIMS-1] / PE) + out_fold;
		return  r;
	endfunction : cond_word_addr

	function automatic int unsigned x_word_addr(
		input outer_idx_t  out_idx,
		input int unsigned out_fold
	);
		automatic int unsigned  r = 0;
		for(int unsigned  i = 0; i+1 < NDIMS; i++) begin
			automatic int signed  source_axis = int'(i) + int'(X_NDIMS) - int'(NDIMS);
			if(source_axis >= 0) begin
				r *= X_SHAPE[source_axis];
				if(X_SHAPE[source_axis] != 1)  r += out_idx[i];
			end
		end
		if(X_SHAPE[X_NDIMS-1] != 1)
			r = r * (X_SHAPE[X_NDIMS-1] / PE) + out_fold;
		return  r;
	endfunction : x_word_addr

	function automatic int unsigned y_word_addr(
		input outer_idx_t  out_idx,
		input int unsigned out_fold
	);
		automatic int unsigned  r = 0;
		for(int unsigned  i = 0; i+1 < NDIMS; i++) begin
			automatic int signed  source_axis = int'(i) + int'(Y_NDIMS) - int'(NDIMS);
			if(source_axis >= 0) begin
				r *= Y_SHAPE[source_axis];
				if(Y_SHAPE[source_axis] != 1)  r += out_idx[i];
			end
		end
		if(Y_SHAPE[Y_NDIMS-1] != 1)
			r = r * (Y_SHAPE[Y_NDIMS-1] / PE) + out_fold;
		return  r;
	endfunction : y_word_addr

	localparam int unsigned  OUT_FOLDS = OUT_SHAPE[NDIMS-1] / PE;
	localparam int unsigned  OUT_WORDS = out_word_count();
	localparam int unsigned  COND_WORDS = cond_word_count();
	localparam int unsigned  X_WORDS = x_word_count();
	localparam int unsigned  Y_WORDS = y_word_count();

	initial begin
		automatic int unsigned  max_dim;
		automatic int unsigned  cd;
		automatic int unsigned  xd;
		automatic int unsigned  yd;

		if(DATA_WIDTH < 1) begin
			$error("%m: DATA_WIDTH must be positive.");
			$finish;
		end
		if(PE < 1) begin
			$error("%m: PE must be positive.");
			$finish;
		end
		if(NDIMS < 1) begin
			$error("%m: NDIMS must be positive.");
			$finish;
		end
		if(COND_NDIMS < 1 || COND_NDIMS > NDIMS) begin
			$error("%m: COND_NDIMS must be in the range 1..NDIMS.");
			$finish;
		end
		if(X_NDIMS < 1 || X_NDIMS > NDIMS) begin
			$error("%m: X_NDIMS must be in the range 1..NDIMS.");
			$finish;
		end
		if(Y_NDIMS < 1 || Y_NDIMS > NDIMS) begin
			$error("%m: Y_NDIMS must be in the range 1..NDIMS.");
			$finish;
		end
		if((OUT_SHAPE[NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide the output innermost dimension.");
			$finish;
		end
		for(int unsigned  i = 0; i < NDIMS; i++) begin
			cd = cond_dim(i);
			xd = x_dim(i);
			yd = y_dim(i);
			max_dim = cd;

			if(cd < 1 || xd < 1 || yd < 1 || OUT_SHAPE[i] < 1) begin
				$error("%m: shape dimensions must be positive.");
				$finish;
			end
			if(xd != 1 && max_dim != 1 && xd != max_dim) begin
				$error("%m: X_SHAPE is not broadcast-compatible.");
				$finish;
			end
			if(xd != 1)  max_dim = xd;
			if(yd != 1 && max_dim != 1 && yd != max_dim) begin
				$error("%m: Y_SHAPE is not broadcast-compatible.");
				$finish;
			end
			if(yd != 1)  max_dim = yd;
			if(cd != 1 && cd != max_dim) begin
				$error("%m: COND_SHAPE is not broadcast-compatible.");
				$finish;
			end
			if(OUT_SHAPE[i] != max_dim) begin
				$error("%m: OUT_SHAPE is not the multidirectional broadcast result.");
				$finish;
			end
		end
		if(COND_SHAPE[COND_NDIMS-1] != 1 && (COND_SHAPE[COND_NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide the condition innermost dimension when not broadcast.");
			$finish;
		end
		if(X_SHAPE[X_NDIMS-1] != 1 && (X_SHAPE[X_NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide the X innermost dimension when not broadcast.");
			$finish;
		end
		if(Y_SHAPE[Y_NDIMS-1] != 1 && (Y_SHAPE[Y_NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide the Y innermost dimension when not broadcast.");
			$finish;
		end
	end

	//------------------------------------------------------------------------
	// Frame Input Buffers
	cond_word_t  Cmem[COND_WORDS];
	x_word_t     Xmem[X_WORDS];
	y_word_t     Ymem[Y_WORDS];

	int unsigned  CWr = 0;
	int unsigned  XWr = 0;
	int unsigned  YWr = 0;
	logic  CLoaded = 0;
	logic  XLoaded = 0;
	logic  YLoaded = 0;
	logic  Emit = 0;

	assign	crdy = !Emit && !CLoaded;
	assign	xrdy = !Emit && !XLoaded;
	assign	yrdy = !Emit && !YLoaded;

	uwire  c_fire = cvld && crdy;
	uwire  x_fire = xvld && xrdy;
	uwire  y_fire = yvld && yrdy;
	uwire  emit_fire = Emit && ordy;

	uwire  c_loaded_now = CLoaded || (c_fire && CWr == COND_WORDS-1);
	uwire  x_loaded_now = XLoaded || (x_fire && XWr == X_WORDS-1);
	uwire  y_loaded_now = YLoaded || (y_fire && YWr == Y_WORDS-1);

	//------------------------------------------------------------------------
	// Output Indexing
	outer_idx_t  OutIdx = '{ default: 0 };
	int unsigned  OutFold = 0;

	uwire  out_last_fold = (OutFold == OUT_FOLDS-1);
	logic  out_last_outer;
	always_comb begin
		out_last_outer = 1;
		for(int unsigned  i = 0; i+1 < NDIMS; i++)
			out_last_outer &= (OutIdx[i] == OUT_SHAPE[i]-1);
	end
	uwire  out_last = out_last_fold && out_last_outer;
	uwire  frame_done = emit_fire && out_last;

	always_ff @(posedge clk) begin
		if(rst) begin
			CWr <= 0;
			XWr <= 0;
			YWr <= 0;
			CLoaded <= 0;
			XLoaded <= 0;
			YLoaded <= 0;
			Emit <= 0;
			OutIdx <= '{ default: 0 };
			OutFold <= 0;
		end
		else begin
			if(frame_done) begin
				CWr <= 0;
				XWr <= 0;
				YWr <= 0;
				CLoaded <= 0;
				XLoaded <= 0;
				YLoaded <= 0;
				Emit <= 0;
				OutIdx <= '{ default: 0 };
				OutFold <= 0;
			end
			else begin
				if(c_fire) begin
					Cmem[CWr] <= cdat;
					CLoaded <= (CWr == COND_WORDS-1);
					if(CWr != COND_WORDS-1)  CWr <= CWr + 1;
				end
				if(x_fire) begin
					Xmem[XWr] <= xdat;
					XLoaded <= (XWr == X_WORDS-1);
					if(XWr != X_WORDS-1)  XWr <= XWr + 1;
				end
				if(y_fire) begin
					Ymem[YWr] <= ydat;
					YLoaded <= (YWr == Y_WORDS-1);
					if(YWr != Y_WORDS-1)  YWr <= YWr + 1;
				end
				if(!Emit && c_loaded_now && x_loaded_now && y_loaded_now)
					Emit <= 1;
				else if(emit_fire) begin
					if(out_last_fold) begin
						automatic bit  carry = 1;
						OutFold <= 0;
						for(int  i = int'(NDIMS)-2; i >= 0; i--) begin
							if(carry) begin
								if(OutIdx[i] == OUT_SHAPE[i]-1) begin
									OutIdx[i] <= 0;
								end
								else begin
									OutIdx[i] <= OutIdx[i] + 1;
									carry = 0;
								end
							end
						end
					end
					else
						OutFold <= OutFold + 1;
				end
			end
		end
	end

	//------------------------------------------------------------------------
	// Broadcast Selection
	uwire logic [31:0]  c_addr = cond_word_addr(OutIdx, OutFold);
	uwire logic [31:0]  x_addr = x_word_addr(OutIdx, OutFold);
	uwire logic [31:0]  y_addr = y_word_addr(OutIdx, OutFold);
	uwire cond_word_t  c_word = Cmem[c_addr];
	uwire x_word_t     x_word = Xmem[x_addr];
	uwire y_word_t     y_word = Ymem[y_addr];

	out_word_t  selected;
	for(genvar  lane = 0; lane < PE; lane++) begin : genSelect
		uwire  c = (COND_SHAPE[COND_NDIMS-1] == 1)? c_word[0] : c_word[lane];
		uwire [DATA_WIDTH-1:0]  x = (X_SHAPE[X_NDIMS-1] == 1)? x_word[0] : x_word[lane];
		uwire [DATA_WIDTH-1:0]  y = (Y_SHAPE[Y_NDIMS-1] == 1)? y_word[0] : y_word[lane];
		assign	selected[lane] = c? x : y;
	end : genSelect

	assign	odat = selected;
	assign	ovld = Emit;

endmodule : where_broadcast

`default_nettype wire

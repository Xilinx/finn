/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	ONNX Where with input_gen-based broadcast expansion.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @description
 *	Implements the ONNX expression:
 *
 *		OUT = COND ? X : Y
 *
 *	with ONNX multidirectional broadcasting.  Three folded input streams
 *	(COND 1-bit, X and Y DATA_WIDTH-bit) produce one folded output stream.
 *	Operand shapes are broadcast-expanded to the output shape following
 *	NumPy rules: trailing dimensions align, size-1 dimensions repeat.
 *
 *	Three `input_gen` instances independently expand each operand stream
 *	to output order via circular-buffer replay.  Each input_gen is
 *	configured with a loop nest whose dimensions match the output shape
 *	and whose coefficients encode the broadcast-mapped stride into the
 *	operand's linear word space.  A broadcast dimension (operand size 1)
 *	maps to coefficient 0 (replay); a non-broadcast dimension maps to
 *	the operand's row-major stride.  The ternary selection is a
 *	registered streaming mux.
 *
 *	Fill/drain overlap is natural from independent input_gen buffers.
 *	Latency is 4 cycles (constant); initiation interval approaches the
 *	output word count (the ideal lower bound).
 ***************************************************************************/

module where #(
	int unsigned  DATA_WIDTH,
	int unsigned  PE = 1,
	int unsigned  NDIMS,

	int unsigned  OUT_SHAPE[NDIMS]  = '{ default: 1 },
	int unsigned  COND_SHAPE[NDIMS] = '{ default: 1 },
	int unsigned  X_SHAPE[NDIMS]    = '{ default: 1 },
	int unsigned  Y_SHAPE[NDIMS]    = '{ default: 1 },
	parameter  RAM_STYLE = "auto",

	localparam int unsigned  COND_PE = (COND_SHAPE[NDIMS-1] == 1)? 1 : PE,
	localparam int unsigned  X_PE = (X_SHAPE[NDIMS-1] == 1)? 1 : PE,
	localparam int unsigned  Y_PE = (Y_SHAPE[NDIMS-1] == 1)? 1 : PE
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Condition Stream
	input	logic [COND_PE-1:0]  cdat,
	input	logic  cvld,
	output	logic  crdy,

	// X Stream
	input	logic [X_PE-1:0][DATA_WIDTH-1:0]  xdat,
	input	logic  xvld,
	output	logic  xrdy,

	// Y Stream
	input	logic [Y_PE-1:0][DATA_WIDTH-1:0]  ydat,
	input	logic  yvld,
	output	logic  yrdy,

	// Output Stream
	output	logic [PE-1:0][DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);
`default_nettype none

	//=== Static Parameter Validation =======================================
	typedef int unsigned  ig_dims_t[NDIMS];
	initial begin
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
		if((OUT_SHAPE[NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide output innermost dim.");
			$finish;
		end
		for(int unsigned  i = 0; i < NDIMS; i++) begin
			automatic int unsigned  cd = COND_SHAPE[i], xd = X_SHAPE[i], yd = Y_SHAPE[i], mx = cd;
			if(cd < 1 || xd < 1 || yd < 1 || OUT_SHAPE[i] < 1) begin
				$error("%m: shape dimensions must be positive.");
				$finish;
			end
			if(xd != 1) begin
				if(mx != 1 && xd != mx) begin
					$error("%m: X not broadcast-compatible.");
					$finish;
				end
				mx = xd;
			end
			if(yd != 1) begin
				if(mx != 1 && yd != mx) begin
					$error("%m: Y not broadcast-compatible.");
					$finish;
				end
				mx = yd;
			end
			if(cd != 1 && cd != mx) begin
				$error("%m: COND not broadcast-compatible.");
				$finish;
			end
			if(OUT_SHAPE[i] != mx) begin
				$error("%m: OUT_SHAPE mismatch.");
				$finish;
			end
		end
		if(COND_SHAPE[NDIMS-1] != 1 && (COND_SHAPE[NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide COND innermost.");
			$finish;
		end
		if(X_SHAPE[NDIMS-1] != 1 && (X_SHAPE[NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide X innermost.");
			$finish;
		end
		if(Y_SHAPE[NDIMS-1] != 1 && (Y_SHAPE[NDIMS-1] % PE) != 0) begin
			$error("%m: PE must divide Y innermost.");
			$finish;
		end
	end

	//=== Loop Nest Configuration for input_gen =============================
	//
	// DIMS[k] = OUT_SHAPE[k] for outer dims, OUT_SHAPE[NDIMS-1]/PE for
	// the folded innermost.  COEFS encode row-major stride into the
	// operand's word space: 0 for broadcast dims that replay, else the
	// product of operand word counts below that axis.
	// FM_SIZE = total operand word count.

	function automatic ig_dims_t  INIT_OUT_DIMS();
		automatic ig_dims_t  d;
		for(int unsigned  i = 0; i+1 < NDIMS; i++)
			d[i] = OUT_SHAPE[i];
		d[NDIMS-1] = OUT_SHAPE[NDIMS-1] / PE;
		return  d;
	endfunction : INIT_OUT_DIMS
	localparam ig_dims_t  OUT_DIMS = INIT_OUT_DIMS();

	function automatic int unsigned  INIT_FM_SIZE(input ig_dims_t s);
		automatic int unsigned  r = 1;
		for(int unsigned  i = 0; i < NDIMS; i++)
			r *= (i == NDIMS-1 && s[i] != 1)? s[i] / PE : s[i];
		return  r;
	endfunction : INIT_FM_SIZE

	function automatic ig_dims_t  INIT_COEFS(input ig_dims_t s);
		automatic ig_dims_t  c;
		for(int unsigned  k = 0; k < NDIMS; k++) begin
			if(s[k] == 1 && OUT_DIMS[k] > 1)  c[k] = 0;  // Replay
			else begin
				automatic int unsigned  stride = 1;
				for(int unsigned  j = k+1; j < NDIMS; j++)
					stride *= (j == NDIMS-1 && s[j] != 1)? s[j] / PE : s[j];
				c[k] = stride;
			end
		end
		return  c;
	endfunction : INIT_COEFS

	//=== input_gen Instances ===============================================

	//- Condition -----------------------
	typedef logic [COND_PE-1:0]  cond_word_t;
	uwire cond_word_t  c_exp_dat;
	uwire  c_exp_vld;
	uwire  c_exp_rdy;
	input_gen #(
		.DATA_WIDTH(COND_PE),
		.FM_SIZE(INIT_FM_SIZE(COND_SHAPE)),
		.D(NDIMS),
		.DIMS(OUT_DIMS),
		.COEFS(INIT_COEFS(COND_SHAPE)),
		.RAM_STYLE(RAM_STYLE)
	) c_gen (
		.clk, .rst,
		.idat(cdat), .ivld(cvld), .irdy(crdy),
		.odat(c_exp_dat), .ovld(c_exp_vld), .olst(), .ordy(c_exp_rdy)
	);

	//- Input X -------------------------
	typedef logic [X_PE-1:0][DATA_WIDTH-1:0]  x_word_t;
	uwire x_word_t  x_exp_dat;
	uwire  x_exp_vld;
	uwire  x_exp_rdy;
	input_gen #(
		.DATA_WIDTH(X_PE * DATA_WIDTH),
		.FM_SIZE(INIT_FM_SIZE(X_SHAPE)),
		.D(NDIMS),
		.DIMS(OUT_DIMS),
		.COEFS(INIT_COEFS(X_SHAPE)),
		.RAM_STYLE(RAM_STYLE)
	) x_gen (
		.clk, .rst,
		.idat(xdat), .ivld(xvld), .irdy(xrdy),
		.odat(x_exp_dat), .ovld(x_exp_vld), .olst(), .ordy(x_exp_rdy)
	);

	//- Input Y -------------------------
	typedef logic [Y_PE-1:0][DATA_WIDTH-1:0]  y_word_t;
	uwire y_word_t  y_exp_dat;
	uwire  y_exp_vld;
	uwire  y_exp_rdy;
	input_gen #(
		.DATA_WIDTH(Y_PE * DATA_WIDTH),
		.FM_SIZE(INIT_FM_SIZE(Y_SHAPE)),
		.D(NDIMS),
		.DIMS(OUT_DIMS),
		.COEFS(INIT_COEFS(Y_SHAPE)),
		.RAM_STYLE(RAM_STYLE)
	) y_gen (
		.clk, .rst,
		.idat(ydat), .ivld(yvld), .irdy(yrdy),
		.odat(y_exp_dat), .ovld(y_exp_vld), .olst(), .ordy(y_exp_rdy)
	);

	//=== Registered Output Selection =======================================
	uwire  all_valid = c_exp_vld && x_exp_vld && y_exp_vld;
	uwire  oload = !ovld || ordy;
	uwire  advance = all_valid && oload;
	assign	c_exp_rdy = advance;
	assign	x_exp_rdy = advance;
	assign	y_exp_rdy = advance;

	logic [PE-1:0][DATA_WIDTH-1:0]  ODat = 'x;
	logic  OVld = 0;
	always_ff @(posedge clk) begin
		if(rst) begin
			OVld <= 0;
			ODat <= 'x;
		end
		else if(oload) begin
			for(int unsigned  lane = 0; lane < PE; lane++) begin
				automatic logic  sel = c_exp_dat[(COND_PE == 1)? 0 : lane];
				automatic logic [DATA_WIDTH-1:0]  xv = x_exp_dat[(X_PE == 1)? 0 : lane];
				automatic logic [DATA_WIDTH-1:0]  yv = y_exp_dat[(Y_PE == 1)? 0 : lane];
				ODat[lane] <= sel? xv : yv;
			end
			OVld <= all_valid;
		end
	end
	assign	odat = ODat;
	assign	ovld = OVld;

`default_nettype wire
endmodule : where

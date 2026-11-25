/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Two-input elementwise stream operation (generalized).
 *		Supports float/float, int/float, float/int, and int/int paths.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @author	Shane Fleming <shane.fleming@amd.com>
 ***************************************************************************/

module eltwise #(
	parameter  OP,	// ADD(a+b), SUB(a-b), SBR(b-a), MUL(a*b)
	int unsigned  PE = 1,
	shortreal  B_SCALE = 1.0,
	bit  FORCE_BEHAVIORAL = 0,

	// Type selection: 1 = float32, 0 = integer
	bit  A_FLOAT = 1,
	bit  B_FLOAT = 1,

	// Integer parameters (ignored when corresponding input is float)
	int unsigned  A_WIDTH  = 32,
	bit           A_SIGNED = 0,
	int unsigned  B_WIDTH  = 32,
	bit           B_SIGNED = 0,

	// Replay of the B stream to broadcast it along axes of A. Per block of
	// B_REP_DIMS[0] * B_REP_COEFS[0] words, input_gen emits (level 0 = outermost):
	//   for(i0 < B_REP_DIMS[0]) ... for(in < B_REP_DIMS[n])
	//     emit(b[B_REP_COEFS[0]*i0 + ... + B_REP_COEFS[n]*in]);
	// A coefficient of 0 replays the words of the levels below, or holds each word on
	// the innermost level. B_REP_D = 0: no replay.
	int unsigned  B_REP_D  = 0,
	bit [(B_REP_D? B_REP_D-1 : 0):0][31:0]  B_REP_DIMS  = '0,
	bit [(B_REP_D? B_REP_D-1 : 0):0][31:0]  B_REP_COEFS = '0,

	// Port-width derivations (do not override)
	localparam int unsigned  A_DAT_W = A_FLOAT? 32 : A_WIDTH,
	localparam int unsigned  B_DAT_W = B_FLOAT? 32 : B_WIDTH,
	localparam bit           BOTH_INT = !A_FLOAT && !B_FLOAT,
	localparam bit           IS_MUL   = (OP == "MUL"),
	localparam int unsigned  INT_WIDTH = BOTH_INT? A_WIDTH : 0,
	localparam int unsigned  O_WIDTH =
		BOTH_INT? (IS_MUL? 2*INT_WIDTH : INT_WIDTH + 1) : 32
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [PE-1:0][A_DAT_W-1:0]  adat,
	input	logic  avld,
	output	logic  ardy,
	input	logic [PE-1:0][B_DAT_W-1:0]  bdat,
	input	logic  bvld,
	output	logic  brdy,

	output	logic [PE-1:0][O_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);

	//=== Derived Parameters ===============================================
	localparam bit  BOTH_FLOAT = A_FLOAT && B_FLOAT;
	localparam bit  HAVE_SCALE = (B_SCALE != 1.0);
	localparam int unsigned  BINOPF_LATENCY = HAVE_SCALE? 4 : 2 + IS_MUL;
	localparam int unsigned  BINOPI_LATENCY = IS_MUL? 3 : 1;
	localparam int unsigned  CONV_LATENCY   = (A_FLOAT ^ B_FLOAT)? 1 : 0;
	localparam int unsigned  LATENCY = BOTH_INT? BINOPI_LATENCY
	                                           : (BINOPF_LATENCY + CONV_LATENCY);

	localparam int unsigned  CREDIT = LATENCY + 3;

	//=== Parameter Validation =============================================
	initial begin
		if(BOTH_INT && B_SCALE != 1.0) begin
			$error("%m: B_SCALE=%f not supported for integer-integer path", B_SCALE);
			$finish;
		end
		if(BOTH_INT && A_SIGNED != B_SIGNED) begin
			$error("%m: A_SIGNED must match B_SIGNED for integer-integer path");
			$finish;
		end
		if(BOTH_INT && A_WIDTH != B_WIDTH) begin
			$error("%m: A_WIDTH must match B_WIDTH for integer-integer path");
			$finish;
		end
	end

	typedef logic [PE-1:0][A_DAT_W-1:0]  a_vec_t;
	typedef logic [PE-1:0][B_DAT_W-1:0]  b_vec_t;
	typedef logic [PE-1:0][O_WIDTH-1:0]  o_vec_t;

	//=== Replay of the B Stream ===========================================
	uwire b_vec_t  brep_dat;
	uwire  brep_vld;
	uwire  brep_rdy;
	if(B_REP_D == 0) begin : genNoRep
		assign	brep_dat = bdat;
		assign	brep_vld = bvld;
		assign	brdy = brep_rdy;
	end : genNoRep
	else begin : genRep
		typedef int unsigned  nest_t[B_REP_D];
		function automatic nest_t  NEST(input bit [B_REP_D-1:0][31:0]  levels);
			automatic nest_t  a;
			foreach(a[i])  a[i] = levels[B_REP_D-1-i];
			return  a;
		endfunction : NEST
		localparam nest_t  DIMS  = NEST(B_REP_DIMS);
		localparam nest_t  COEFS = NEST(B_REP_COEFS);

		input_gen #(
			.DATA_WIDTH($bits(b_vec_t)),
			.FM_SIZE(DIMS[0] * COEFS[0]),
			.D(B_REP_D),
			.DIMS(DIMS),
			.COEFS(COEFS)
		) replay (
			.clk, .rst,
			.idat(bdat), .ivld(bvld), .irdy(brdy),
			.odat(brep_dat), .ovld(brep_vld), .olst(), .odone(), .ordy(brep_rdy)
		);
	end : genRep

	//=== Input Sidestep Registers =========================================
	uwire  take;

	typedef struct {
		a_vec_t  val;
		logic    rdy;
	} abuf_t;
	typedef struct {
		b_vec_t  val;
		logic    rdy;
	} bbuf_t;
	abuf_t  A = '{ val: 'x, rdy: '1 };
	bbuf_t  B = '{ val: 'x, rdy: '1 };
	always_ff @(posedge clk) begin
		if(rst) begin
			A <= '{ val: 'x, rdy: '1 };
			B <= '{ val: 'x, rdy: '1 };
		end
		else begin
			if(A.rdy)  A.val <= adat;
			A.rdy <= (A.rdy && !avld) || take;
			if(B.rdy)  B.val <= brep_dat;
			B.rdy <= (B.rdy && !brep_vld) || take;
		end
	end
	assign	ardy = A.rdy;
	assign	brep_rdy = B.rdy;
	uwire a_vec_t  a = A.rdy? adat : A.val;
	uwire b_vec_t  b = B.rdy? brep_dat : B.val;

	//=== Credit-based Operation Issue =====================================
	logic signed [$clog2(CREDIT):0]  Credit = -CREDIT;
	uwire  give = ovld && ordy;
	assign	take = (avld || !ardy) && (brep_vld || !brep_rdy) && Credit[$left(Credit)];
	always_ff @(posedge clk) begin
		if(rst)  Credit <= -CREDIT;
		else     Credit <= Credit + ((give == take)? 0 : give? -1 : 1);
	end

	//=== Converter Valid Alignment =======================================
	logic  Take = 1'b0;
	always_ff @(posedge clk)  Take <= rst? 1'b0 : take;

	//=== Free-running Compute Pipeline ====================================
	uwire o_vec_t  r;
	uwire [PE-1:0]  rvld_vec;
	uwire  rvld;

	for(genvar  i = 0; i < PE; i++) begin : genPE

		if(BOTH_FLOAT) begin : genFF
			binopf #(.OP(OP), .B_SCALE(B_SCALE), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk, .rst,
				.a(a[i]), .avld(take),
				.b(b[i]), .bload('1),
				.r(r[i]), .rvld(rvld_vec[i])
			);
		end : genFF

		else if(!A_FLOAT && B_FLOAT) begin : genIF
			uwire [31:0]  a_fp;
			int_to_fp32 #(.WIDTH(A_WIDTH), .SIGNED(A_SIGNED)) conv (
				.ival(a[i]), .fval(a_fp)
			);
			logic [31:0]  AFp = '0;
			logic [31:0]  Bd  = '0;
			always_ff @(posedge clk) begin
				if(rst) begin
					AFp <= '0;  Bd <= '0;
				end
				else begin
					AFp <= a_fp;
					Bd  <= b[i];
				end
			end
			binopf #(.OP(OP), .B_SCALE(B_SCALE), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk, .rst,
				.a(AFp), .avld(Take),
				.b(Bd), .bload('1),
				.r(r[i]), .rvld(rvld_vec[i])
			);
		end : genIF

		else if(A_FLOAT && !B_FLOAT) begin : genFI
			uwire [31:0]  b_fp;
			int_to_fp32 #(.WIDTH(B_WIDTH), .SIGNED(B_SIGNED)) conv (
				.ival(b[i]), .fval(b_fp)
			);
			logic [31:0]  BFp = '0;
			logic [31:0]  Ad  = '0;
			always_ff @(posedge clk) begin
				if(rst) begin
					BFp <= '0;  Ad <= '0;
				end
				else begin
					BFp <= b_fp;
					Ad  <= a[i];
				end
			end
			binopf #(.OP(OP), .B_SCALE(B_SCALE), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk, .rst,
				.a(Ad), .avld(Take),
				.b(BFp), .bload('1),
				.r(r[i]), .rvld(rvld_vec[i])
			);
		end : genFI

		else begin : genII
			binopi #(.OP(OP), .WIDTH(INT_WIDTH), .SIGNED(A_SIGNED)) core (
				.clk, .rst,
				.a(a[i]), .avld(take),
				.b(b[i]), .bload('1),
				.r(r[i]), .rvld(rvld_vec[i])
			);
		end : genII

	end : genPE

	// All PE results should be valid simultaneously
	assign	rvld = rvld_vec[0];
	always_ff @(posedge clk) begin
		assert(rvld_vec == {(PE){rvld}}) else begin
			$error("%m: Inconsistent output valid indications.");
			$stop;
		end
	end

	//=== Credit-backing Elastic Output Queue ==============================
	uwire  rrdy;
	fifo #(.DATA_WIDTH($bits(o_vec_t)), .DEPTH(CREDIT)) obuf (
		.clk, .rst,
		.idat(r), .ivld(rvld), .irdy(rrdy),
		.odat, .ovld, .ordy
	);
	always_ff @(posedge clk) begin
		assert(rrdy || !rvld) else begin
			$error("%m: Result queue overrun.");
			$stop;
		end
	end

endmodule : eltwise

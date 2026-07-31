/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Streaming softmax over fixed-length FP32 vectors (max+exp+recip+div).
 * @author	Shane Fleming <shane.fleming@amd.com>
 *
 * @description
 *	Streaming softmax over fixed-length vectors of N FP32 elements
 *	consumed at SIMD elements per beat (N/SIMD beats per vector).
 *
 *	Pipeline (free-running, elastic):
 *	  Stage 1 (softmaxf_max):    SIMD-tree max + time-domain max accum
 *	                             + replay buffer; emits (max, has_infty).
 *	  Stage 2 (softmaxf_exp):    per-lane FP SUB(x-max), poly exp, ufixed
 *	                             conversion, SIMD sum tree + time-domain
 *	                             sum accum.
 *	  Stage 3 (softmaxf_recip):  Newton-Raphson reciprocal of the sum
 *	                             (FP32 input, FP32 output).
 *	  Stage 4 (softmaxf_div):    per-lane ufixed->FP32, FP MUL by recip.
 *
 *	Infinity handling propagates implicitly: when has_infty is set the
 *	exp stage substitutes 1.0 for +inf lanes and 0 elsewhere; the sum
 *	therefore equals the inf-lane count and the reciprocal computes
 *	1/k, yielding 1/k for +inf lanes and 0 elsewhere after the multiply.
 *
 *	All four stage modules live in this file; previously they were
 *	split across softmaxf_{max,exp,recip,div}.sv.  None of the stage
 *	modules is independently useful, so colocating them mirrors the
 *	pwpolyf/binopf single-file convention.
 ***************************************************************************/

module softmaxf #(
	int unsigned  N,
	int unsigned  SIMD,
	int unsigned  NR_ITERS = 2,
	int unsigned  TI_WIDTH = 32
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [SIMD-1:0][TI_WIDTH-1:0]  idat,
	input	logic                           ivld,
	output	logic                           irdy,

	output	logic [SIMD-1:0][31:0]  odat,
	output	logic                   ovld,
	input	logic                   ordy
);

	localparam int unsigned  SUM_INT = $clog2(N + 1);
	localparam int unsigned  SUM_W   = SUM_INT + 23;

	// Mirror of softmaxf_recip's TOTAL_LAT (IN_LAT=1 + NR_ITERS*ITER_LAT(=8)).
	// Plus 1 cycle for the sum_q output queue and 1 for the recip obuf output
	// queue, so stage 2's CREDIT_Y can size around the full round trip.
	localparam int unsigned  RECIP_LAT = 1 + NR_ITERS*8 + 2;

	initial begin
		if(N % SIMD != 0) begin
			$error("%m: N=%0d must be divisible by SIMD=%0d", N, SIMD);
			$finish;
		end
		if(TI_WIDTH != 32) begin
			$error("%m: only TI_WIDTH=32 (float) currently supported, got %0d", TI_WIDTH);
			$finish;
		end
		if(SUM_W < 24) begin
			$error("%m: SUM_W=%0d must be >= 24", SUM_W);
			$finish;
		end
	end

	//===========================================================================
	// Section: ufixed sum -> FP32 conversion (combinational)
	//   sum represents value sum/2^23 across SUM_W bits (SUM_INT integer).
	//---------------------------------------------------------------------------
	function automatic logic [31:0] sum_to_fp32(input logic [SUM_W-1:0] x);
		automatic int unsigned  k;
		automatic logic [22:0]  mant;
		if(x == '0)  return 32'h00000000;
		k = 0;
		for(int  i = SUM_W - 1; i >= 0; i--) begin
			if(x[i]) begin
				k = i;
				break;
			end
		end
		if(k >= 23)  mant = 23'((x >> (k - 23)) & 23'h7FFFFF);
		else         mant = 23'((x << (23 - k)) & 23'h7FFFFF);
		return  { 1'b0, 8'(k + 104), mant };
	endfunction

	//===========================================================================
	// Section: stage 1 - max extraction
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0][TI_WIDTH-1:0]  s1_repl;
	uwire                           s1_repl_vld;
	uwire                           s1_repl_rdy;
	uwire [31:0]                    s1_max;
	uwire                           s1_has_infty;
	uwire                           s1_max_vld;
	uwire                           s1_max_rdy;

	softmaxf_max #(
		.N(N), .SIMD(SIMD), .TI_WIDTH(TI_WIDTH)
	) max_inst (
		.clk, .rst,
		.xdat(idat), .xvld(ivld), .xrdy(irdy),
		.ydat(s1_repl), .yvld(s1_repl_vld), .yrdy(s1_repl_rdy),
		.mdat(s1_max),  .mhas_infty(s1_has_infty),
		.mvld(s1_max_vld), .mrdy(s1_max_rdy)
	);

	//===========================================================================
	// Section: stage 2 - exp + sum
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0][23:0]  s2_exp;
	uwire                   s2_exp_vld;
	uwire                   s2_exp_rdy;
	uwire [SUM_W-1:0]       s2_sum;
	uwire                   s2_sum_vld;
	uwire                   s2_sum_rdy;

	softmaxf_exp #(
		.N(N), .SIMD(SIMD), .TI_WIDTH(TI_WIDTH), .RECIP_LAT(RECIP_LAT)
	) exp_inst (
		.clk, .rst,
		.xdat(s1_repl), .xvld(s1_repl_vld), .xrdy(s1_repl_rdy),
		.mdat(s1_max),  .mhas_infty(s1_has_infty),
		.mvld(s1_max_vld), .mrdy(s1_max_rdy),
		.ydat(s2_exp),  .yvld(s2_exp_vld), .yrdy(s2_exp_rdy),
		.sdat(s2_sum),  .svld(s2_sum_vld), .srdy(s2_sum_rdy)
	);

	//===========================================================================
	// Section: stage 3 - reciprocal (sum -> 1/sum, both FP32)
	//---------------------------------------------------------------------------
	uwire [31:0]  s3_sum_fp32 = sum_to_fp32(s2_sum);
	uwire [31:0]  s3_recip;
	uwire         s3_recip_vld;
	uwire         s3_recip_rdy;

	softmaxf_recip #(
		.NR_ITERS(NR_ITERS)
	) recip_inst (
		.clk, .rst,
		.xdat(s3_sum_fp32), .xvld(s2_sum_vld), .xrdy(s2_sum_rdy),
		.rdat(s3_recip),    .rvld(s3_recip_vld), .rrdy(s3_recip_rdy)
	);

	//===========================================================================
	// Section: stage 4 - per-lane multiply
	//---------------------------------------------------------------------------
	softmaxf_div #(
		.N(N), .SIMD(SIMD)
	) div_inst (
		.clk, .rst,
		.ydat(s2_exp), .yvld(s2_exp_vld), .yrdy(s2_exp_rdy),
		.rdat(s3_recip), .rvld(s3_recip_vld), .rrdy(s3_recip_rdy),
		.zdat(odat), .zvld(ovld), .zrdy(ordy)
	);

endmodule : softmaxf


/****************************************************************************
 * Stage 1: per-vector max extraction + infinity flag.
 *
 * Consumes an input stream of SIMD-wide vectors.  For each window of
 * N/SIMD beats it produces:
 *   (a) the original vectors replayed onto an output stream (queued
 *       internally so the downstream subtract/exp stage can re-use them),
 *   (b) a single FP32 maximum value plus a "+infinity present" flag.
 *
 * The intra-beat reduction uses the style.md pipelined binary reduction
 * tree (latency $clog2(SIMD) cycles).  The time-domain accumulator
 * combines the tree result with a running max register that resets to
 * -infinity at the start of each new vector.
 ***************************************************************************/
module softmaxf_max #(
	int unsigned  N,
	int unsigned  SIMD,
	int unsigned  TI_WIDTH = 32
)(
	input	logic  clk,
	input	logic  rst,

	// Input vector stream
	input	logic [SIMD-1:0][TI_WIDTH-1:0]  xdat,
	input	logic                           xvld,
	output	logic                           xrdy,

	// Vector replay output (consumed by stage 2)
	output	logic [SIMD-1:0][TI_WIDTH-1:0]  ydat,
	output	logic                           yvld,
	input	logic                           yrdy,

	// Per-vector max emission (one beat per N/SIMD inputs)
	output	logic [31:0]  mdat,
	output	logic         mhas_infty,
	output	logic         mvld,
	input	logic         mrdy
);
	import softmaxf_pkg::*;

	localparam int unsigned  BEATS    = N / SIMD;
	localparam int unsigned  TREE_LAT = $clog2(SIMD);

	initial begin
		if(N % SIMD != 0) begin
			$error("%m: N=%0d must be divisible by SIMD=%0d", N, SIMD);
			$finish;
		end
		if(TI_WIDTH != 32) begin
			$error("%m: only TI_WIDTH=32 (float) currently supported, got %0d", TI_WIDTH);
			$finish;
		end
	end

	// Sign-magnitude FP32 comparator: returns a > b.  Treats NaN as undefined
	// (matches the HLS reference semantics) and lumps -0/+0 with the sign rule
	// (which is correct for max selection).
	function automatic bit fp32_gt(input logic [31:0] a, input logic [31:0] b);
		bit  sa = a[31];
		bit  sb = b[31];
		bit  mag_gt = (a[30:0] > b[30:0]);
		if(sa != sb)  return sb;          // a positive, b negative => a > b
		return  sa? !mag_gt : mag_gt;     // same sign: invert magnitude order if both negative
	endfunction

	function automatic logic [31:0] fp32_max(input logic [31:0] a, input logic [31:0] b);
		return  fp32_gt(a, b)? a : b;
	endfunction

	//===========================================================================
	// Section: input handshake / queue tee
	//   - Vector replay queue must be ready (vec_irdy)
	//   - Max output queue must be ready when the in-flight beat would be the
	//     last beat of a vector (otherwise we'd risk dropping the emit)
	//---------------------------------------------------------------------------
	uwire  vec_irdy;
	uwire  max_q_irdy;

	// Predict which input beat is the last beat of its window.
	logic [$clog2(BEATS+1)-1:0]  InBeatCnt = '0;
	uwire  in_is_last = (InBeatCnt == BEATS-1);

	uwire  take = xvld && vec_irdy && (!in_is_last || max_q_irdy);
	assign  xrdy = vec_irdy && (!in_is_last || max_q_irdy);

	always_ff @(posedge clk) begin
		if(rst)        InBeatCnt <= '0;
		else if(take)  InBeatCnt <= in_is_last? '0 : InBeatCnt + 1;
	end

	//===========================================================================
	// Section: vector replay queue
	//---------------------------------------------------------------------------
	queue #(
		.DATA_WIDTH(SIMD * TI_WIDTH),
		.ELASTICITY(BEATS + TREE_LAT + 4)
	) replay (
		.clk, .rst,
		.idat(xdat), .ivld(take), .irdy(vec_irdy),
		.odat(ydat), .ovld(yvld), .ordy(yrdy)
	);

	//===========================================================================
	// Section: SIMD max reduction tree (combinational compare, registered output)
	//---------------------------------------------------------------------------
	uwire [31:0]  beat_max;
	uwire         beat_vld;
	uwire         beat_is_last;

	if(SIMD == 1) begin : genNoTree
		// Trivial reduction: pass-through with no tree latency.
		assign  beat_max     = xdat[0];
		assign  beat_vld     = take;
		assign  beat_is_last = take && in_is_last;
	end : genNoTree
	else begin : genTree
		typedef logic [31:0]  edge_t;
		uwire edge_t  tree[2*SIMD-1];

		for(genvar  i = 0; i < SIMD; i++) begin : genLeaves
			assign  tree[SIMD-1+i] = xdat[i];
		end : genLeaves

		// Edge delay balancing for non-power-of-two SIMD.
		typedef bit edge_delays_t[2*SIMD-1];
		function automatic edge_delays_t INIT_EDGE_DELAYS();
			localparam int unsigned  LEVELS = 1 + $clog2(SIMD);
			automatic edge_delays_t  d = '{ default: 0 };
			for(int unsigned  i = SIMD-1; i < 2*SIMD-1; i++) begin
				if($clog2(i+2) == LEVELS)  break;
				d[i] = 1;
			end
			for(int unsigned  i = SIMD-1; i > 0; i--) begin
				if(d[2*i+1]) begin
					d[2*i+1] = 0;
					d[2*i+2] = 0;
					d[i] = 1;
				end
			end
			return  d;
		endfunction
		localparam edge_delays_t  EDGE_DELAYS = INIT_EDGE_DELAYS();

		for(genvar  i = 0; i < SIMD-1; i++) begin : genNodes
			uwire edge_t  a;
			if(EDGE_DELAYS[2*i+2]) begin : genDelay
				edge_t  Del = 'x;
				always_ff @(posedge clk)  Del <= tree[2*i+2];
				assign  a = Del;
			end : genDelay
			else begin : genDirect
				assign  a = tree[2*i+2];
			end : genDirect

			edge_t  Reg = 'x;
			always_ff @(posedge clk)  Reg <= fp32_max(tree[2*i+1], a);
			assign  tree[i] = Reg;
		end : genNodes

		assign  beat_max = tree[0];

		// Valid + last shift registers tracking the tree pipeline.
		// TREE_LAT==1 (SIMD==2) needs a degenerate "shreg" with no slice;
		// xsim tolerates [-1:0] but xsynth rejects it.
		logic [TREE_LAT-1:0]  VldShreg  = '0;
		logic [TREE_LAT-1:0]  LastShreg = '0;
		if(TREE_LAT == 1) begin : genShreg1
			always_ff @(posedge clk) begin
				if(rst) begin
					VldShreg  <= '0;
					LastShreg <= '0;
				end
				else begin
					VldShreg  <= take;
					LastShreg <= take && in_is_last;
				end
			end
		end : genShreg1
		else begin : genShregN
			always_ff @(posedge clk) begin
				if(rst) begin
					VldShreg  <= '0;
					LastShreg <= '0;
				end
				else begin
					VldShreg  <= { VldShreg [$left(VldShreg )-1:0], take };
					LastShreg <= { LastShreg[$left(LastShreg)-1:0], take && in_is_last };
				end
			end
		end : genShregN
		assign  beat_vld     = VldShreg [$left(VldShreg )];
		assign  beat_is_last = LastShreg[$left(LastShreg)];
	end : genTree

	//===========================================================================
	// Section: time-domain max accumulator + emission
	//---------------------------------------------------------------------------
	logic [31:0]  MaxAcc = FP32_NEG_INF;
	uwire [31:0]  new_max = fp32_max(MaxAcc, beat_max);

	always_ff @(posedge clk) begin
		if(rst)               MaxAcc <= FP32_NEG_INF;
		else if(beat_vld)     MaxAcc <= beat_is_last? FP32_NEG_INF : new_max;
	end

	uwire         emit_max       = beat_vld && beat_is_last;
	uwire         emit_has_infty = fp32_is_pos_inf(new_max);

	queue #(
		.DATA_WIDTH(33),
		.ELASTICITY(4)
	) max_q (
		.clk, .rst,
		.idat({emit_has_infty, new_max}),
		.ivld(emit_max),
		.irdy(max_q_irdy),
		.odat({mhas_infty, mdat}),
		.ovld(mvld),
		.ordy(mrdy)
	);

endmodule : softmaxf_max


/****************************************************************************
 * Stage 2: per-lane FP SUB, polynomial exp, fixed-point sum.
 *
 * Consumes the replayed input vector stream from stage 1 plus the per-
 * vector scalar (max, has_infty).  For each beat:
 *   - Per lane: x' = xi - max via DSPFP32 SUB (binopf, latency 2)
 *   - Per lane: y_fp = exp(x') via Horner chain on EXP_DEGREE pwpolyf_dspfp32
 *     instances (latency EXP_DEGREE * DSP_LAT)
 *   - Per lane: convert y_fp to ufixed<1+SUM_PRECISION,1>; if has_infty,
 *     bypass with 1.0 for +inf inputs and 0 otherwise.
 * Per-vector summation (fixed-point) tracks the running sum across the
 * N/SIMD beats of a window via a SIMD-wide adder tree (style.md pattern)
 * feeding a time-domain accumulator.
 *
 * Outputs:
 *   ydat[SIMD]: per-lane ufixed exp value per beat
 *   sdat:       per-vector ufixed sum (one per N/SIMD beats)
 ***************************************************************************/
module softmaxf_exp #(
	int unsigned  N,
	int unsigned  SIMD,
	int unsigned  TI_WIDTH = 32,
	// Round-trip latency from sum_q push to first stage-4 take, in cycles.
	// Sized by the parent module to cover the recip pipeline + queue overhead.
	// y_obuf credit must absorb a full vector PLUS this round trip; otherwise
	// stage 2 stalls every vector boundary and II degrades to ~recip_lat/BEATS.
	int unsigned  RECIP_LAT = 20
)(
	input	logic  clk,
	input	logic  rst,

	// Replayed input vector stream (from stage 1)
	input	logic [SIMD-1:0][TI_WIDTH-1:0]  xdat,
	input	logic                           xvld,
	output	logic                           xrdy,

	// Per-vector max + infinity flag (from stage 1)
	input	logic [31:0]  mdat,
	input	logic         mhas_infty,
	input	logic         mvld,
	output	logic         mrdy,

	// Per-beat fixed-point exp vector (to stage 4)
	//   width = 1 + SUM_PRECISION (=24 for FP32)
	output	logic [SIMD-1:0][23:0]  ydat,
	output	logic                   yvld,
	input	logic                   yrdy,

	// Per-vector fixed-point sum (to stage 3)
	//   width = $clog2(N+1) + SUM_PRECISION
	output	logic [$clog2(N + 1) + 22 :0]  sdat,
	output	logic                               svld,
	input	logic                               srdy
);
	import  softmaxf_pkg::*;

	localparam int unsigned  SUM_PRECISION = 23;
	localparam int unsigned  EXP_W    = 1 + SUM_PRECISION;
	localparam int unsigned  SUM_INT  = $clog2(N + 1);
	localparam int unsigned  SUM_W    = SUM_INT + SUM_PRECISION;
	localparam int unsigned  BEATS    = N / SIMD;
	localparam int unsigned  DSP_LAT  = 4;
	localparam int unsigned  EXP_LAT  = EXP_DEGREE * DSP_LAT;       // Horner pipeline
	localparam int unsigned  SUB_LAT  = 2;                           // binopf SUB
	localparam int unsigned  COMP_LAT = SUB_LAT + EXP_LAT;           // input -> FP32 exp
	localparam int unsigned  TREE_LAT = (SIMD <= 1)? 0 : $clog2(SIMD);
	localparam int unsigned  Y_LAT    = COMP_LAT;                    // input -> y_obuf
	localparam int unsigned  S_LAT    = COMP_LAT + TREE_LAT;         // input -> sum_q

	// y_obuf must hold an entire vector window PLUS the recip round trip,
	// otherwise stage 2 starves of credit while stage 4 waits for recip(V0)
	// and the resulting feedback drives steady-state II to ~RECIP_LAT/BEATS.
	// S_LAT covers the in-stage path to sum_q; RECIP_LAT covers everything
	// downstream up to and including stage 4's first take of the new vector.
	localparam int unsigned  CREDIT_Y = BEATS + S_LAT + RECIP_LAT + 3;
	localparam int unsigned  CREDIT_S = S_LAT + 3;

	initial begin
		if(N % SIMD != 0) begin
			$error("%m: N=%0d must be divisible by SIMD=%0d", N, SIMD);
			$finish;
		end
		if(TI_WIDTH != 32) begin
			$error("%m: only TI_WIDTH=32 (float) currently supported, got %0d", TI_WIDTH);
			$finish;
		end
	end

	//===========================================================================
	// Section: input handshake / sidestep / vector-aligned scheduling
	//---------------------------------------------------------------------------
	typedef logic [SIMD-1:0][31:0]  fp_vec_t;

	uwire  take;

	// Sidestep buffer for xdat (pwpolyf-style elastic input)
	typedef struct {
		fp_vec_t  val;
		logic     rdy;
	} ibuf_t;
	ibuf_t  Ibuf = '{ val: 'x, rdy: '1 };
	always_ff @(posedge clk) begin
		if(rst)  Ibuf <= '{ val: 'x, rdy: '1 };
		else begin
			if(Ibuf.rdy)  Ibuf.val <= xdat;
			Ibuf.rdy <= (Ibuf.rdy && !xvld) || take;
		end
	end
	assign	xrdy = Ibuf.rdy;
	uwire fp_vec_t  x_cur = Ibuf.rdy? xdat : Ibuf.val;

	// Vector beat counter -- predicts first/last beat of a window
	logic [$clog2(BEATS+1)-1:0]  InBeatCnt = '0;
	uwire  in_is_first = (InBeatCnt == '0);
	uwire  in_is_last  = (InBeatCnt == BEATS-1);
	always_ff @(posedge clk) begin
		if(rst)        InBeatCnt <= '0;
		else if(take)  InBeatCnt <= in_is_last? '0 : InBeatCnt + 1;
	end

	// Credit-based output queue gating (two streams: y_obuf and sum_q)
	logic signed [$clog2(CREDIT_Y):0]  CreditY = -CREDIT_Y;
	logic signed [$clog2(CREDIT_S):0]  CreditS = -CREDIT_S;
	uwire  give_y;
	uwire  give_s;
	uwire  take_s = take && in_is_last;

	assign	take = (xvld || !Ibuf.rdy)
	             && CreditY[$left(CreditY)]
	             && (!in_is_first || mvld)
	             && (!in_is_last  || CreditS[$left(CreditS)]);

	always_ff @(posedge clk) begin
		if(rst)  CreditY <= -CREDIT_Y;
		else     CreditY <= CreditY + (give_y == take?   0 : give_y? -1 : 1);
	end
	always_ff @(posedge clk) begin
		if(rst)  CreditS <= -CREDIT_S;
		else     CreditS <= CreditS + (give_s == take_s? 0 : give_s? -1 : 1);
	end

	// Pop max queue when consuming the FIRST beat of a vector.  The binopf SUB
	// instances capture mdat into their D register via bload on this same cycle
	// (bload = take && in_is_first), so the max scalar must be valid on first-
	// beat take; popping here releases the upstream max queue immediately.
	assign	mrdy = take && in_is_first;

	//===========================================================================
	// Section: per-lane FP SUB (xi - max) via binopf SUB, latency = SUB_LAT
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0][31:0]  sub_out;

	for(genvar  i = 0; i < SIMD; i++) begin : genSub
		binopf #(.OP("SUB")) sub_inst (
			.clk, .rst,
			.a(x_cur[i]), .avld(take),
			.b(mdat),     .bload(take && in_is_first),
			.r(sub_out[i]), .rvld()
		);
	end : genSub

	// Valid pipeline (take -> SUB output, latency SUB_LAT)
	logic [SUB_LAT-1:0]  SubVld = '0;
	always_ff @(posedge clk) begin
		if(rst)  SubVld <= '0;
		else     SubVld <= { SubVld[$left(SubVld)-1:0], take };
	end
	uwire  sub_vld = SubVld[$left(SubVld)];

	//===========================================================================
	// Section: per-lane polynomial exp via Horner chain on pwpolyf_dspfp32
	//   Stage 0: s[0] = c[D-1] + c[D]   * x'
	//   Stage j: s[j] = c[D-1-j] + s[j-1] * x'_delayed
	//   Latency: EXP_DEGREE * DSP_LAT
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0][31:0]  poly_out;

	for(genvar  pe = 0; pe < SIMD; pe++) begin : genExp
		uwire [31:0]  xi = sub_out[pe];

		//--- Segment selector (combinational, x' is non-positive) ----------
		uwire        s_sign = xi[31];
		uwire [7:0]  s_exp  = xi[30:23];
		uwire [EXP_K-1:0]  s_sub = xi[22 -: EXP_K];

		// Octave: s_exp 127 -> 0, 128 -> 1, ..., 130 -> 3 (NUM_OCTAVES=4)
		uwire [$clog2(EXP_NUM_OCTAVES)-1:0]  octave =
			s_exp[$clog2(EXP_NUM_OCTAVES)-1:0] - 8'd127;

		uwire  is_near_zero = (s_exp < 127);
		uwire  is_clamp     = s_sign && (s_exp >= EXP_CLAMP_EXP);

		uwire [$clog2(EXP_NUM_SEGS)-1:0]  seg_idx =
			is_near_zero? '0 :
			$clog2(EXP_NUM_SEGS)'(1 + (octave * (1 << EXP_K)) + s_sub);

		//--- Horner pipeline valid shreg -----------------------------------
		logic [EXP_LAT-1:0]  PolyVld = '0;
		always_ff @(posedge clk) begin
			if(rst)  PolyVld <= '0;
			else     PolyVld <= { PolyVld[$left(PolyVld)-1:0], sub_vld };
		end

		// Delay x' for B inputs of stages 1..D-1
		logic [31:0]  XDly[EXP_LAT] = '{ default: 'x };
		always_ff @(posedge clk) begin
			XDly[0] <= xi;
			for(int  d = 1; d < EXP_LAT; d++)  XDly[d] <= XDly[d-1];
		end

		// DSP chain
		uwire [31:0]  s[EXP_DEGREE];

		for(genvar  j = 0; j < EXP_DEGREE; j++) begin : genStage
			uwire [31:0]  dsp_a = (j == 0)? EXP_COEFFS[seg_idx][EXP_DEGREE] : s[j-1];
			uwire [31:0]  dsp_b = (j == 0)? xi                              : XDly[j*DSP_LAT - 1];

			uwire [31:0]  dsp_c;
			if(j == 0) begin : genCdir
				assign	dsp_c = EXP_COEFFS[seg_idx][EXP_DEGREE-1];
			end : genCdir
			else begin : genCdly
				logic [31:0]  CDly[j*DSP_LAT] = '{ default: 'x };
				always_ff @(posedge clk) begin
					CDly[0] <= EXP_COEFFS[seg_idx][EXP_DEGREE-1-j];
					for(int  d = 1; d < j*DSP_LAT; d++)  CDly[d] <= CDly[d-1];
				end
				assign	dsp_c = CDly[j*DSP_LAT - 1];
			end : genCdly

			pwpolyf_dspfp32  dsp (
				.clk, .rst,
				.a(dsp_a), .b(dsp_b), .c(dsp_c),
				.r(s[j]), .rvld(PolyVld[(j+1)*DSP_LAT - 1])
			);
		end : genStage

		//--- Clamp passthrough (x <= -16 -> exp ~= 0) ----------------------
		logic [EXP_LAT-1:0]  ClampSh = '0;
		always_ff @(posedge clk) begin
			if(rst)  ClampSh <= '0;
			else     ClampSh <= { ClampSh[$left(ClampSh)-1:0], is_clamp };
		end

		assign	poly_out[pe] = ClampSh[$left(ClampSh)]? 32'h00000000 : s[EXP_DEGREE-1];
	end : genExp

	//===========================================================================
	// Section: pipeline-aligned valid + infinity bypass tracking
	//   Carry has_infty (per beat) and per-lane "xi == +inf" through COMP_LAT
	//   cycles to align with the polynomial output.  When out_has_infty is set,
	//   replace the polynomial result with 1.0 for +inf lanes and 0 otherwise.
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0]  in_is_pos_inf;
	for(genvar  i = 0; i < SIMD; i++) begin : genInfDet
		assign	in_is_pos_inf[i] = fp32_is_pos_inf(x_cur[i]);
	end : genInfDet

	// Latch mhas_infty on first-beat take (mvld is only required on first beat,
	// so we cannot sample the input port on subsequent beats of the same vector).
	logic  HasInftyReg = 1'b0;
	always_ff @(posedge clk) begin
		if(rst)                            HasInftyReg <= 1'b0;
		else if(take && in_is_first)       HasInftyReg <= mhas_infty;
	end
	uwire  cur_has_infty = (take && in_is_first)? mhas_infty : HasInftyReg;

	// Global compute valid: take delayed by COMP_LAT.  All per-lane PolyVlds
	// share the same timing, so this is also a valid handle to use for output
	// muxing without crossing instance hierarchies.
	logic [COMP_LAT-1:0]            VldShr      = '0;
	logic [COMP_LAT-1:0]            HasInfShr   = '0;
	logic [COMP_LAT-1:0][SIMD-1:0]  InfLaneShr  = '0;
	always_ff @(posedge clk) begin
		if(rst) begin
			VldShr     <= '0;
			HasInfShr  <= '0;
			InfLaneShr <= '0;
		end
		else begin
			VldShr     <= { VldShr    [$left(VldShr    )-1:0], take };
			HasInfShr  <= { HasInfShr [$left(HasInfShr )-1:0], take? cur_has_infty : 1'b0 };
			InfLaneShr <= { InfLaneShr[$left(InfLaneShr)-1:0], in_is_pos_inf };
		end
	end
	uwire             out_vld       = VldShr    [$left(VldShr    )];
	uwire             out_has_infty = HasInfShr [$left(HasInfShr )];
	uwire [SIMD-1:0]  out_is_inf    = InfLaneShr[$left(InfLaneShr)];

	//===========================================================================
	// Section: FP32 -> ufixed<1+SUM_PRECISION,1> conversion (combinational)
	//   - Sign bit set or magnitude underflow  -> 0
	//   - Magnitude >= 2.0                     -> saturate to all-ones
	//   - Otherwise: significand >> (127 - exp)
	//---------------------------------------------------------------------------
	function automatic logic [EXP_W-1:0] fp32_to_ufixed(input logic [31:0] x);
		automatic logic [7:0]   exp_bits = x[30:23];
		automatic logic [22:0]  mant     = x[22:0];
		automatic logic         sign     = x[31];
		automatic logic [EXP_W-1:0]  sig = { 1'b1, mant };  // implicit-1 significand
		automatic int           shift;
		if(sign)                                       return '0;
		if(exp_bits == 0)                              return '0;
		if(exp_bits >= 128)                            return '1;
		if(int'(exp_bits) < (127 - SUM_PRECISION))     return '0;
		shift = 127 - int'(exp_bits);
		return  EXP_W'(sig >> shift);
	endfunction

	//===========================================================================
	// Section: per-lane output mux + y_obuf push
	//---------------------------------------------------------------------------
	logic [SIMD-1:0][EXP_W-1:0]  beat_y;
	always_comb begin
		for(int  i = 0; i < SIMD; i++) begin
			if(out_has_infty)
				beat_y[i] = out_is_inf[i]? { 1'b1, {SUM_PRECISION{1'b0}} } : '0;
			else
				beat_y[i] = fp32_to_ufixed(poly_out[i]);
		end
	end

	uwire  y_irdy;
	queue #(
		.DATA_WIDTH(SIMD * EXP_W),
		.ELASTICITY(CREDIT_Y)
	) y_obuf (
		.clk, .rst,
		.idat(beat_y), .ivld(out_vld), .irdy(y_irdy),
		.odat(ydat),   .ovld(yvld),    .ordy(yrdy)
	);
	assign	give_y = yvld && yrdy;
	always_ff @(posedge clk) begin
		assert(y_irdy || !out_vld) else begin
			$error("%m: y_obuf overrun.");
			$stop;
		end
	end

	//===========================================================================
	// Section: SIMD-wide fixed-point sum tree (style.md pattern, uniform width)
	//---------------------------------------------------------------------------
	localparam int unsigned  RED_INT = (SIMD <= 1)? 1 : ($clog2(SIMD) + 1);
	localparam int unsigned  RED_W   = RED_INT + SUM_PRECISION;
	typedef logic [RED_W-1:0]  red_t;

	uwire red_t  beat_sum;

	if(SIMD <= 1) begin : genSumNoTree
		assign	beat_sum = red_t'(beat_y[0]);
	end : genSumNoTree
	else begin : genSumTree
		uwire red_t  tree[2*SIMD-1];

		for(genvar  i = 0; i < SIMD; i++) begin : genLeaves
			assign	tree[SIMD-1+i] = red_t'(beat_y[i]);
		end : genLeaves

		// Edge delay balancing for non-power-of-two SIMD (matches softmaxf_max).
		typedef bit edge_delays_t[2*SIMD-1];
		function automatic edge_delays_t INIT_EDGE_DELAYS();
			localparam int unsigned  LEVELS = 1 + $clog2(SIMD);
			automatic edge_delays_t  d = '{ default: 0 };
			for(int unsigned  i = SIMD-1; i < 2*SIMD-1; i++) begin
				if($clog2(i+2) == LEVELS)  break;
				d[i] = 1;
			end
			for(int unsigned  i = SIMD-1; i > 0; i--) begin
				if(d[2*i+1]) begin
					d[2*i+1] = 0;
					d[2*i+2] = 0;
					d[i]     = 1;
				end
			end
			return  d;
		endfunction
		localparam edge_delays_t  EDGE_DELAYS = INIT_EDGE_DELAYS();

		for(genvar  i = 0; i < SIMD-1; i++) begin : genNodes
			uwire red_t  a;
			if(EDGE_DELAYS[2*i+2]) begin : genDelay
				red_t  Del = 'x;
				always_ff @(posedge clk)  Del <= tree[2*i+2];
				assign	a = Del;
			end : genDelay
			else begin : genDirect
				assign	a = tree[2*i+2];
			end : genDirect

			red_t  Reg = 'x;
			always_ff @(posedge clk)  Reg <= tree[2*i+1] + a;
			assign	tree[i] = Reg;
		end : genNodes

		assign	beat_sum = tree[0];
	end : genSumTree

	//===========================================================================
	// Section: time-domain fixed-point accumulator + sum emission
	//   Latency from input take -> sum_q.ivld = COMP_LAT + TREE_LAT
	//---------------------------------------------------------------------------
	logic [S_LAT-1:0]  SumVld  = '0;
	logic [S_LAT-1:0]  SumLast = '0;
	always_ff @(posedge clk) begin
		if(rst) begin
			SumVld  <= '0;
			SumLast <= '0;
		end
		else begin
			SumVld  <= { SumVld [$left(SumVld )-1:0], take };
			SumLast <= { SumLast[$left(SumLast)-1:0], take && in_is_last };
		end
	end
	uwire  sum_beat_vld  = SumVld [$left(SumVld )];
	uwire  sum_beat_last = SumLast[$left(SumLast)];

	logic [SUM_W-1:0]  SumAcc = '0;
	uwire [SUM_W-1:0]  new_sum = SumAcc + SUM_W'(beat_sum);

	// MUX-before-ADD: gate both addends so on the last beat the next-state adder
	// folds to 0 in the same LUT level as the addition (no terminal MUX).
	uwire [SUM_W-1:0]  acc_base = sum_beat_last? '0 : SumAcc;
	uwire [SUM_W-1:0]  acc_inc  = sum_beat_last? '0 : SUM_W'(beat_sum);
	always_ff @(posedge clk) begin
		if(rst)                SumAcc <= '0;
		else if(sum_beat_vld)  SumAcc <= acc_base + acc_inc;
	end

	uwire  emit_sum = sum_beat_vld && sum_beat_last;

	uwire  s_irdy;
	queue #(
		.DATA_WIDTH(SUM_W),
		.ELASTICITY(CREDIT_S)
	) sum_q (
		.clk, .rst,
		.idat(new_sum), .ivld(emit_sum), .irdy(s_irdy),
		.odat(sdat),    .ovld(svld),     .ordy(srdy)
	);
	assign	give_s = svld && srdy;
	always_ff @(posedge clk) begin
		assert(s_irdy || !emit_sum) else begin
			$error("%m: sum_q overrun.");
			$stop;
		end
	end

endmodule : softmaxf_exp


/****************************************************************************
 * Stage 3: FP32 reciprocal via Newton-Raphson on pipelined DSPFP32.
 *
 * Computes y = 1/x in FP32 via the standard Newton-Raphson recurrence
 *   y_{n+1} = y_n * (2 - x*y_n)
 * with an integer bit-magic seed
 *   y0 = 32'h7EF477D5 - x.i
 * (Quake-style: ~8 bit accuracy after seed; each NR iteration roughly
 * doubles the precision -- 2 iters reach FP32 ULP for typical inputs.)
 *
 * Implemented as a fully-pipelined chain using `binopf` MUL/SUB
 * instances.  The expected input rate is one sample per N/SIMD beats
 * (stage 3 of softmaxf), so DSP utilisation is low; resource sharing
 * is left as a future optimisation (cf. rsqrtf.sv genExclusive variant).
 *
 * Per iteration:
 *   m1 = x * y    (binopf MUL, latency 3)
 *   m2 = 2.0 - m1 (binopf SUB, latency 2)
 *   yn = y * m2   (binopf MUL, latency 3)
 * Two iterations therefore add 16 cycles of latency.
 ***************************************************************************/
module softmaxf_recip #(
	int unsigned  NR_ITERS = 2
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  xdat,
	input	logic         xvld,
	output	logic         xrdy,

	output	logic [31:0]  rdat,
	output	logic         rvld,
	input	logic         rrdy
);

	localparam int unsigned  MUL_LAT  = 3;
	localparam int unsigned  SUB_LAT  = 2;
	localparam int unsigned  ITER_LAT = MUL_LAT + SUB_LAT + MUL_LAT; // 8
	localparam int unsigned  IN_LAT   = 1;                            // input register stage
	localparam int unsigned  TOTAL_LAT = IN_LAT + NR_ITERS * ITER_LAT;
	localparam int unsigned  CREDIT    = TOTAL_LAT + 3;

	localparam logic [31:0]  FP32_TWO   = 32'h40000000;
	localparam logic [31:0]  SEED_MAGIC = 32'h7EF477D5;

	initial begin
		if(NR_ITERS == 0 || NR_ITERS > 4) begin
			$error("%m: NR_ITERS=%0d unsupported (must be 1..4)", NR_ITERS);
			$finish;
		end
	end

	//===========================================================================
	// Section: input handshake / credit-gated take
	//---------------------------------------------------------------------------
	logic signed [$clog2(CREDIT):0]  Credit = -CREDIT;
	uwire  give;
	uwire  take = xvld && Credit[$left(Credit)];
	assign	xrdy = Credit[$left(Credit)];

	always_ff @(posedge clk) begin
		if(rst)  Credit <= -CREDIT;
		else     Credit <= Credit + (give == take? 0 : give? -1 : 1);
	end

	//===========================================================================
	// Section: input register stage + Quake-style integer-magic seed
	//   Breaks the long combinational path that runs from softmaxf_exp's
	//   sum_q register through the parent module's sum_to_fp32 conversion
	//   and the SEED_MAGIC-xdat subtractor into the first DSPFP32
	//   multiplier.  XReg/SeedReg also isolate the multiplier inputs from
	//   the upstream conversion logic, giving the placer freedom to colocate
	//   the converter with either side.
	//---------------------------------------------------------------------------
	uwire [31:0]  seed = SEED_MAGIC - xdat;

	logic [31:0]  XReg    = 'x;
	logic [31:0]  SeedReg = 'x;
	always_ff @(posedge clk) begin
		XReg    <= xdat;
		SeedReg <= seed;
	end

	//===========================================================================
	// Section: NR iteration chain
	//   y[i+1] = y[i] * (2 - x*y[i])
	//
	//   x must be delayed alongside y across each iteration.  Within one
	//   iteration: x_in -> m1=x*y (MUL_LAT) -> m2=2-m1 (SUB_LAT) -> ymul=y*m2
	//   (MUL_LAT).  y also needs delaying for ymul; we use binopf MUL with
	//   bload=avld (so b is loaded each cycle alongside a).
	//---------------------------------------------------------------------------
	uwire [31:0]  x_chain[NR_ITERS+1];
	uwire [31:0]  y_chain[NR_ITERS+1];

	assign	x_chain[0] = XReg;
	assign	y_chain[0] = SeedReg;

	for(genvar  i = 0; i < NR_ITERS; i++) begin : genIter
		// m1 = x * y  -- both stream, both loaded each cycle
		uwire [31:0]  m1;
		binopf #(.OP("MUL")) mul1 (
			.clk, .rst,
			.a(y_chain[i]), .avld(take),
			.b(x_chain[i]), .bload('1),
			.r(m1), .rvld()
		);

		// Delay x by MUL_LAT so it aligns with m1 for the SUB-and-onward path
		logic [31:0]  XAfterMul1[MUL_LAT] = '{ default: 'x };
		always_ff @(posedge clk) begin
			XAfterMul1[0] <= x_chain[i];
			for(int  d = 1; d < MUL_LAT; d++)  XAfterMul1[d] <= XAfterMul1[d-1];
		end

		// m2 = 2 - m1.  binopf SUB: a=2.0 (constant), b=m1, bload=avld.
		uwire [31:0]  m2;
		binopf #(.OP("SUB")) sub1 (
			.clk, .rst,
			.a(FP32_TWO), .avld('1),
			.b(m1),       .bload('1),
			.r(m2), .rvld()
		);

		// Delay y by MUL_LAT + SUB_LAT to align with m2 for the final MUL
		localparam int unsigned  YDLY = MUL_LAT + SUB_LAT;
		logic [31:0]  YDly[YDLY] = '{ default: 'x };
		always_ff @(posedge clk) begin
			YDly[0] <= y_chain[i];
			for(int  d = 1; d < YDLY; d++)  YDly[d] <= YDly[d-1];
		end

		// y_new = y_delayed * m2
		uwire [31:0]  y_new;
		binopf #(.OP("MUL")) mul2 (
			.clk, .rst,
			.a(YDly[YDLY-1]), .avld('1),
			.b(m2),           .bload('1),
			.r(y_new), .rvld()
		);

		// Delay x by ITER_LAT to feed the next iteration aligned with y_new
		logic [31:0]  XPipe[ITER_LAT] = '{ default: 'x };
		always_ff @(posedge clk) begin
			XPipe[0] <= x_chain[i];
			for(int  d = 1; d < ITER_LAT; d++)  XPipe[d] <= XPipe[d-1];
		end

		assign	x_chain[i+1] = XPipe[ITER_LAT-1];
		assign	y_chain[i+1] = y_new;
	end : genIter

	//===========================================================================
	// Section: output valid pipeline + buffer
	//---------------------------------------------------------------------------
	logic [TOTAL_LAT-1:0]  Vld = '0;
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '0;
		else     Vld <= { Vld[$left(Vld)-1:0], take };
	end
	uwire  out_vld = Vld[$left(Vld)];
	uwire [31:0]  out_dat = y_chain[NR_ITERS];

	uwire  obuf_irdy;
	queue #(
		.DATA_WIDTH(32),
		.ELASTICITY(CREDIT)
	) obuf (
		.clk, .rst,
		.idat(out_dat), .ivld(out_vld), .irdy(obuf_irdy),
		.odat(rdat),    .ovld(rvld),    .ordy(rrdy)
	);
	assign	give = rvld && rrdy;

	always_ff @(posedge clk) begin
		assert(obuf_irdy || !out_vld) else begin
			$error("%m: output queue overrun.");
			$stop;
		end
	end

endmodule : softmaxf_recip


/****************************************************************************
 * Stage 4: ufixed exp value -> FP32 multiply by reciprocal.
 *
 * Consumes the per-beat ufixed<1+SUM_PRECISION,1> exp vectors from
 * stage 2 plus the per-vector FP32 reciprocal from softmaxf_recip and
 * emits the final FP32 softmax outputs.
 *
 * Per beat:
 *   - Combinational ufixed<24,1> -> FP32 conversion per lane.
 *   - Per-lane binopf MUL (latency 3) by the broadcast reciprocal,
 *     captured into the D register on the first-beat take of each
 *     vector window (mirrors stage 2's bload/mvld discipline).
 *
 * Infinity handling is implicit: stage 2 already substitutes 1.0 for
 * +inf lanes and 0 for others when the vector contains an infinity, and
 * the reciprocal stage computes 1/inf_count when has_infty is set.  The
 * straight per-lane multiply therefore yields the correct
 *   inf_count^-1  for +inf lanes
 *   0             elsewhere
 * without an explicit bypass mux.
 ***************************************************************************/
module softmaxf_div #(
	int unsigned  N,
	int unsigned  SIMD
)(
	input	logic  clk,
	input	logic  rst,

	// Per-beat ufixed exp vectors (from stage 2)
	input	logic [SIMD-1:0][23:0]  ydat,
	input	logic                   yvld,
	output	logic                   yrdy,

	// Per-vector FP32 reciprocal (from softmaxf_recip)
	input	logic [31:0]  rdat,
	input	logic         rvld,
	output	logic         rrdy,

	// Per-beat FP32 softmax outputs
	output	logic [SIMD-1:0][31:0]  zdat,
	output	logic                   zvld,
	input	logic                   zrdy
);

	localparam int unsigned  EXP_W   = 24;
	localparam int unsigned  BEATS   = N / SIMD;
	localparam int unsigned  MUL_LAT = 3;
	localparam int unsigned  IN_LAT  = 1;                       // y_fp register stage
	localparam int unsigned  CREDIT  = MUL_LAT + IN_LAT + 3;

	initial begin
		if(N % SIMD != 0) begin
			$error("%m: N=%0d must be divisible by SIMD=%0d", N, SIMD);
			$finish;
		end
	end

	//===========================================================================
	// Section: ufixed<24,1> -> FP32 conversion (combinational)
	//   - Zero input  -> +0.0
	//   - Otherwise: locate leading 1, normalise mantissa, exponent = 127 - lz.
	//---------------------------------------------------------------------------
	function automatic logic [31:0] ufixed24_to_fp32(input logic [EXP_W-1:0] x);
		automatic int unsigned  lz;
		automatic logic [EXP_W-1:0]  shifted;
		if(x == '0)  return 32'h00000000;
		lz = 0;
		for(int i = EXP_W - 1; i >= 0; i--) begin
			if(x[i]) begin
				lz = (EXP_W - 1) - i;
				break;
			end
		end
		shifted = EXP_W'(x << lz);
		return  { 1'b0, 8'(127 - lz), shifted[22:0] };
	endfunction

	//===========================================================================
	// Section: input handshake / sidestep / vector-aligned scheduling
	//   Mirrors softmaxf_exp's pattern: a sidestep buffer holds y, and `take`
	//   is gated by credit and (on the first beat) by reciprocal validity.
	//---------------------------------------------------------------------------
	typedef logic [SIMD-1:0][EXP_W-1:0]  ufx_vec_t;

	uwire  take;

	typedef struct {
		ufx_vec_t  val;
		logic      rdy;
	} ibuf_t;
	ibuf_t  Ibuf = '{ val: 'x, rdy: '1 };
	always_ff @(posedge clk) begin
		if(rst)  Ibuf <= '{ val: 'x, rdy: '1 };
		else begin
			if(Ibuf.rdy)  Ibuf.val <= ydat;
			Ibuf.rdy <= (Ibuf.rdy && !yvld) || take;
		end
	end
	assign	yrdy = Ibuf.rdy;
	uwire ufx_vec_t  y_cur = Ibuf.rdy? ydat : Ibuf.val;

	logic [$clog2(BEATS+1)-1:0]  InBeatCnt = '0;
	uwire  in_is_first = (InBeatCnt == '0);
	uwire  in_is_last  = (InBeatCnt == BEATS-1);
	always_ff @(posedge clk) begin
		if(rst)        InBeatCnt <= '0;
		else if(take)  InBeatCnt <= in_is_last? '0 : InBeatCnt + 1;
	end

	logic signed [$clog2(CREDIT):0]  CreditZ = -CREDIT;
	uwire  give_z;

	assign	take = (yvld || !Ibuf.rdy)
	             && CreditZ[$left(CreditZ)]
	             && (!in_is_first || rvld);

	always_ff @(posedge clk) begin
		if(rst)  CreditZ <= -CREDIT;
		else     CreditZ <= CreditZ + (give_z == take? 0 : give_z? -1 : 1);
	end

	//===========================================================================
	// Section: per-lane FP32 multiply (binopf MUL with broadcast reciprocal)
	//   YfpReg breaks the long combinational path that runs from softmaxf_exp's
	//   y_obuf register through ufixed24_to_fp32 into the DSP B inputs.  The
	//   take/first gating is registered alongside y_fp so binopf's bload of
	//   rdat aligns with YfpReg in the same cycle.
	//---------------------------------------------------------------------------
	uwire [SIMD-1:0][31:0]  y_fp;
	for(genvar  i = 0; i < SIMD; i++) begin : genConv
		assign	y_fp[i] = ufixed24_to_fp32(y_cur[i]);
	end : genConv

	logic [SIMD-1:0][31:0]  YfpReg  = '{ default: 'x };
	logic                   TakeD1  = 1'b0;
	logic                   FirstD1 = 1'b0;
	always_ff @(posedge clk) begin
		YfpReg <= y_fp;
		if(rst) begin
			TakeD1  <= 1'b0;
			FirstD1 <= 1'b0;
		end
		else begin
			TakeD1  <= take;
			FirstD1 <= take && in_is_first;
		end
	end

	// Pop the reciprocal queue when binopf actually captures rdat -- one
	// cycle after the first-beat take, due to the YfpReg register stage.
	// The recip queue holds its head until popped, so rdat remains valid
	// across the extra cycle.
	assign	rrdy = FirstD1;

	uwire [SIMD-1:0][31:0]  prod;
	for(genvar  i = 0; i < SIMD; i++) begin : genMul
		binopf #(.OP("MUL")) mul_inst (
			.clk, .rst,
			.a(YfpReg[i]), .avld(TakeD1),
			.b(rdat),      .bload(FirstD1),
			.r(prod[i]), .rvld()
		);
	end : genMul

	logic [MUL_LAT+IN_LAT-1:0]  Vld = '0;
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '0;
		else     Vld <= { Vld[$left(Vld)-1:0], take };
	end
	uwire  out_vld = Vld[$left(Vld)];

	//===========================================================================
	// Section: output buffer
	//---------------------------------------------------------------------------
	uwire  obuf_irdy;
	queue #(
		.DATA_WIDTH(SIMD * 32),
		.ELASTICITY(CREDIT)
	) obuf (
		.clk, .rst,
		.idat(prod), .ivld(out_vld), .irdy(obuf_irdy),
		.odat(zdat), .ovld(zvld),    .ordy(zrdy)
	);
	assign	give_z = zvld && zrdy;

	always_ff @(posedge clk) begin
		assert(obuf_irdy || !out_vld) else begin
			$error("%m: output queue overrun.");
			$stop;
		end
	end

endmodule : softmaxf_div

/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	One-dimensional padding for folded token streams.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	For each sequence, emits PAD_LEFT_TOKENS tokens from PAD_LEFT_DATA,
 *	NUM_TOKENS input tokens, and PAD_RIGHT_TOKENS tokens from
 *	PAD_RIGHT_DATA.
 *
 *	Each token is transferred as NUM_CHANNELS/SIMD folds of SIMD
 *	ELEM_WIDTH-bit elements.  All packed data dimensions are
 *	little-endian.
 ***************************************************************************/

module pad1d #(
	int unsigned  NUM_TOKENS,
	int unsigned  NUM_CHANNELS,
	int unsigned  ELEM_WIDTH,
	int unsigned  SIMD = 1,
	int unsigned  PAD_LEFT_TOKENS = 0,
	int unsigned  PAD_RIGHT_TOKENS = 0,

	localparam int unsigned  PAD_LEFT_DEPTH  = PAD_LEFT_TOKENS  > 0? PAD_LEFT_TOKENS  : 1,
	localparam int unsigned  PAD_RIGHT_DEPTH = PAD_RIGHT_TOKENS > 0? PAD_RIGHT_TOKENS : 1,
	parameter logic [PAD_LEFT_DEPTH -1:0][NUM_CHANNELS-1:0][ELEM_WIDTH-1:0]  PAD_LEFT_DATA  = '0,
	parameter logic [PAD_RIGHT_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_WIDTH-1:0]  PAD_RIGHT_DATA = '0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	output	logic  irdy,
	input	logic  ivld,
	input	logic [SIMD-1:0][ELEM_WIDTH-1:0]  idat,

	// Output Stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [SIMD-1:0][ELEM_WIDTH-1:0]  odat
);

	//=== Parameter Validation ==============================================
	initial begin
		automatic bit  fail = 0;

		if(NUM_TOKENS < 1) begin
			$error("%m: NUM_TOKENS must be at least 1.");
			fail = 1;
		end
		if(NUM_CHANNELS < 1) begin
			$error("%m: NUM_CHANNELS must be at least 1.");
			fail = 1;
		end
		if(SIMD < 1) begin
			$error("%m: SIMD must be at least 1.");
			fail = 1;
		end
		if(ELEM_WIDTH < 1) begin
			$error("%m: ELEM_WIDTH must be at least 1.");
			fail = 1;
		end
		if(NUM_CHANNELS % SIMD != 0) begin
			$error("%m: SIMD must divide NUM_CHANNELS.");
			fail = 1;
		end
		if(fail)  $finish;
	end

	if(PAD_LEFT_TOKENS + PAD_RIGHT_TOKENS == 0) begin : genTrivial
		// No Padding: simple passthru wiring
		assign	irdy = ordy;
		assign	ovld = ivld;
		assign	odat = idat;
	end : genTrivial
	else begin : genPad
		// At least one side is padded
		typedef logic [SIMD-1:0][ELEM_WIDTH-1:0]  fold_t;

		//=== Input Sidestep Register Stage ======================================
		fold_t  ADat = 'x;
		logic   AVld = 0;

		uwire  issue_input;
		always_ff @(posedge clk) begin
			if(rst) begin
				ADat <= 'x;
				AVld <= 0;
			end
			else begin
				ADat <= AVld? ADat : idat;
				AVld <= (AVld || (irdy && ivld)) && !issue_input;
			end
		end

		//=== Output Sequencing =================================================
		localparam int unsigned  NUM_PHASES = 1 + (PAD_LEFT_TOKENS > 0) + (PAD_RIGHT_TOKENS > 0);
		logic [$clog2(NUM_PHASES)-1:0]  Phase = 0;
		uwire  phase_lpad = PAD_LEFT_TOKENS  && !Phase;
		uwire  phase_rpad = PAD_RIGHT_TOKENS && Phase[$left(Phase)];
		uwire  phase_thru = !phase_lpad && !phase_rpad;

		localparam int unsigned  FOLDS_PER_TOKEN = NUM_CHANNELS / SIMD;
		localparam int unsigned  MAX_PAD_TOKENS = (PAD_LEFT_TOKENS > PAD_RIGHT_TOKENS)? PAD_LEFT_TOKENS : PAD_RIGHT_TOKENS;
		localparam int unsigned  MAX_PHASE_TOKENS = (NUM_TOKENS > MAX_PAD_TOKENS)? NUM_TOKENS : MAX_PAD_TOKENS;
		localparam int unsigned  CNT_WIDTH = $clog2(MAX_PHASE_TOKENS * FOLDS_PER_TOKEN - 1) + 1;

		typedef logic signed [CNT_WIDTH-1:0]  cnt_t;  // TOKENS*FOLDS-2, .., 1, 0, -1 (last)
		typedef int  cnt_wrap_t[NUM_PHASES];
		function cnt_wrap_t INIT_CNT_WRAP();
			automatic cnt_wrap_t  res;
			automatic int unsigned  phase_tokens[NUM_PHASES];
			if(PAD_LEFT_TOKENS)  phase_tokens[0] = PAD_LEFT_TOKENS;
			phase_tokens[(PAD_LEFT_TOKENS > 0)] = NUM_TOKENS;
			if(PAD_RIGHT_TOKENS)  phase_tokens[(PAD_LEFT_TOKENS > 0)+1] = PAD_RIGHT_TOKENS;
			foreach(res[i])  res[i] = 1 - int'(FOLDS_PER_TOKEN) * int'(phase_tokens[(i + 1) % NUM_PHASES]);
			return  res;
		endfunction : INIT_CNT_WRAP
		localparam cnt_wrap_t  CNT_WRAP = INIT_CNT_WRAP();

		cnt_t  Cnt = -1 - CNT_WRAP[$high(CNT_WRAP)];
		always_ff @(posedge clk) begin
			if(rst) begin
				Phase <= 0;
				Cnt <= -1 - CNT_WRAP[$high(CNT_WRAP)];
			end
			else if(issue) begin
				automatic logic  cnt_last = Cnt[$left(Cnt)];
				Cnt <= Cnt - (!cnt_last? 1 : CNT_WRAP[Phase]);
				Phase <= Phase + (
					!cnt_last? 0 :
					PAD_LEFT_TOKENS && PAD_RIGHT_TOKENS && phase_rpad? 2 : 1
				);
			end
		end

		//=== Output Register Stage =============================================
		// Keep the padding lookup proportional to the padding itself.  Indexing
		// this table with the full sequence counter would allocate space for the
		// (usually much longer) input phase as well, even though those entries can
		// never be read while padding is active.
		localparam int unsigned  MAX_PAD_FOLDS = MAX_PAD_TOKENS * FOLDS_PER_TOKEN;
		localparam int unsigned  PAD_FOLD_IDX_WIDTH = MAX_PAD_FOLDS > 1? $clog2(MAX_PAD_FOLDS) : 1;
		typedef logic [PAD_FOLD_IDX_WIDTH-1:0]  pad_fold_idx_t;
		typedef fold_t  pad_folds_t[2][MAX_PAD_FOLDS];
		function pad_folds_t INIT_PAD_FOLDS();
			automatic pad_folds_t  f = '{default: 'x};
			for(int unsigned  t = 0; t < PAD_LEFT_TOKENS; t++)
				for(int unsigned  s = 0; s < FOLDS_PER_TOKEN; s++) begin
					f[0][t * FOLDS_PER_TOKEN + s] = PAD_LEFT_DATA[t][s * SIMD +: SIMD];
				end
			for(int unsigned  t = 0; t < PAD_RIGHT_TOKENS; t++)
				for(int unsigned  s = 0; s < FOLDS_PER_TOKEN; s++) begin
					f[1][t * FOLDS_PER_TOKEN + s] = PAD_RIGHT_DATA[t][s * SIMD +: SIMD];
				end
			return  f;
		endfunction
		localparam pad_folds_t  PAD_FOLDS = INIT_PAD_FOLDS();
		uwire pad_fold_idx_t  pad_fold_idx = pad_fold_idx_t'(
			int'(phase_rpad? PAD_RIGHT_TOKENS : PAD_LEFT_TOKENS) * int'(FOLDS_PER_TOKEN)
			- 2 - int'(Cnt)
		);

		fold_t  BDat = 'x;
		logic   BVld = 0;

		uwire  bload = !BVld || ordy;
		// A left pad starts a new frame.  Wait for that frame's first input
		// beat before emitting it; otherwise an idle input would cause a
		// speculative pad-only frame after every completed sequence.  The
		// producer keeps ivld asserted until phase_thru raises irdy and accepts
		// the same first beat.  Right padding, in contrast, must finish the
		// frame that has already been accepted.
		uwire  issue = bload && (phase_thru? (AVld || ivld) : (!phase_lpad || ivld));
		always_ff @(posedge clk) begin
			if(rst) begin
				BDat <= 'x;
				BVld <= 0;
			end
			else if(bload) begin
				BVld <= issue;
				if(issue) begin
					BDat <= !phase_thru?
						PAD_FOLDS[phase_rpad][pad_fold_idx] :
						AVld? ADat : idat;
				end
			end
		end
		assign	issue_input = issue && phase_thru;

		assign	irdy = phase_thru && !AVld;
		assign	odat = BDat;
		assign	ovld = BVld;

	end : genPad

endmodule : pad1d

/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	One-dimensional padding for folded token streams.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 *
 * @description
 *	For each sequence, emits PAD_LEFT left-padding tokens, NUM_TOKENS input
 *	tokens, and PAD_RIGHT right-padding tokens.  A leading special token is
 *	represented as custom left padding by setting PAD_LEFT to one and driving
 *	pad_left_data with that token.
 *
 *	Each token is transferred as NUM_CHANNELS/SIMD folds of SIMD
 *	ELEM_WIDTH-bit elements.  pad_left_data and pad_right_data are packed
 *	token arrays with token 0 in the least-significant TOKEN_WIDTH bits.
 *	If a pad count is zero, the corresponding input is still one token wide.
 ***************************************************************************/

module pad1d #(
	int unsigned  NUM_TOKENS = 196,
	int unsigned  NUM_CHANNELS = 192,
	int unsigned  SIMD = 1,
	int unsigned  ELEM_WIDTH = 8,
	int unsigned  PAD_LEFT = 0,
	int unsigned  PAD_RIGHT = 0,

	localparam int unsigned  FOLD_WIDTH = SIMD * ELEM_WIDTH,
	localparam int unsigned  TOKEN_WIDTH = NUM_CHANNELS * ELEM_WIDTH,
	localparam int unsigned  PAD_LEFT_DATA_TOKENS = (PAD_LEFT == 0)? 1 : PAD_LEFT,
	localparam int unsigned  PAD_RIGHT_DATA_TOKENS = (PAD_RIGHT == 0)? 1 : PAD_RIGHT,
	localparam int unsigned  PAD_LEFT_DATA_WIDTH = PAD_LEFT_DATA_TOKENS * TOKEN_WIDTH,
	localparam int unsigned  PAD_RIGHT_DATA_WIDTH = PAD_RIGHT_DATA_TOKENS * TOKEN_WIDTH
)(
	input	logic  clk,
	input	logic  rst,

	output	logic  irdy,
	input	logic  ivld,
	input	logic [FOLD_WIDTH-1:0]  idat,

	input	logic  ordy,
	output	logic  ovld,
	output	logic [FOLD_WIDTH-1:0]  odat,

	// Packed pad token inputs; see header for layout.
	input	logic [PAD_LEFT_DATA_WIDTH-1:0]  pad_left_data,
	input	logic [PAD_RIGHT_DATA_WIDTH-1:0]  pad_right_data
);

	localparam int unsigned  FOLDS_PER_TOKEN = NUM_CHANNELS / SIMD;
	localparam int unsigned  MAX_PAD_TOKENS = (PAD_LEFT > PAD_RIGHT)? PAD_LEFT : PAD_RIGHT;
	localparam int unsigned  MAX_PHASE_TOKENS = (NUM_TOKENS > MAX_PAD_TOKENS)? NUM_TOKENS : MAX_PAD_TOKENS;
	localparam int unsigned  FOLD_CNT_WIDTH = (FOLDS_PER_TOKEN <= 1)? 1 : $clog2(FOLDS_PER_TOKEN-1)+1;
	localparam int unsigned  TOKEN_CNT_WIDTH = (MAX_PHASE_TOKENS <= 1)? 1 : $clog2(MAX_PHASE_TOKENS-1)+1;

	//=== Parameter Validation ==============================================
	initial begin
		bit  fail;

		fail = 0;

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
		if((SIMD > 0) && (NUM_CHANNELS % SIMD != 0)) begin
			$error("%m: SIMD must divide NUM_CHANNELS.");
			fail = 1;
		end
		if(fail)  $finish;
	end

	typedef logic [FOLD_WIDTH-1:0]  fold_t;
	typedef logic [TOKEN_WIDTH-1:0]  token_t;
	typedef logic signed [FOLD_CNT_WIDTH-1:0]   fold_cnt_t;
	typedef logic signed [TOKEN_CNT_WIDTH-1:0]  token_cnt_t;

	typedef enum logic [1:0] {
		PHASE_LEFT_PAD,
		PHASE_INPUT,
		PHASE_RIGHT_PAD
	} phase_e;

	localparam fold_cnt_t  FOLD_INIT = fold_cnt_t'(int'(FOLDS_PER_TOKEN) - 2);
	localparam token_cnt_t  LEFT_TOKEN_INIT = token_cnt_t'(int'(PAD_LEFT) - 2);
	localparam token_cnt_t  INPUT_TOKEN_INIT = token_cnt_t'(int'(NUM_TOKENS) - 2);
	localparam token_cnt_t  RIGHT_TOKEN_INIT = token_cnt_t'(int'(PAD_RIGHT) - 2);
	localparam phase_e  FIRST_PHASE = (PAD_LEFT != 0)? PHASE_LEFT_PAD : PHASE_INPUT;
	localparam token_cnt_t  FIRST_TOKEN_INIT = (PAD_LEFT != 0)? LEFT_TOKEN_INIT : INPUT_TOKEN_INIT;

	function automatic token_t select_pad_token(
		input phase_e      phase,
		input token_cnt_t  token_cnt
	);
		automatic int signed  pad_idx_signed;
		automatic int unsigned  pad_idx;
		begin
			pad_idx_signed = 0;
			if(phase == PHASE_LEFT_PAD) begin
				pad_idx_signed = int'(PAD_LEFT) - 2 - int'(token_cnt);
				pad_idx = (pad_idx_signed < 0)? 0 : int'(pad_idx_signed);
				select_pad_token = pad_left_data[pad_idx*TOKEN_WIDTH +: TOKEN_WIDTH];
			end
			else if(phase == PHASE_RIGHT_PAD) begin
				pad_idx_signed = int'(PAD_RIGHT) - 2 - int'(token_cnt);
				pad_idx = (pad_idx_signed < 0)? 0 : int'(pad_idx_signed);
				select_pad_token = pad_right_data[pad_idx*TOKEN_WIDTH +: TOKEN_WIDTH];
			end
			else begin
				select_pad_token = 'x;
			end
		end
	endfunction : select_pad_token

	//=== Output Sequence State =============================================
	phase_e      Phase = FIRST_PHASE;
	token_cnt_t  TokenCnt = FIRST_TOKEN_INIT;
	fold_cnt_t   FoldCnt = FOLD_INIT;
	token_t      ConstToken = 'x;

	uwire  fold_last = FoldCnt[$left(FoldCnt)];
	uwire  token_last = TokenCnt[$left(TokenCnt)];
	uwire  input_token = Phase == PHASE_INPUT;

	phase_e      next_phase;
	token_cnt_t  next_token_cnt;
	always_comb begin
		case(Phase)
		PHASE_LEFT_PAD: begin
			next_phase = PHASE_INPUT;
			next_token_cnt = INPUT_TOKEN_INIT;
		end
		PHASE_INPUT: begin
			next_phase = (PAD_RIGHT != 0)? PHASE_RIGHT_PAD : FIRST_PHASE;
			next_token_cnt = (PAD_RIGHT != 0)? RIGHT_TOKEN_INIT : FIRST_TOKEN_INIT;
		end
		default: begin
			next_phase = FIRST_PHASE;
			next_token_cnt = FIRST_TOKEN_INIT;
		end
		endcase
	end

	uwire token_t  next_phase_pad_data = select_pad_token(next_phase, next_token_cnt);
	uwire token_t  next_token_pad_data = select_pad_token(Phase, token_cnt_t'(TokenCnt - 1));

	//=== Input Sidestep Register Stage ======================================
	fold_t  ADat = 'x;
	logic   AVld = 0;

	//=== Output Register Stage =============================================
	fold_t  BDat = 'x;
	logic   BVld = 0;

	uwire  bload = !BVld || ordy;
	uwire  source_vld = AVld || ivld;
	uwire  issue = bload && (!input_token || source_vld);
	uwire  issue_input = issue && input_token;
	uwire  issue_from_a = issue_input && AVld;
	uwire  capture_a = input_token && !AVld && ivld && !issue_input;

	assign	irdy = input_token && !AVld;

	always_ff @(posedge clk) begin
		if(rst) begin
			ADat <= 'x;
			AVld <= 0;
		end
		else begin
			if(issue_from_a) begin
				ADat <= 'x;
				AVld <= 0;
			end
			else if(capture_a) begin
				ADat <= idat;
				AVld <= 1;
			end
		end
	end

	assign	odat = BDat;
	assign	ovld = BVld;

	always_ff @(posedge clk) begin
		if(rst) begin
			BDat <= 'x;
			BVld <= 0;
		end
		else if(bload) begin
			BVld <= issue;
			if(issue)  BDat <= !input_token? ConstToken[FOLD_WIDTH-1:0] : AVld? ADat : idat;
		end
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Phase <= FIRST_PHASE;
			TokenCnt <= FIRST_TOKEN_INIT;
			FoldCnt <= FOLD_INIT;
			ConstToken <= (FIRST_PHASE == PHASE_LEFT_PAD)?
				select_pad_token(FIRST_PHASE, FIRST_TOKEN_INIT) : 'x;
		end
		else if(issue) begin
			if(fold_last) begin
				FoldCnt <= FOLD_INIT;
				if(token_last) begin
					Phase <= next_phase;
					TokenCnt <= next_token_cnt;
					ConstToken <= (next_phase == PHASE_INPUT)? 'x : next_phase_pad_data;
				end
				else begin
					TokenCnt <= TokenCnt - 1;
					if(input_token)  ConstToken <= 'x;
					else             ConstToken <= next_token_pad_data;
				end
			end
			else begin
				FoldCnt <= FoldCnt - 1;
				if(!input_token)  ConstToken <= token_t'(ConstToken >> FOLD_WIDTH);
			end
		end
	end

endmodule : pad1d

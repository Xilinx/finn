/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 * @brief	Select one token from a folded token stream.
 *
 * @description
 *	Consumes NUM_TOKENS token vectors, each consisting of TOKEN_BEATS stream
 *	beats. Beats belonging to TOKEN_INDEX are forwarded to the output; all
 *	other beats are consumed and discarded.
 ***************************************************************************/

`default_nettype none

module select_token #(
	int unsigned  NUM_TOKENS,
	int unsigned  TOKEN_BEATS,
	int unsigned  DATA_WIDTH,
	int unsigned  TOKEN_INDEX
)(
	// Global Control
	input	wire logic  clk,
	input	wire logic  rst,

	// Input Stream
	output	logic  irdy,
	input	wire logic  ivld,
	input	wire logic [DATA_WIDTH-1:0]  idat,

	// Output Stream - beats belonging to TOKEN_INDEX
	input	wire logic  ordy,
	output	logic  ovld,
	output	logic [DATA_WIDTH-1:0]  odat
);

	localparam int unsigned  TOKEN_CNT_BITS = (NUM_TOKENS <= 1)? 1 : $clog2(NUM_TOKENS);
	localparam int unsigned  BEAT_CNT_BITS = (TOKEN_BEATS <= 1)? 1 : $clog2(TOKEN_BEATS);
	typedef logic [TOKEN_CNT_BITS-1:0]  token_cnt_t;
	typedef logic [  BEAT_CNT_BITS-1:0]  beat_cnt_t;
	typedef logic [DATA_WIDTH-1:0]  data_t;
	localparam token_cnt_t  TOKEN_INDEX_PRE = (TOKEN_INDEX == 0)? NUM_TOKENS-1 : TOKEN_INDEX-1;
	localparam token_cnt_t  TOKEN_PRE_LAST  = (NUM_TOKENS == 1)? 0 : NUM_TOKENS-2;
	localparam beat_cnt_t   BEAT_PRE_LAST   = (TOKEN_BEATS == 1)? 0 : TOKEN_BEATS-2;

	initial begin
		if(NUM_TOKENS < 1) begin
			$error("%m: NUM_TOKENS must be positive.");
			$finish;
		end
		if(TOKEN_BEATS < 1) begin
			$error("%m: TOKEN_BEATS must be positive.");
			$finish;
		end
		if(DATA_WIDTH < 1) begin
			$error("%m: DATA_WIDTH must be positive.");
			$finish;
		end
		if(TOKEN_INDEX >= NUM_TOKENS) begin
			$error("%m: TOKEN_INDEX must be less than NUM_TOKENS.");
			$finish;
		end
	end

	// Beat and Token Position
	token_cnt_t  TokenCnt = '0; // 0, ..., NUM_TOKENS-1
	beat_cnt_t   BeatCnt  = '0; // 0, ..., TOKEN_BEATS-1
	logic  Selected = TOKEN_INDEX == 0; // TokenCnt == TOKEN_INDEX
	logic  BeatLst  = TOKEN_BEATS == 1; // BeatCnt == TOKEN_BEATS-1
	logic  TokenLst = NUM_TOKENS == 1; // TokenCnt == NUM_TOKENS-1

	// Selected-Token Forwarding
	data_t  ADat = 'x;
	logic   AVld = 0;
	data_t  BDat = 'x;
	logic   BVld = 0;

	assign	irdy = !Selected || !AVld;
	assign	odat = BDat;
	assign	ovld = BVld;

	uwire  take = irdy && ivld;
	uwire  selected_take = Selected && take;
	uwire  bload = !BVld || ordy;

	always_ff @(posedge clk) begin
		if(rst) begin
			TokenCnt <= '0;
			BeatCnt <= '0;
			Selected <= TOKEN_INDEX == 0;
			BeatLst <= TOKEN_BEATS == 1;
			TokenLst <= NUM_TOKENS == 1;
		end
		else if(take) begin
			if(BeatLst) begin
				BeatCnt <= '0;
				BeatLst <= TOKEN_BEATS == 1;
				Selected <= TokenCnt == TOKEN_INDEX_PRE;
				if(TokenLst) begin
					TokenCnt <= '0;
					TokenLst <= NUM_TOKENS == 1;
				end
				else begin
					TokenCnt <= TokenCnt + 1;
					TokenLst <= TokenCnt == TOKEN_PRE_LAST;
				end
			end
			else begin
				BeatCnt <= BeatCnt + 1;
				BeatLst <= BeatCnt == BEAT_PRE_LAST;
			end
		end
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			ADat <= 'x;
			AVld <= 0;
			BDat <= 'x;
			BVld <= 0;
		end
		else begin
			if(bload) begin
				BDat <= AVld? ADat : idat;
				BVld <= AVld || selected_take;
			end

			if(bload)  AVld <= 0;
			else if(selected_take) begin
				ADat <= idat;
				AVld <= 1;
			end
		end
	end

endmodule : select_token

`default_nettype wire

/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

module weights_buff_tile #(
	int unsigned  WEIGHT_WIDTH = 8,
	int unsigned  SIMD,
	int unsigned  PE,
	int unsigned  TH,
	int unsigned  WSIMD,
	int unsigned  NW = (PE*SIMD)/WSIMD,
	int unsigned  N_DCPL_STAGES
)(
	input	logic  clk,
	input	logic  rst,

	input   logic  ivld,
	output  logic  irdy,
	input   logic [WSIMD-1:0][WEIGHT_WIDTH-1:0]  idat,

	output  logic  ovld,
	input   logic  ordy,
	output  logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  odat
);

	//=== Parameter Validation ==============================================
	initial begin
		if((PE*SIMD) % WSIMD != 0) begin
			$error("Weight stream width not set properly (WSIMD: %0d, PE %0d, SIMD %0d).", WSIMD, PE, SIMD);
			$finish;
		end
	end

	//=== Constants and Types ===============================================
	localparam int unsigned  NW_BITS = (NW == 1)? 1 : $clog2(NW);
	localparam int unsigned  TH_BITS = (TH == 1)? 1 : $clog2(TH);

	typedef enum logic [1:0] {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT}  state_wr_e;
	typedef enum logic       {ST_RD_0, ST_RD_1}  state_rd_e;

	//=== Input Slice =======================================================
	uwire  ivld_int;
	logic  irdy_int;
	uwire [WSIMD-1:0][WEIGHT_WIDTH-1:0]  idat_int;

	skid #(.DATA_WIDTH(WSIMD*WEIGHT_WIDTH), .FEED_STAGES(1)) inst_ireg (
		.clk(clk), .rst(rst),
		.ivld(ivld), .irdy(irdy), .idat(idat),
		.ovld(ivld_int), .ordy(irdy_int), .odat(idat_int)
	);

	//=== Writer ============================================================
	state_wr_e  StateWr = ST_WR_0;
	state_wr_e  state_wr_n;
	state_rd_e  StateRd = ST_RD_0;
	state_rd_e  state_rd_n;

	logic [NW_BITS-1:0]  Curr = '0;
	logic [NW_BITS-1:0]  curr_n;

	logic  done;

	logic  ovld_int;
	logic  ordy_int;
	logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  odat_int;

	logic [1:0][NW-1:0][WSIMD*WEIGHT_WIDTH-1:0]  Mem = '0;
	logic [1:0][NW-1:0][WSIMD*WEIGHT_WIDTH-1:0]  mem_n;

	//--- Writer Sequential -------------------------------------------------
	always_ff @(posedge clk) begin
		if(rst) begin
			StateWr <= ST_WR_0;
			Curr    <= '0;
			Mem     <= '0;
		end
		else begin
			StateWr <= state_wr_n;
			Curr    <= curr_n;
			Mem     <= mem_n;
		end
	end

	//--- Writer Next State -------------------------------------------------
	always_comb begin
		state_wr_n = StateWr;

		case(StateWr)
		ST_WR_0:
			if((Curr == NW - 1) && ivld_int)
				state_wr_n = (done || (StateRd == ST_RD_0))? ST_WR_1 : ST_WR_0_WAIT;

		ST_WR_0_WAIT:
			state_wr_n = (done || (StateRd == ST_RD_0))? ST_WR_1 : ST_WR_0_WAIT;

		ST_WR_1:
			if((Curr == NW - 1) && ivld_int)
				state_wr_n = (done || (StateRd == ST_RD_1))? ST_WR_0 : ST_WR_1_WAIT;

		ST_WR_1_WAIT:
			state_wr_n = (done || (StateRd == ST_RD_1))? ST_WR_0 : ST_WR_1_WAIT;
		endcase
	end

	//--- Writer Datapath ---------------------------------------------------
	always_comb begin
		curr_n   = Curr;
		mem_n    = Mem;
		irdy_int = 0;

		case(StateWr)
		ST_WR_0, ST_WR_1: begin
			irdy_int = 1;

			if(ivld_int) begin
				if(StateWr == ST_WR_0) begin
					mem_n[0]      = (Mem[0] >> WSIMD*WEIGHT_WIDTH);
					mem_n[0][NW-1] = idat_int;
				end
				else begin
					mem_n[1]      = (Mem[1] >> WSIMD*WEIGHT_WIDTH);
					mem_n[1][NW-1] = idat_int;
				end

				curr_n = (Curr == NW-1)? 0 : Curr + 1;
			end
		end
		endcase
	end

	//=== Reader ============================================================
	logic [TH_BITS-1:0]  ConsR = '0;
	logic [TH_BITS-1:0]  cons_r_n;

	//--- Reader Sequential -------------------------------------------------
	always_ff @(posedge clk) begin
		if(rst) begin
			StateRd <= ST_RD_0;
			ConsR   <= 0;
		end
		else begin
			StateRd <= state_rd_n;
			ConsR   <= cons_r_n;
		end
	end

	//--- Reader Next State -------------------------------------------------
	always_comb begin
		state_rd_n = StateRd;

		case(StateRd)
		ST_RD_0:
			if(ordy_int && (StateWr != ST_WR_0))
				if(ConsR == TH-1)
					state_rd_n = ST_RD_1;

		ST_RD_1:
			if(ordy_int && (StateWr != ST_WR_1))
				if(ConsR == TH-1)
					state_rd_n = ST_RD_0;
		endcase
	end

	//--- Reader Datapath ---------------------------------------------------
	always_comb begin
		cons_r_n = ConsR;
		done     = 0;
		ovld_int = 0;
		odat_int = 0;

		case(StateRd)
		ST_RD_0:
			if(ordy_int && (StateWr != ST_WR_0)) begin
				ovld_int = 1;
				odat_int = Mem[0];
				done     = (ConsR == TH-1);
				cons_r_n = (ConsR == TH-1)? 0 : ConsR + 1;
			end

		ST_RD_1:
			if(ordy_int && (StateWr != ST_WR_1)) begin
				ovld_int = 1;
				odat_int = Mem[1];
				done     = (ConsR == TH-1);
				cons_r_n = (ConsR == TH-1)? 0 : ConsR + 1;
			end
		endcase
	end

	//=== Output Slice ======================================================
	skid #(.DATA_WIDTH(PE*SIMD*WEIGHT_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg (
		.clk(clk), .rst(rst),
		.ivld(ovld_int), .irdy(ordy_int), .idat(odat_int),
		.ovld(ovld), .ordy(ordy), .odat(odat)
	);

endmodule : weights_buff_tile

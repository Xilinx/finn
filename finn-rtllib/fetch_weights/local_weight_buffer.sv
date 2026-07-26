/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/

module local_weight_ram #(
	int unsigned  DATA_WIDTH,
	int unsigned  DEPTH,
	int unsigned  ADDR_WIDTH = (DEPTH == 1)? 1 : $clog2(DEPTH),
	parameter  RAM_STYLE = "block"
)(
	input   logic  clk,
	input   logic  w_en,
	input   logic [ADDR_WIDTH-1:0]  w_addr,
	input   logic [DATA_WIDTH-1:0]  w_data,
	input   logic  r_en,
	input   logic [ADDR_WIDTH-1:0]  r_addr,
	output  logic [DATA_WIDTH-1:0]  r_data
);

	(* RAM_STYLE = RAM_STYLE *)
	logic [DATA_WIDTH-1:0]  Ram[DEPTH];

	always_ff @(posedge clk) begin
		if(w_en)  Ram[w_addr] <= w_data;
		if(r_en)  r_data <= Ram[r_addr];
	end

endmodule : local_weight_ram

module local_weight_buffer #(
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  WEIGHT_WIDTH = 8,
	int unsigned  MH,
	int unsigned  MW,
	int unsigned  N_REPS,
	int unsigned  DBG = 0,
	parameter  RAM_STYLE = "block"
)(
	input   logic  clk,
	input   logic  rst,

	input   logic  ivld,
	output  logic  irdy,
	input   logic [SIMD-1:0][WEIGHT_WIDTH-1:0]  idat,

	output  logic  ovld,
	input   logic  ordy,
	output  logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  odat
);

	//=== Constants and Types ===============================================
	localparam int unsigned  SF            = MW / SIMD;
	localparam int unsigned  NF            = MH / PE;
	localparam int unsigned  N_TLS         = SF * NF;
	localparam int unsigned  PE_BITS       = (PE == 1)? 1 : $clog2(PE);
	localparam int unsigned  WGT_ADDR_BITS = $clog2(NF * SF);
	localparam int unsigned  N_TLS_BITS    = $clog2(N_TLS);
	localparam int unsigned  N_REPS_BITS   = $clog2(N_REPS);
	localparam int unsigned  WGT_WORD_BITS = SIMD * WEIGHT_WIDTH;
	localparam int unsigned  MAX_RAM_BITS  = 1_000_000;
	localparam int unsigned  MAX_BANK_DEPTH = (WGT_WORD_BITS > MAX_RAM_BITS)?
	                                           1 : MAX_RAM_BITS / WGT_WORD_BITS;
	localparam int unsigned  BANK_ADDR_BITS = (MAX_BANK_DEPTH == 1)?
	                                           0 : $clog2(MAX_BANK_DEPTH + 1) - 1;
	localparam int unsigned  BANK_DEPTH     = 2**BANK_ADDR_BITS;
	localparam int unsigned  N_RAM_BANKS    = (N_TLS + BANK_DEPTH - 1) / BANK_DEPTH;
	localparam int unsigned  RAM_BANK_BITS  = (N_RAM_BANKS == 1)? 1 : $clog2(N_RAM_BANKS);

	typedef enum logic [1:0] {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT}  state_wr_e;
	typedef enum logic       {ST_RD_0, ST_RD_1}  state_rd_e;

	//=== Writer ============================================================

	//--- Registers ---------------------------------------------------------
	state_wr_e  StateWr = ST_WR_0;
	state_wr_e  state_wr_n;
	state_rd_e  StateRd = ST_RD_0;
	state_rd_e  state_rd_n;

	logic [N_TLS_BITS-1:0]  WrPntr = '0;
	logic [N_TLS_BITS-1:0]  wr_pntr_n;

	logic [PE_BITS-1:0]  CurrPe = '0;
	logic [PE_BITS-1:0]  curr_pe_n;

	//--- Signals -----------------------------------------------------------
	logic [1:0][PE-1:0]                      a_we;
	logic [1:0][WGT_ADDR_BITS-1:0]           a_addr;
	logic [1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  a_data_in;

	//--- Sequential --------------------------------------------------------
	always_ff @(posedge clk) begin
		if(rst) begin
			StateWr <= ST_WR_0;
			WrPntr  <= '0;
			CurrPe  <= '0;
		end
		else begin
			StateWr <= state_wr_n;
			WrPntr  <= wr_pntr_n;
			CurrPe  <= curr_pe_n;
		end
	end

	//--- Next State --------------------------------------------------------
	always_comb begin
		state_wr_n = StateWr;

		case(StateWr)
		ST_WR_0:
			if((CurrPe == PE-1) && (WrPntr == N_TLS-1) && ivld)
				state_wr_n = (StateRd == ST_RD_0)? ST_WR_1 : ST_WR_0_WAIT;

		ST_WR_0_WAIT:
			state_wr_n = (StateRd == ST_RD_0)? ST_WR_1 : ST_WR_0_WAIT;

		ST_WR_1:
			if((CurrPe == PE-1) && (WrPntr == N_TLS-1) && ivld)
				state_wr_n = (StateRd == ST_RD_1)? ST_WR_0 : ST_WR_1_WAIT;

		ST_WR_1_WAIT:
			state_wr_n = (StateRd == ST_RD_1)? ST_WR_0 : ST_WR_1_WAIT;
		endcase
	end

	//--- Datapath ----------------------------------------------------------
	always_comb begin
		wr_pntr_n = WrPntr;
		curr_pe_n = CurrPe;

		irdy = 0;

		a_we = '0;
		for(int i = 0; i < 2; i++) begin
			a_addr[i]    = WrPntr;
			a_data_in[i] = idat;
		end

		case(StateWr)
		ST_WR_0, ST_WR_1: begin
			irdy = 1;

			if(ivld) begin
				for(int i = 0; i < PE; i++)
					if(CurrPe == i)
						a_we[StateWr == ST_WR_1][i] = 1;

				curr_pe_n = (CurrPe == PE-1)? 0 : CurrPe + 1;
				wr_pntr_n = (CurrPe == PE-1)? ((WrPntr == N_TLS-1)? 0 : WrPntr + 1) : WrPntr;
			end
		end
		endcase
	end

	//=== Reader ============================================================

	//--- Registers ---------------------------------------------------------
	logic [N_TLS_BITS-1:0]   RdPntr = '0;
	logic [N_TLS_BITS-1:0]   rd_pntr_n;

	logic [N_REPS_BITS-1:0]  Reps = '0;
	logic [N_REPS_BITS-1:0]  reps_n;

	logic [1:0]  VldS0 = '0;
	logic [1:0]  vld_s0_n;

	logic [1:0]  VldS1 = '0;
	logic [1:0]  vld_s1_n;

	logic  Vld = 0;
	logic  vld_n;

	logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  Odat = '0;
	logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  odat_n;

	//--- Signals -----------------------------------------------------------
	logic [1:0][WGT_ADDR_BITS-1:0]                    b_addr;
	logic [1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  odat_ram;

	//--- Sequential --------------------------------------------------------
	always_ff @(posedge clk) begin
		if(rst) begin
			StateRd <= ST_RD_0;
			RdPntr  <= '0;
			Reps    <= '0;
			VldS0   <= '0;
			VldS1   <= '0;
			Vld     <= 0;
			Odat    <= 'x;
		end
		else begin
			StateRd <= state_rd_n;
			RdPntr  <= rd_pntr_n;
			Reps    <= reps_n;
			VldS0   <= vld_s0_n;
			VldS1   <= vld_s1_n;
			Vld     <= vld_n;
			Odat    <= odat_n;
		end
	end

	//--- Next State --------------------------------------------------------
	always_comb begin
		state_rd_n = StateRd;

		case(StateRd)
		ST_RD_0:
			if(ordy && ((StateWr == ST_WR_0)? (WrPntr > RdPntr) : 1))
				if((RdPntr == N_TLS-1) && (Reps == N_REPS-1))
					state_rd_n = ST_RD_1;

		ST_RD_1:
			if(ordy && ((StateWr == ST_WR_1)? (WrPntr > RdPntr) : 1))
				if((RdPntr == N_TLS-1) && (Reps == N_REPS-1))
					state_rd_n = ST_RD_0;
		endcase
	end

	//--- Datapath ----------------------------------------------------------
	always_comb begin
		rd_pntr_n = RdPntr;
		reps_n    = Reps;

		for(int i = 0; i < 2; i++) begin
			vld_s0_n[i] = ordy? 0 : VldS0[i];
			vld_s1_n[i] = ordy? VldS0[i] : VldS1[i];
		end

		vld_n  = ordy? |VldS1 : Vld;
		odat_n = ordy? (VldS1[0]? odat_ram[0] : odat_ram[1]) : Odat;

		for(int i = 0; i < 2; i++)
			b_addr[i] = RdPntr;

		case(StateRd)
		ST_RD_0: begin
			if(ordy) begin
				if((StateWr == ST_WR_0)? (WrPntr > RdPntr) : 1) begin
					vld_s0_n[0] = 1;
					rd_pntr_n   = (RdPntr == N_TLS-1)? 0 : RdPntr + 1;
					reps_n      = (RdPntr == N_TLS-1)? ((Reps == N_REPS-1)? 0 : Reps + 1) : Reps;
				end
			end
		end

		ST_RD_1: begin
			if(ordy) begin
				if((StateWr == ST_WR_1)? (WrPntr > RdPntr) : 1) begin
					vld_s0_n[1] = 1;
					rd_pntr_n   = (RdPntr == N_TLS-1)? 0 : RdPntr + 1;
					reps_n      = (RdPntr == N_TLS-1)? ((Reps == N_REPS-1)? 0 : Reps + 1) : Reps;
				end
			end
		end
		endcase
	end

	assign	ovld = Vld;
	assign	odat = Odat;

	//=== Weight RAMs =======================================================
	for(genvar i = 0; i < 2; i++) begin : genBank
		for(genvar j = 0; j < PE; j++) begin : genPe
			if(N_TLS * WGT_WORD_BITS <= MAX_RAM_BITS) begin : genSingle
				logic [SIMD-1:0][WEIGHT_WIDTH-1:0]  RdData;

				local_weight_ram #(
					.DATA_WIDTH(WGT_WORD_BITS), .DEPTH(N_TLS), .RAM_STYLE(RAM_STYLE)
				) inst_ram (
					.clk(clk),
					.w_en(a_we[i][j]), .w_addr(a_addr[i]), .w_data(a_data_in[i]),
					.r_en(ordy), .r_addr(b_addr[i]), .r_data(RdData)
				);

				always_ff @(posedge clk) begin
					if(ordy)  odat_ram[i][j] <= RdData;
				end
			end
			else begin : genBanked
				logic [N_RAM_BANKS-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  BankRdData;
				logic [RAM_BANK_BITS-1:0]  ReadBank;

				for(genvar k = 0; k < N_RAM_BANKS; k++) begin : genRam
					localparam int unsigned  THIS_BANK_DEPTH = (k == N_RAM_BANKS-1)?
					                                               N_TLS - k * BANK_DEPTH : BANK_DEPTH;
					localparam int unsigned  THIS_ADDR_BITS = (THIS_BANK_DEPTH == 1)?
					                                             1 : $clog2(THIS_BANK_DEPTH);
					logic [THIS_ADDR_BITS-1:0]  LocalWAddr;
					logic [THIS_ADDR_BITS-1:0]  LocalRAddr;

					assign  LocalWAddr = a_addr[i] % BANK_DEPTH;
					assign  LocalRAddr = b_addr[i] % BANK_DEPTH;

					local_weight_ram #(
						.DATA_WIDTH(WGT_WORD_BITS), .DEPTH(THIS_BANK_DEPTH), .RAM_STYLE(RAM_STYLE)
					) inst_ram (
						.clk(clk),
						.w_en(a_we[i][j] && (a_addr[i] / BANK_DEPTH == k)),
						.w_addr(LocalWAddr), .w_data(a_data_in[i]),
						.r_en(ordy && (b_addr[i] / BANK_DEPTH == k)),
						.r_addr(LocalRAddr), .r_data(BankRdData[k])
					);
				end

				always_ff @(posedge clk) begin
					if(ordy) begin
						ReadBank       <= b_addr[i] / BANK_DEPTH;
						odat_ram[i][j] <= BankRdData[ReadBank];
					end
				end
			end
		end : genPe
	end : genBank

endmodule : local_weight_buffer

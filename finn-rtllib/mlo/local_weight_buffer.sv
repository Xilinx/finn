// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

module local_weight_buffer #(
    parameter int unsigned              PE,
    parameter int unsigned              SIMD,
    parameter int unsigned              WEIGHT_WIDTH = 8,
    parameter int unsigned              MH,
    parameter int unsigned              MW,
    parameter int unsigned              N_REPS,
    parameter int unsigned              DBG = 0
) (
    input	logic  clk,
    input	logic  rst,

    input   logic  ivld,
    output  logic  irdy,
    input   logic  [PE-1:0][WEIGHT_WIDTH-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat
);

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------

localparam int unsigned  SF = MW/SIMD;
localparam int unsigned  NF = MH/PE;
localparam int unsigned  N_TLS = SF * NF;

localparam int unsigned SIMD_BITS = (SIMD == 1) ? 1 : $clog2(SIMD);
localparam int unsigned WGT_ADDR_BITS = $clog2(NF * SF);
localparam int unsigned RAM_BITS = (PE*WEIGHT_WIDTH + 7)/8 * 8;
localparam int unsigned WGT_EN_BITS = RAM_BITS / 8;
localparam int unsigned N_TLS_BITS = $clog2(N_TLS);
localparam int unsigned N_REPS_BITS = $clog2(N_REPS);

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

//logic[N_TLS_BITS-1:0] wr_pntr_C = '0, wr_pntr_N;
//logic[SIMD_BITS-1:0] curr_simd_C = '0, curr_simd_N;

logic[15:0] wr_pntr_C = '0, wr_pntr_N;
logic[15:0] curr_simd_C = '0, curr_simd_N;

// -- Signals
logic [1:0][SIMD-1:0][WGT_EN_BITS-1:0] a_we; // Bank enables
logic [1:0][WGT_ADDR_BITS-1:0] a_addr;
logic [1:0][PE-1:0][WEIGHT_WIDTH-1:0] a_data_in;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_WR
    if(rst) begin
        state_wr_C <= ST_WR_0;

        wr_pntr_C <= '0;
        curr_simd_C <= '0;
    end
    else begin
        state_wr_C <= state_wr_N;

        wr_pntr_C <= wr_pntr_N;
        curr_simd_C <= curr_simd_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_0:
            if((curr_simd_C == SIMD - 1) && (wr_pntr_C == N_TLS - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if((curr_simd_C == SIMD - 1) && (wr_pntr_C == N_TLS - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    wr_pntr_N = wr_pntr_C;
    curr_simd_N = curr_simd_C;

    // Input
    irdy = 1'b0;

    // Buffers
    a_we = '0;
    for(int i = 0; i < 2; i++) begin
        a_addr[i] = wr_pntr_C;
        a_data_in[i] = idat;
    end

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy = 1'b1;

            if(ivld) begin
                for(int i = 0; i < SIMD; i++) begin
                    if(curr_simd_C == i) begin
                        a_we[state_wr_C == ST_WR_1][i] = '1;
                    end
                end

                curr_simd_N = (curr_simd_C == SIMD-1) ? 0 : curr_simd_C + 1;
                wr_pntr_N = (curr_simd_C == SIMD-1) ? ((wr_pntr_C == N_TLS-1) ? 0 : wr_pntr_C + 1) : wr_pntr_C;
            end
        end
    endcase

end

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

// -- Regs
//logic [N_TLS_BITS-1:0] rd_pntr_C = '0, rd_pntr_N;
//logic [N_REPS_BITS-1:0] reps_C = '0, reps_N;

logic [15:0] rd_pntr_C = '0, rd_pntr_N;
logic [15:0] reps_C = '0, reps_N;

logic [1:0] vld_s0_C = '0, vld_s0_N;
logic [1:0] vld_s1_C = '0, vld_s1_N;

logic vld_C = '0, vld_N;
logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_C = '0, odat_N;

// -- Signals
logic [1:0][WGT_ADDR_BITS-1:0] b_addr;
logic [1:0][SIMD-1:0][PE-1:0][WEIGHT_WIDTH-1:0] odat_ram;
logic [1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_ram_arr;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_RD
    if(rst) begin
        state_rd_C <= ST_RD_0;

        rd_pntr_C <= '0;
        reps_C <= '0;

        vld_s0_C <= '0;
        vld_s1_C <= '0;
        vld_C <= '0;
        odat_C <= 'X;
    end
    else begin
        state_rd_C <= state_rd_N;

        rd_pntr_C <= rd_pntr_N;
        reps_C <= reps_N;

        vld_s0_C <= vld_s0_N;
        vld_s1_C <= vld_s1_N;
        vld_C <= vld_N;
        odat_C <= odat_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_0:
            if(ordy && ((state_wr_C == ST_WR_0) ? (wr_pntr_C > rd_pntr_C) : 1'b1)) begin
                if((rd_pntr_C == N_TLS-1) && (reps_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_1;
                end
            end

        ST_RD_1:
            if(ordy && ((state_wr_C == ST_WR_1) ? (wr_pntr_C > rd_pntr_C) : 1'b1)) begin
                if((rd_pntr_C == N_TLS-1) && (reps_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_0;
                end
            end
    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    rd_pntr_N = rd_pntr_C;
    reps_N = reps_C;

    for(int i = 0; i < 2; i++) begin
        vld_s0_N[i] = ordy ? 1'b0 : vld_s0_C[i];
        vld_s1_N[i] = ordy ? vld_s0_C[i] : vld_s1_C[i];
    end

    vld_N = ordy ? |vld_s1_C : vld_C;
    odat_N = ordy ? (vld_s1_C[0] ? odat_ram_arr[0] : odat_ram_arr[1]) : odat_C;

    for(int i = 0; i < 2; i++) begin
        b_addr[i] = rd_pntr_C;
    end

    case(state_rd_C)
        ST_RD_0: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_0) ? (wr_pntr_C > rd_pntr_C) : 1'b1) begin

                    vld_s0_N[0] = 1'b1;

                    rd_pntr_N = (rd_pntr_C == N_TLS-1) ? 0 : rd_pntr_C + 1;
                    reps_N = (rd_pntr_C == N_TLS-1) ? ((reps_C == N_REPS-1) ? 0 : reps_C + 1) : reps_C;
                end
            end
        end

        ST_RD_1: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_1) ? (wr_pntr_C > rd_pntr_C) : 1'b1) begin

                    vld_s0_N[1] = 1'b1;

                    rd_pntr_N = (rd_pntr_C == N_TLS-1) ? 0 : rd_pntr_C + 1;
                    reps_N = (rd_pntr_C == N_TLS-1) ? ((reps_C == N_REPS-1) ? 0 : reps_C + 1) : reps_C;
                end
            end
        end

    endcase

end

assign ovld = vld_C;
assign odat = odat_C;

// ----------------------------------------------------------------------------
// Weights
// ----------------------------------------------------------------------------

for(genvar i = 0; i < 2; i++) begin
    for(genvar j = 0; j < SIMD; j++) begin
        ram_p_c #(
            .ADDR_BITS(WGT_ADDR_BITS),
            .DATA_BITS(RAM_BITS),
            .RAM_STYLE("block")
        ) inst_ram_tp_c (
            .clk(clk),
            .a_en(1'b1),
            .a_we(a_we[i][j]),
            .a_addr(a_addr[i]),
            .b_en(ordy),
            .b_addr(b_addr[i]),
            .a_data_in(a_data_in[i]),
            .a_data_out(),
            .b_data_out(odat_ram[i][j])
        );

        for(genvar k = 0; k < PE; k++) begin
            assign odat_ram_arr[i][k][j] = odat_ram[i][j][k];
        end
    end
end

endmodule

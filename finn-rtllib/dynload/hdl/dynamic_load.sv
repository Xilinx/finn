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

module dynamic_load #(
    int unsigned  PE,
    int unsigned  SIMD,
    int unsigned  WEIGHT_WIDTH,
    int unsigned  MH,
    int unsigned  MW,
    int unsigned  N_REPS
)(
    input	logic  ap_clk,
    input	logic  ap_rst_n,

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
localparam int unsigned  N_TLS = SF*NF;

localparam int unsigned SIMD_BITS = (SIMD == 1) ? 1 : $clog2(SIMD);
localparam int unsigned WGT_ADDR_BITS = (N_TLS == 1) ? 1 : $clog2(N_TLS);
localparam int unsigned RAM_BITS = (WEIGHT_WIDTH + 7)/8 * 8;
localparam int unsigned WGT_EN_BITS = RAM_BITS / 8;
localparam int unsigned NF_BITS = (NF == 1) ? 1 : $clog2(NF);
localparam int unsigned SF_BITS = (SF == 1) ? 1 : $clog2(SF);
localparam int unsigned N_TLS_BITS = (N_TLS == 1) ? 1 : $clog2(N_TLS);
localparam int unsigned N_REPS_BITS = (N_REPS == 1) ? 1 : $clog2(N_REPS);

logic [NF-1:0][WGT_ADDR_BITS-1:0] offsets;

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

logic[NF_BITS-1:0] curr_nf_C = '0, curr_nf_N;
logic[N_TLS_BITS-1:0] curr_sf_C = '0, curr_sf_N;
logic[SIMD_BITS-1:0] curr_simd_C = '0, curr_simd_N;

// -- Signals
logic [1:0][PE-1:0][SIMD-1:0][WGT_EN_BITS-1:0] a_we; // Bank enables
logic [1:0][WGT_ADDR_BITS-1:0] a_addr;
logic [1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] a_data_in;

// -- Offsets
for(genvar i = 0; i < NF; i++) begin
    assign offsets[i] = i * SF;
end

// -- REG
always_ff @( posedge ap_clk ) begin : REG_PROC_WR
    if(~ap_rst_n) begin
        state_wr_C <= ST_WR_0;

        curr_nf_C <= 0;
        curr_sf_C <= 0;
        curr_simd_C <= 0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_nf_C <= curr_nf_N;
        curr_sf_C <= curr_sf_N;
        curr_simd_C <= curr_simd_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    unique case (state_wr_C)
        ST_WR_0:
            if ((curr_simd_C == SIMD - 1) && (curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if ((curr_simd_C == SIMD - 1) && (curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_nf_N = curr_nf_C;
    curr_sf_N = curr_sf_C;
    curr_simd_N = curr_simd_C;

    // Input
    irdy = 1'b0;

    // Buffers
    a_we = '0;
    for(int i = 0; i < 2; i++) begin
        a_addr[i] = offsets[curr_nf_C] + curr_sf_C;
        for(int j = 0; j < PE; j++)
            for(int k = 0; k < SIMD; k++)
                a_data_in[i][j][k] = idat[j];
    end

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy = 1'b1;

            if(ivld) begin
                for(int i = 0; i < PE; i++) begin
                    for(int j = 0; j < SIMD; j++) begin
                        if(curr_simd_C == j) begin
                            if(state_wr_C == ST_WR_0)
                                a_we[0][i][j] = '1;
                            else
                                a_we[1][i][j] = '1;
                        end
                    end
                end

                curr_nf_N   = (curr_nf_C == NF-1) ? 0 : curr_nf_C + 1;
                curr_simd_N = (curr_nf_C == NF-1) ? ((curr_simd_C == SIMD-1) ? 0 : curr_simd_C + 1) : curr_simd_C;
                curr_sf_N   = (curr_nf_C == NF-1) ? ((curr_simd_C == SIMD-1) ? ((curr_sf_C == SF-1) ? 0 : curr_sf_C + 1) : curr_sf_C) : curr_sf_C;
            end
        end
    endcase

end

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

// -- Regs
logic [N_TLS_BITS-1:0] cons_sfnf_C = '0, cons_sfnf_N;
logic [N_REPS_BITS-1:0] cons_r_C = '0, cons_r_N;

logic [1:0] vld_s0_C = '0, vld_s0_N;
logic [1:0] vld_s1_C = '0, vld_s1_N;

logic vld_C = '0, vld_N;
logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_C = '0, odat_N;

// -- Signals
logic [1:0][WGT_ADDR_BITS-1:0] b_addr;
logic [1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_ram;

// -- REG
always_ff @( posedge ap_clk ) begin : REG_PROC_RD
    if(~ap_rst_n) begin
        state_rd_C <= ST_RD_0;

        cons_sfnf_C <= 0;
        cons_r_C  <= 0;

        vld_s0_C <= 0;
        vld_s1_C <= 0;
        vld_C <= 0;
        odat_C <= 0;
    end
    else begin
        state_rd_C <= state_rd_N;

        cons_sfnf_C <= cons_sfnf_N;
        cons_r_C  <= cons_r_N;

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
            if(ordy && ((state_wr_C != ST_WR_0) || (curr_sf_C > cons_sfnf_C))) begin
                if((cons_sfnf_C == N_TLS-1) && (cons_r_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_1;
                end
            end

        ST_RD_1:
            if(ordy && ((state_wr_C != ST_WR_1) || (curr_sf_C > cons_sfnf_C))) begin
                if((cons_sfnf_C == N_TLS-1) && (cons_r_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_0;
                end
            end

    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    cons_sfnf_N = cons_sfnf_C;
    cons_r_N = cons_r_C;

    for(int i = 0; i < 2; i++) begin
        vld_s0_N[i] = ordy ? 1'b0 : vld_s0_C[i];
        vld_s1_N[i] = ordy ? vld_s0_C[i] : vld_s1_C[i];
    end

    vld_N = ordy ? |vld_s1_C : vld_C;
    odat_N = ordy ? (vld_s1_C[0] ? odat_ram[0] : odat_ram[1]) : odat_C;

    for(int i = 0; i < 2; i++) begin
        b_addr[i] = cons_sfnf_C;
    end

    case(state_rd_C)
        ST_RD_0: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_0) ? (curr_sf_C > cons_sfnf_C) : 1'b1) begin
                    vld_s0_N[0] = 1'b1;

                    cons_sfnf_N = (cons_sfnf_C == N_TLS-1) ? 0 : cons_sfnf_C + 1;
                    cons_r_N = (cons_sfnf_C == N_TLS-1) ? ((cons_r_C == N_REPS-1) ? 0 : cons_r_C + 1) : cons_r_C;
                end
            end
        end

        ST_RD_1: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_1) ? (curr_sf_C > cons_sfnf_C) : 1'b1) begin

                    vld_s0_N[1] = 1'b1;

                    cons_sfnf_N = (cons_sfnf_C == N_TLS-1) ? 0 : cons_sfnf_C + 1;
                    cons_r_N = (cons_sfnf_C == N_TLS-1) ? ((cons_r_C == N_REPS-1) ? 0 : cons_r_C + 1) : cons_r_C;
                end
            end
        end

    endcase

end

assign ovld = vld_C;
assign odat = odat_C;

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

for(genvar i = 0; i < 2; i++) begin
    for(genvar j = 0; j < PE; j++) begin
        for(genvar k = 0; k < SIMD; k++) begin
            ram_p_c #(
                .ADDR_BITS(WGT_ADDR_BITS),
                .DATA_BITS(RAM_BITS),
                .RAM_STYLE("distributed")
            ) inst_ram_tp_c (
                .clk(ap_clk),
                .a_en(1'b1),
                .a_we(a_we[i][j][k]),
                .a_addr(a_addr[i]),
                .b_en(ordy),
                .b_addr(b_addr[i]),
                .a_data_in(a_data_in[i][j][k]),
                .a_data_out(),
                .b_data_out(odat_ram[i][j][k])
            );
        end
    end
end

endmodule : dynamic_load

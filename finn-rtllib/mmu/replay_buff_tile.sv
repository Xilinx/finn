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

/*****************************************************************************
 * @brief	Buffer the inner matrix
 * @author	Dario Korolija <dario.korolija@amd.com>
 *****************************************************************************/

module replay_buff_tile #(
    int unsigned  SF,
    int unsigned  NF,
    int unsigned  SIMD,
    int unsigned  ACTIVATION_WIDTH,
    int unsigned  N_RPLYS
)(
	input	logic  clk,
	input	logic  rst,

    input   logic  ivld,
    output  logic  irdy,
    input   logic  [SIMD-1:0][ACTIVATION_WIDTH-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [SIMD-1:0][ACTIVATION_WIDTH-1:0] odat,
    output  logic  olast
);

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------
localparam integer ADDR_BITS = $clog2(NF * SF);
localparam integer EN_BITS = SIMD * ACTIVATION_WIDTH / 8;

localparam integer SF_BITS = $clog2(SF);
localparam integer NF_BITS = $clog2(NF);
localparam integer SF_NF_BITS = $clog2(SF*NF);
localparam integer N_RPLYS_BITS = $clog2(N_RPLYS);

logic [NF-1:0][ADDR_BITS-1:0] offsets;

typedef enum logic  {ST_WR_0, ST_WR_1} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;
typedef logic [NF_BITS:0] nf_t;
typedef logic [SF_BITS:0] sf_t;
typedef logic [SF_NF_BITS:0] sf_nf_rd_t;
typedef logic [N_RPLYS_BITS:0] reps_t;

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C, state_wr_N;

nf_t curr_nf_C, curr_nf_N;
sf_t curr_sf_C, curr_sf_N;

logic rd_0_C, rd_0_N;
logic rd_1_C, rd_1_N;

// -- Signals
logic [EN_BITS-1:0] a_we_0; // Bank enables
logic [ADDR_BITS-1:0] a_addr_0;
logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] a_data_in_0;

logic [EN_BITS-1:0] a_we_1; // Bank enables
logic [ADDR_BITS-1:0] a_addr_1;
logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] a_data_in_1;

logic done_0, done_1; // Completion

// -- Offsets
for(genvar i = 0; i < NF; i++) begin
    assign offsets[i] = i * SF;
end

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_WR
    if(rst) begin
        state_wr_C <= ST_WR_0;

        curr_nf_C <= 0;
        curr_sf_C <= 0;

        rd_0_C <= 1'b0;
        rd_1_C <= 1'b0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_nf_C <= curr_nf_N;
        curr_sf_C <= curr_sf_N;

        rd_0_C <= rd_0_N;
        rd_1_C <= rd_1_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_0:
            state_wr_N = ((curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld && ~rd_0_C) ? ST_WR_1 : ST_WR_0;

        ST_WR_1:
            state_wr_N = ((curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld && ~rd_1_C) ? ST_WR_0 : ST_WR_1;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_nf_N = curr_nf_C;
    curr_sf_N = curr_sf_C;

    rd_0_N = done_0 ? 1'b0 : rd_0_C;
    rd_1_N = done_1 ? 1'b0 : rd_1_C;

    irdy = 1'b0;

    a_we_0 = 0;
    a_addr_0 = offsets[curr_nf_C] + curr_sf_C;
    a_data_in_0 = idat;

    a_we_1 = 0;
    a_addr_1 = offsets[curr_nf_C] + curr_sf_C;
    a_data_in_1 = idat;

    case (state_wr_C)
        ST_WR_0: begin
            if(~rd_0_C) begin
                irdy = 1'b1;

                if(ivld) begin
                    a_we_0 = '1;

                    curr_nf_N = (curr_nf_C == NF-1) ? 0 : curr_nf_C + 1;
                    curr_sf_N = (curr_nf_C == NF-1) ? curr_sf_C + 1 : curr_sf_C;

                    if((curr_sf_C == SF - 1) && (curr_nf_C == NF - 1)) begin
                        rd_0_N = 1'b1;

                        curr_nf_N = 0;
                        curr_sf_N = 0;
                    end
                end
            end
        end

        ST_WR_1: begin
            if(~rd_1_C) begin
                irdy = 1'b1;

                if(ivld && ~rd_1_C) begin
                    a_we_1 = '1;

                    curr_nf_N = (curr_nf_C == NF-1) ? 0 : curr_nf_C + 1;
                    curr_sf_N = (curr_nf_C == NF-1) ? curr_sf_C + 1 : curr_sf_C;

                    if((curr_sf_C == SF - 1) && (curr_nf_C == NF - 1)) begin
                        rd_1_N = 1'b1;

                        curr_nf_N = 0;
                        curr_sf_N = 0;
                    end
                end
            end
        end
        
    endcase
end

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

// -- Regs
state_rd_t state_rd_C, state_rd_N;

sf_nf_rd_t cons_C, cons_N;
reps_t cons_r_C, cons_r_N;

logic vld_0_s0_C, vld_0_s0_N;
logic vld_0_s1_C, vld_0_s1_N;

logic vld_1_s0_C, vld_1_s0_N;
logic vld_1_s1_C, vld_1_s1_N;

logic vld_C, vld_N;
logic [SIMD*ACTIVATION_WIDTH-1:0] odat_C, odat_N;

logic last_s0_C, last_s0_N;
logic last_s1_C, last_s1_N;
logic last_C, last_N;

// -- Signals
logic [ADDR_BITS-1:0] b_addr_0, b_addr_1;
logic [SIMD*ACTIVATION_WIDTH-1:0] odat_0, odat_1;

logic [SIMD*ACTIVATION_WIDTH-1:0] odat_int;
logic ovld_int;
logic ordy_int;
logic olast_int;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_RD
    if(rst) begin
        state_rd_C <= ST_RD_0;

        cons_C <= 0;
        cons_r_C  <= 0;

        vld_0_s0_C <= 0;
        vld_0_s1_C <= 0;
        vld_1_s0_C <= 0;
        vld_1_s1_C <= 0;
        vld_C <= 0;
        odat_C <= 0;
        last_s0_C <= 0;
        last_s1_C <= 0;
        last_C <= 0;
    end
    else begin
        state_rd_C <= state_rd_N;

        cons_C <= cons_N;
        cons_r_C  <= cons_r_N;

        vld_0_s0_C <= vld_0_s0_N;
        vld_0_s1_C <= vld_0_s1_N;
        vld_1_s0_C <= vld_1_s0_N;
        vld_1_s1_C <= vld_1_s1_N;
        vld_C <= vld_N;
        odat_C <= odat_N;
        last_s0_C <= last_s0_N;
        last_s1_C <= last_s1_N;
        last_C <= last_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_0:
            if(rd_0_C) begin
                if(ordy_int) begin
                    if((cons_C == NF*SF-1) && (cons_r_C == N_RPLYS-1)) begin
                        state_rd_N = ST_RD_1;
                    end
                end
            end

        ST_RD_1:
            if(rd_1_C) begin
                if(ordy_int) begin
                    if((cons_C == NF*SF-1) && (cons_r_C == N_RPLYS-1)) begin
                        state_rd_N = ST_RD_0;
                    end
                end
            end
        
    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    cons_N = cons_C;
    cons_r_N = cons_r_C;

    vld_0_s0_N = ordy_int ? 1'b0 : vld_0_s0_C;
    vld_0_s1_N = ordy_int ? vld_0_s0_C : vld_0_s1_C;
    vld_1_s0_N = ordy_int ? 1'b0 : vld_1_s0_C;
    vld_1_s1_N = ordy_int ? vld_1_s0_C : vld_1_s1_C;
    vld_N = ordy_int ? (vld_0_s1_C | vld_1_s1_C) : vld_C;

    last_s0_N = ordy_int ? 1'b0 : last_s0_C;
    last_s1_N = ordy_int ? last_s0_C : last_s1_C;
    last_N = ordy_int ? last_s1_C : last_C;
    
    odat_N = ordy_int ? (vld_0_s1_C ? odat_0 : odat_1) : odat_C;
    
    b_addr_0 = cons_C;
    b_addr_1 = cons_C;

    done_0 = 1'b0;
    done_1 = 1'b0;

    case (state_rd_C)
        ST_RD_0: begin
            if(rd_0_C) begin
                if(ordy_int) begin
                    vld_0_s0_N = 1'b1;

                    cons_N = (cons_C == NF*SF-1) ? 0 : cons_C + 1;
                    cons_r_N = (cons_C == NF*SF-1) ? cons_r_C + 1 : cons_r_C;

                    if((cons_C == NF*SF-1) && (cons_r_C == N_RPLYS-1)) begin
                        done_0 = 1'b1;
                        cons_r_N = 0;
                    end

                    if(cons_C >= (SF*(NF-1))) begin
                        last_s0_N = 1'b1;
                    end
                end
            end
        end

        ST_RD_1: begin
            if(rd_1_C) begin
                if(ordy_int) begin
                    vld_1_s0_N = 1'b1;

                    cons_N = (cons_C == NF*SF-1) ? 0 : cons_C + 1;
                    cons_r_N = (cons_C == NF*SF-1) ? cons_r_C + 1 : cons_r_C;

                    if((cons_C == NF*SF-1) && (cons_r_C == N_RPLYS-1)) begin
                        done_1 = 1'b1;
                        cons_r_N = 0;
                    end

                    if(cons_C >= (SF*(NF-1))) begin
                        last_s0_N = 1'b1;
                    end
                end
            end
        end
    endcase
end

assign ovld_int = vld_C;
assign olast_int = last_C;
assign odat_int = odat_C;

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

ram_p_c #( 
    .ADDR_BITS(ADDR_BITS),
    .DATA_BITS(SIMD*ACTIVATION_WIDTH),
    .RAM_TYPE("block")
) inst_ram_tp_c_0 (
    .clk(clk),
    .a_en(1'b1),
    .a_we(a_we_0),
    .a_addr(a_addr_0),
    .b_en(ordy_int),
    .b_addr(b_addr_0),
    .a_data_in(a_data_in_0),
    .a_data_out(),
    .b_data_out(odat_0)
);

ram_p_c #( 
    .ADDR_BITS(ADDR_BITS),
    .DATA_BITS(SIMD*ACTIVATION_WIDTH),
    .RAM_TYPE("block")
) inst_ram_tp_c_1 (
    .clk(clk),
    .a_en(1'b1),
    .a_we(a_we_1),
    .a_addr(a_addr_1),
    .b_en(ordy_int),
    .b_addr(b_addr_1),
    .a_data_in(a_data_in_1),
    .a_data_out(),
    .b_data_out(odat_1)
);

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------

Q_srl #(
    .depth(2), .width(1+SIMD*ACTIVATION_WIDTH)
) inst_out_fifo (
    .clock(clk),
    .reset(rst),
    .count(count),
    .maxcount(maxcount),
    .i_d({olast_int, odat_int}),
    .i_v(ovld_int),
    .i_r(ordy_int),
    .o_d({olast, odat}),
    .o_v(ovld),
    .o_r(ordy)
);


endmodule
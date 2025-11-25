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
/******************************************************************************
 * @brief	Buffer the inner matrix
 * @author	Dario Korolija <dario.korolija@amd.com>
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
    input   logic  [WSIMD-1:0][WEIGHT_WIDTH-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat
);


//-------------------- Parameter sanity checks --------------------------------

initial begin
    if ((PE*SIMD) % WSIMD != 0) begin
        $error("Weight stream width not set properly (WSIMD: %0d, PE %0d, SIMD %0d).", WSIMD, PE, SIMD);
        $finish;
    end
end

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------

localparam integer NW_BITS = (NW == 1) ? 1 : $clog2(NW);
localparam integer TH_BITS = (TH == 1) ? 1 : $clog2(TH);

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Slice
// ----------------------------------------------------------------------------

logic ivld_int;
logic irdy_int;
logic [WSIMD-1:0][WEIGHT_WIDTH-1:0] idat_int;

// Ireg
skid #(.DATA_WIDTH(WSIMD*WEIGHT_WIDTH), .FEED_STAGES(1)) inst_ireg (
    .clk(clk), .rst(rst), 
    .ivld(ivld), .irdy(irdy), .idat(idat),
    .ovld(ivld_int), .ordy(irdy_int), .odat(idat_int)
);

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

logic [NW_BITS-1:0] curr_C = '0, curr_N;

logic done;

logic ovld_int;
logic ordy_int;
logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_int;

// -- Mem
logic [1:0][NW-1:0][WSIMD*WEIGHT_WIDTH-1:0] mem_C, mem_N;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_WR
    if(rst) begin
        state_wr_C <= ST_WR_0;

        curr_C <= '0;
        mem_C <= '0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_C <= curr_N;
        mem_C <= mem_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_0:
            if ((curr_C == NW - 1) && ivld_int) begin
                state_wr_N = (done || (state_rd_C == ST_RD_0)) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (done || (state_rd_C == ST_RD_0)) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if ((curr_C == NW - 1) && ivld_int) begin
                state_wr_N = (done || (state_rd_C == ST_RD_1)) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (done || (state_rd_C == ST_RD_1)) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_N = curr_C;
    mem_N = mem_C;

    // Input
    irdy_int = 1'b0;

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy_int = 1'b1;

            if(ivld_int) begin
                if(state_wr_C == ST_WR_0) begin
                    mem_N[0] = (mem_C[0] >> WSIMD*WEIGHT_WIDTH);
                    mem_N[0][NW-1] = idat_int;
                end
                else begin
                    mem_N[1] = (mem_C[1] >> WSIMD*WEIGHT_WIDTH);
                    mem_N[1][NW-1] = idat_int;
                end

                curr_N = (curr_C == NW-1) ? 0 : curr_C + 1;
            end
        end
    endcase
end

// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

// -- Regs
logic [TH_BITS-1:0] cons_r_C = '0, cons_r_N;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_RD
    if(rst) begin
        state_rd_C <= ST_RD_0;

        cons_r_C  <= 0;
    end
    else begin
        state_rd_C <= state_rd_N;

        cons_r_C  <= cons_r_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_0:
            if(ordy_int && (state_wr_C != ST_WR_0)) begin
                if(cons_r_C == TH-1) begin
                    state_rd_N = ST_RD_1;
                end
            end

        ST_RD_1:
            if(ordy_int && (state_wr_C != ST_WR_1)) begin
                if(cons_r_C == TH-1) begin
                    state_rd_N = ST_RD_0;
                end
            end

    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    cons_r_N = cons_r_C;

    done = 1'b0;

    ovld_int = 1'b0;
    odat_int = 0;

    case (state_rd_C)
        ST_RD_0: begin
            if(ordy_int && (state_wr_C != ST_WR_0)) begin
                ovld_int = 1'b1;
                odat_int = mem_C[0];

                done = (cons_r_C == TH-1);
                cons_r_N = (cons_r_C == TH-1) ? 0 : cons_r_C + 1;
            end
        end

        ST_RD_1: begin
            if(ordy_int && (state_wr_C != ST_WR_1)) begin
                ovld_int = 1'b1;
                odat_int = mem_C[1];

                done = (cons_r_C == TH-1);
                cons_r_N = (cons_r_C == TH-1) ? 0 : cons_r_C + 1;
            end
        end

    endcase
end

// Oreg
skid #(.DATA_WIDTH(PE*SIMD*WEIGHT_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg (
    .clk(clk), .rst(rst), 
    .ivld(ovld_int), .irdy(ordy_int), .idat(odat_int),
    .ovld(ovld), .ordy(ordy), .odat(odat)
);

endmodule
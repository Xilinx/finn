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
    int unsigned  ACTIVATION_WIDTH = 8,
    int unsigned  SIMD,
    int unsigned  PE,
    int unsigned  TH
)(
	input	logic  clk,
	input	logic  rst,

    input   logic  ivld,
    output  logic  irdy,
    input   logic  [PE*ACTIVATION_WIDTH-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [PE-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0] odat
);

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------

localparam integer SIMD_BITS = $clog2(SIMD);
localparam integer TH_BITS = $clog2(TH);

typedef enum logic  {ST_WR_0, ST_WR_1} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;
typedef logic [SIMD_BITS:0] simd_cnt_t;
typedef logic [TH_BITS:0] reps_cnt_t;

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C, state_wr_N;

simd_cnt_t curr_C, curr_N;

logic rd_0_C, rd_0_N;
logic rd_1_C, rd_1_N;

// -- Mem
logic [SIMD-1:0][PE*ACTIVATION_WIDTH-1:0] mem_0_C, mem_0_N;
logic [SIMD-1:0][PE*ACTIVATION_WIDTH-1:0] mem_1_C, mem_1_N;

// -- Signals
logic done_0, done_1; // Completion

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_WR
    if(rst) begin
        state_wr_C <= ST_WR_0;

        curr_C <= 0;
        mem_0_C <= 0;
        mem_1_C <= 0;

        rd_0_C <= 1'b0;
        rd_1_C <= 1'b0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_C <= curr_N;
        mem_0_C <= mem_0_N;
        mem_1_C <= mem_1_N;

        rd_0_C <= rd_0_N;
        rd_1_C <= rd_1_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_0:
            state_wr_N = ((curr_C == SIMD - 1) && ivld && ~rd_0_C) ? ST_WR_1 : ST_WR_0;

        ST_WR_1:
            state_wr_N = ((curr_C == SIMD - 1) && ivld && ~rd_1_C) ? ST_WR_0 : ST_WR_1;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_N = curr_C;
    mem_0_N = mem_0_C;
    mem_1_N = mem_1_C;

    rd_0_N = done_0 ? 1'b0 : rd_0_C;
    rd_1_N = done_1 ? 1'b0 : rd_1_C;

    irdy = 1'b0;

    case (state_wr_C)
        ST_WR_0: begin
            if(~rd_0_C) begin
                irdy = 1'b1;

                if(ivld) begin
                    mem_0_N = (mem_0_N >> PE*ACTIVATION_WIDTH);
                    mem_0_N[SIMD-1] = idat;
                    //mem_0_N[curr_C] = idat;

                    curr_N = (curr_C == SIMD-1) ? 0 : curr_C + 1;

                    if((curr_C == SIMD - 1)) begin
                        rd_0_N = 1'b1;

                        curr_N = 0;
                    end
                end
            end
        end

        ST_WR_1: begin
            if(~rd_1_C) begin
                irdy = 1'b1;

                if(ivld) begin
                    mem_1_N = (mem_1_N >> PE*ACTIVATION_WIDTH);
                    mem_1_N[SIMD-1] = idat;
                    //mem_1_N[curr_C] = idat;

                    curr_N = (curr_C == SIMD-1) ? 0 : curr_C + 1;

                    if((curr_C == SIMD - 1)) begin
                        rd_1_N = 1'b1;

                        curr_N = 0;
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

reps_cnt_t cons_r_C, cons_r_N;

logic [SIMD-1:0][PE-1:0][ACTIVATION_WIDTH-1:0] odat_int;

for(genvar i = 0; i < PE; i++) begin
    for(genvar j = 0; j < SIMD; j++) begin
        assign odat[i][j] = odat_int[j][i];
    end
end
/*
always_comb begin
    for(int i = 0; i < PE; i++) begin
        for(int j = 0; j < SIMD; j++) begin
            odat[i][j] = odat_int[j][i];
        end
    end
end
*/
//assign odat = odat_int;

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
            if(rd_0_C) begin
                if(ordy) begin
                    if(cons_r_C == TH-1) begin
                        state_rd_N = ST_RD_1;
                    end
                end
            end

        ST_RD_1:
            if(rd_1_C) begin
                if(ordy) begin
                    if(cons_r_C == TH-1) begin
                        state_rd_N = ST_RD_0;
                    end
                end
            end
        
    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    cons_r_N = cons_r_C;

    done_0 = 1'b0;
    done_1 = 1'b0;

    ovld = 1'b0;
    odat_int = mem_0_C;

    case (state_rd_C)
        ST_RD_0: begin
            if(rd_0_C) begin
                ovld = 1'b1;
                odat_int = mem_0_C;

                if(ordy) begin
                    cons_r_N = (cons_r_C == TH-1) ? 0 : cons_r_C + 1;

                    if(cons_r_C == TH-1) begin
                        done_0 = 1'b1;
                        cons_r_N = 0;
                    end
                end
            end
        end

        ST_RD_1: begin
            if(rd_1_C) begin
                ovld = 1'b1;
                odat_int = mem_1_C;

                if(ordy) begin
                    cons_r_N = (cons_r_C == TH-1) ? 0 : cons_r_C + 1;

                    if(cons_r_C == TH-1) begin
                        done_1 = 1'b1;
                        cons_r_N = 0;
                    end
                end
            end
        end
    endcase
end

endmodule
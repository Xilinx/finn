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
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
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

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

module reorder_out #(
    int unsigned  W,
    int unsigned  XC,
    int unsigned  YC
)(
	input	logic  clk,
	input	logic  rst,

    input   logic  ivld,
    output  logic  irdy,
    input   logic  [W-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [W-1:0] odat
);

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------

localparam int unsigned RAM_BITS = (W + 7)/8 * 8;
localparam int unsigned WGT_EN_BITS = RAM_BITS / 8;
localparam int unsigned XYC = XC * YC;
localparam int unsigned XCNT_BITS = (XC == 1) ? 1 : $clog2(XC);
localparam int unsigned YCNT_BITS = (YC == 1) ? 1 : $clog2(YC);
localparam int unsigned XYCNT_BITS = (XYC == 1) ? 1 : $clog2(XYC);

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

logic [XCNT_BITS-1:0] curr_wrX_C = '0, curr_wrX_N;
logic [YCNT_BITS-1:0] curr_wrY_C = '0, curr_wrY_N;

// -- Ram
logic [1:0][WGT_EN_BITS-1:0] a_we; // Bank enables
logic [1:0][XYCNT_BITS-1:0] a_addr;
logic [1:0][W-1:0] a_data_in;

// -- Offsets
logic [XC-1:0][XYCNT_BITS-1:0] x_offsets;
for(genvar i = 0; i < XC; i++) begin
    assign x_offsets[i] = i*YC;
end

// -- IPC
logic done;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_WR
    if(rst) begin
        state_wr_C <= ST_WR_0;

        curr_wrX_C <= 0;
        curr_wrY_C <= 0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_wrX_C <= curr_wrX_N;
        curr_wrY_C <= curr_wrY_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_0:
            if ((curr_wrY_C == YC - 1) && (curr_wrX_C == XC - 1) && ivld) begin
                state_wr_N = (done || (state_rd_C == ST_RD_0)) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (done || (state_rd_C == ST_RD_0)) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if ((curr_wrY_C == YC - 1) && (curr_wrX_C == XC - 1) && ivld) begin
                state_wr_N = (done || (state_rd_C == ST_RD_1)) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (done || (state_rd_C == ST_RD_1)) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_wrX_N = curr_wrX_C;
    curr_wrY_N = curr_wrY_C;

    // Input
    irdy = 1'b0;

    // Buffer control
    a_we = '0;
    for(int i = 0; i < 2; i++) begin
        a_addr[i] = x_offsets[curr_wrX_C] + curr_wrY_C;
        a_data_in[i] = idat;
    end

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy = 1'b1;

            if(ivld) begin
                if(state_wr_C == ST_WR_0) a_we[0] = '1; else a_we[1] = '1;

                curr_wrY_N = (curr_wrY_C == YC-1) ? 0 : curr_wrY_C + 1;
                curr_wrX_N = (curr_wrY_C == YC-1) ? ((curr_wrX_C == XC-1) ? 0 : curr_wrX_C + 1) : curr_wrX_C;
            end
        end
    endcase

end


// ----------------------------------------------------------------------------
// Reader
// ----------------------------------------------------------------------------

// -- Regs
logic [XCNT_BITS-1:0] curr_rdX_C = '0, curr_rdX_N;
logic [YCNT_BITS-1:0] curr_rdY_C = '0, curr_rdY_N;

// -- Ram
logic [1:0] vld_s0_C = '0, vld_s0_N;
logic [1:0] vld_s1_C = '0, vld_s1_N;
logic vld_C = '0, vld_N;
logic [W-1:0] odat_C = '0, odat_N;

logic [1:0][XYCNT_BITS-1:0] b_addr;
logic [1:0][W-1:0] odat_ram;

// -- Cond
logic cond_go;

// -- Oreg
logic [W-1:0] odat_int;
logic ovld_int;
logic ordy_int;

// -- REG
always_ff @( posedge clk ) begin : REG_PROC_RD
    if(rst) begin
        state_rd_C <= ST_RD_0;

        curr_rdX_C <= 0;
        curr_rdY_C <= 0;

        vld_s0_C <= 0;
        vld_s1_C <= 0;
        vld_C <= 0;
        odat_C <= 0;
    end
    else begin
        state_rd_C <= state_rd_N;

        curr_rdX_C <= curr_rdX_N;
        curr_rdY_C <= curr_rdY_N;

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
            if(ordy_int && ((state_wr_C == ST_WR_0) ? cond_go : 1'b1)) begin
                if((curr_rdX_C == XC-1) && (curr_rdY_C == YC-1)) begin
                    state_rd_N = ST_RD_1;
                end
            end

        ST_RD_1:
            if(ordy_int && ((state_wr_C == ST_WR_1) ? cond_go : 1'b1)) begin
                if((curr_rdX_C == XC-1) && (curr_rdY_C == YC-1)) begin
                    state_rd_N = ST_RD_0;
                end
            end

    endcase
end

// -- DP cond
always_comb begin
    cond_go = 1'b0;

    if(curr_wrX_C > curr_rdX_C) begin
        cond_go = 1'b1;
    end
    else if(curr_wrX_C == curr_rdX_C) begin
        if(curr_wrY_C > curr_rdY_C) begin
            cond_go = 1'b1;
        end
    end
end

// -- DP
always_comb begin : DP_PROC_RD
    curr_rdX_N = curr_rdX_C;
    curr_rdY_N = curr_rdY_C;

    for(int i = 0; i < 2; i++) begin
        vld_s0_N[i] = ordy_int ? 1'b0 : vld_s0_C[i];
        vld_s1_N[i] = ordy_int ? vld_s0_C[i] : vld_s1_C[i];
    end

    vld_N = ordy_int ? |vld_s1_C : vld_C;
    odat_N = ordy_int ? (vld_s1_C[0] ? odat_ram[0] : odat_ram[1]) : odat_C;

    for(int i = 0; i < 2; i++) begin
        b_addr[i] = x_offsets[curr_rdX_C] + curr_rdY_C;
    end

    done = 1'b0;

    case(state_rd_C)
        ST_RD_0: begin
            if(ordy_int) begin
                if((state_wr_C == ST_WR_0) ? cond_go : 1'b1) begin
                    vld_s0_N[0] = 1'b1;

                    curr_rdX_N = (curr_rdX_C == XC-1) ? 0 : curr_rdX_C + 1;
                    curr_rdY_N = (curr_rdX_C == XC-1) ? ((curr_rdY_C == YC-1) ? 0 : curr_rdY_C + 1) : curr_rdY_C;
                    done = ((curr_rdY_C == YC-1) && (curr_rdX_C == XC-1));
                end
            end
        end

        ST_RD_1: begin
            if(ordy_int) begin
                if((state_wr_C == ST_WR_1) ? cond_go : 1'b1) begin
                    vld_s0_N[1] = 1'b1;

                    curr_rdX_N = (curr_rdX_C == XC-1) ? 0 : curr_rdX_C + 1;
                    curr_rdY_N = (curr_rdX_C == XC-1) ? ((curr_rdY_C == YC-1) ? 0 : curr_rdY_C + 1) : curr_rdY_C;
                    done = ((curr_rdY_C == YC-1) && (curr_rdX_C == XC-1));
                end
            end
        end

    endcase

end

assign ovld_int = vld_C;
assign odat_int = odat_C;

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

for(genvar i = 0; i < 2; i++) begin
    ram_p_c #(
        .ADDR_BITS(XYCNT_BITS),
        .DATA_BITS(RAM_BITS),
        .RAM_STYLE("distributed")
    ) inst_ram_tp_c (
        .clk(clk),
        .a_en(1'b1),
        .a_we(a_we[i]),
        .a_addr(a_addr[i]),
        .b_en(ordy_int),
        .b_addr(b_addr[i]),
        .a_data_in(a_data_in[i]),
        .a_data_out(),
        .b_data_out(odat_ram[i])
    );
end

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------

Q_srl #(
    .depth(2), .width(W)
) inst_out_fifo (
    .clock(clk),
    .reset(rst),
    .count(),
    .maxcount(),
    .i_d(odat_int),
    .i_v(ovld_int),
    .i_r(ordy_int),
    .o_d(odat),
    .o_v(ovld),
    .o_r(ordy)
);


endmodule

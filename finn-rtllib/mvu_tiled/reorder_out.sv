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
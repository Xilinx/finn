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

import iwTypes::*;

`include "axi_macros.svh"

module dma_rd_gather #(
    parameter int unsigned ADDR_BITS = HBM_ADDR_BITS,
    parameter int unsigned LEN_BITS = HBM_LEN_BITS,
    parameter int unsigned DATA_BITS = HBM_DATA_BITS,
    
    parameter int unsigned N_CH = 1,
    parameter logic[HBM_ADDR_BITS-1:0] BASE_ADDR,

    parameter integer Q_DEPTH = 8,
    parameter integer N_DCPL_STAGES = 1,
    parameter integer DBG = 0
)(
    input  logic                                        aclk,
    input  logic                                        aresetn,
    input  logic                                        dclk,
    input  logic                                        dresetn,

    AXI4S.slave                                         rd_ctrl,
    output logic [N_CH-1:0]                             rd_done,

    AXI4.master                                         m_axi_ddr [N_CH],

    AXI4S_PCKT.master                                   rd_axis_ddr
);

logic [N_CH-1:0][HBM_ADDR_BITS-1:0] CH_ADDRESSES;

for(genvar i = 0; i < N_CH; i++) begin
    assign CH_ADDRESSES[i] = BASE_ADDR + i*HBM_RNG;
end
 
//
// CTRL
//

AXI4S #(.AXI4S_DATA_BITS(CDMA_CTRL_BITS)) rd_ctrl_int();

axis_reg_array_rtl #(.DATA_BITS(CDMA_CTRL_BITS), .N_STAGES(1)) inst_int_reg (
    .aclk(aclk), .aresetn(aresetn), .s_axis(rd_ctrl), .m_axis(rd_ctrl_int)
);

AXI4S #(.AXI4S_DATA_BITS(CDMA_CTRL_BITS)) dma_ctrl_in [N_CH] ();
AXI4S #(.AXI4S_DATA_BITS(CDMA_CTRL_BITS)) dma_ctrl_out [N_CH] ();
logic [N_CH-1:0][CDMA_CTRL_BITS-1:0] dma_ctrl_out_data;
logic [N_CH-1:0] dma_done;

logic [N_CH-1:0] dma_ctrl_agg_ready;
for (genvar i = 0; i < N_CH; i++) begin
    assign dma_ctrl_agg_ready[i] = dma_ctrl_in[i].tready;
end

assign rd_ctrl_int.tready = &dma_ctrl_agg_ready;
for(genvar i = 0; i < N_CH; i++) begin
    assign dma_ctrl_in[i].tvalid = rd_ctrl_int.tvalid && rd_ctrl_int.tready;
    assign dma_ctrl_in[i].tdata[0+:ADDR_BITS] = rd_ctrl_int.tdata[0+:ADDR_BITS];
    assign dma_ctrl_in[i].tdata[ADDR_BITS+:LEN_BITS] = rd_ctrl_int.tdata[ADDR_BITS+:LEN_BITS];

    axis_fifo_hbm_ctrl_cc inst_fifo_hbm_ctrl (
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aclk(dclk),
        .s_axis_tvalid(dma_ctrl_in[i].tvalid),
        .s_axis_tready(dma_ctrl_in[i].tready),
        .s_axis_tdata (dma_ctrl_in[i].tdata),
        .m_axis_tvalid(dma_ctrl_out[i].tvalid),
        .m_axis_tready(dma_ctrl_out[i].tready),
        .m_axis_tdata (dma_ctrl_out[i].tdata)
    );

    assign dma_ctrl_out_data[i][0+:ADDR_BITS] = CH_ADDRESSES[i] | dma_ctrl_out[i].tdata[0+:ADDR_BITS];
    assign dma_ctrl_out_data[i][ADDR_BITS+:LEN_BITS] = dma_ctrl_out[i].tdata[ADDR_BITS+:LEN_BITS];

    axis_cc_done inst_cc_done (
        .s_axis_aclk(dclk),
        .s_axis_aresetn(dresetn),
        .m_axis_aclk(aclk),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(dma_done[i]),
        .s_axis_tready(),
        .m_axis_tvalid(rd_done[i]),
        .m_axis_tready(1'b1)
    );

end

//
// DMAs
// 

AXI4S_PCKT #(.AXI4S_DATA_BITS(DATA_BITS)) axis_tmp [N_CH] ();
AXI4S_PCKT #(.AXI4S_DATA_BITS(DATA_BITS)) axis_dma [N_CH] ();
AXI4S_PCKT #(.AXI4S_DATA_BITS(DATA_BITS)) axis_dma_int [N_CH] ();

for(genvar i = 0; i < N_CH; i++) begin
    cdma_top #(
        .ADDR_BITS(ADDR_BITS),
        .LEN_BITS(LEN_BITS),
        .DATA_BITS(DATA_BITS),
        .CDMA_RD(1),
        .CDMA_WR(0),
        .CDMA_TYPE(1)
    ) inst_dma_rd (
        .aclk(dclk),
        .aresetn(dresetn),

        .m_axi_ddr(m_axi_ddr[i]),

        .rd_valid(dma_ctrl_out[i].tvalid),
        .rd_ready(dma_ctrl_out[i].tready),
        .rd_paddr(dma_ctrl_out_data[i][0+:ADDR_BITS]),
        .rd_len(dma_ctrl_out_data[i][ADDR_BITS+:LEN_BITS]),
        .rd_done(dma_done[i]),

        .wr_valid(1'b0),
        .wr_ready(),
        .wr_paddr('0),
        .wr_len('0),
        .wr_done(),

        .s_axis_ddr(axis_tmp[i]),
        .m_axis_ddr(axis_dma[i])
    );
    `AXISP_TIE_OFF_M(axis_tmp[i])
end

//
// Reg
//

for(genvar i = 0; i < N_CH; i++) begin
    axisp_dma_reg inst_int_reg (
        .aclk(dclk), .aresetn(dresetn),
        .s_axis_tvalid(axis_dma[i].tvalid), .s_axis_tready(axis_dma[i].tready), .s_axis_tdata(axis_dma[i].tdata), .s_axis_tkeep(axis_dma[i].tkeep), .s_axis_tlast(axis_dma[i].tlast),
        .m_axis_tvalid(axis_dma_int[i].tvalid), .m_axis_tready(axis_dma_int[i].tready), .m_axis_tdata(axis_dma_int[i].tdata), .m_axis_tkeep(axis_dma_int[i].tkeep), .m_axis_tlast(axis_dma_int[i].tlast)
    );
end

//
// Clock crossing
//

AXI4S_PCKT #(.AXI4S_DATA_BITS(DATA_BITS)) axis_cc [N_CH] ();

for(genvar i = 0; i < N_CH; i++) begin
    axis_fifo_hbm_data_cc inst_fifo_hbm_data (
        .s_axis_aclk(dclk),
        .s_axis_aresetn(dresetn),
        .m_axis_aclk(aclk),
        .s_axis_tvalid(axis_dma_int[i].tvalid),
        .s_axis_tready(axis_dma_int[i].tready),
        .s_axis_tdata (axis_dma_int[i].tdata),
        .s_axis_tkeep (axis_dma_int[i].tkeep),
        .s_axis_tlast (axis_dma_int[i].tlast),
        .m_axis_tvalid(axis_cc[i].tvalid),
        .m_axis_tready(axis_cc[i].tready),
        .m_axis_tdata (axis_cc[i].tdata),
        .m_axis_tkeep (axis_cc[i].tkeep),
        .m_axis_tlast (axis_cc[i].tlast)
    );
end

//
// Agg
//

AXI4S_PCKT #(.AXI4S_DATA_BITS(N_CH*DATA_BITS)) axis_cat ();

if(N_CH > 1) begin
    logic [N_CH-1:0] dma_agg_valid;
    for (genvar i = 0; i < N_CH; i++) begin
            assign dma_agg_valid[i] = axis_cc[i].tvalid;
    end

    assign axis_cat.tvalid = &dma_agg_valid;
    assign axis_cat.tlast = axis_cc[0].tlast;

    for(genvar i = 0; i < N_CH; i++) begin
        assign axis_cc[i].tready = axis_cat.tvalid & axis_cat.tready;
        assign axis_cat.tdata[i*DATA_BITS+:DATA_BITS] = axis_cc[i].tdata;
        assign axis_cat.tkeep[i*(DATA_BITS/8)+:DATA_BITS/8] = axis_cc[i].tkeep;
    end
end
else begin
    assign axis_cat.tvalid = axis_cc[0].tvalid;
    assign axis_cc[0].tready = axis_cat.tready;
    assign axis_cat.tdata = axis_cc[0].tdata;
    assign axis_cat.tlast = axis_cc[0].tlast;
    assign axis_cat.tkeep = axis_cc[0].tkeep;
end

//
// Oreg
//

for(genvar i = 0; i < N_CH; i++) begin
    if(i == 0) begin
        axisp_dma_reg inst_oreg (
            .aclk(aclk), .aresetn(aresetn),
            .s_axis_tvalid(axis_cat.tvalid), .s_axis_tready(axis_cat.tready), .s_axis_tdata(axis_cat.tdata[i*DATA_BITS+:DATA_BITS]), .s_axis_tkeep(axis_cat.tkeep[(i*DATA_BITS)/8+:DATA_BITS/8]), .s_axis_tlast(axis_cat.tlast),
            .m_axis_tvalid(rd_axis_ddr.tvalid), .m_axis_tready(rd_axis_ddr.tready), .m_axis_tdata(rd_axis_ddr.tdata[i*DATA_BITS+:DATA_BITS]), .m_axis_tkeep(rd_axis_ddr.tkeep[(i*DATA_BITS)/8+:DATA_BITS/8]), .m_axis_tlast(rd_axis_ddr.tlast)
        );
    end else begin
        axisp_dma_reg inst_oreg (
            .aclk(aclk), .aresetn(aresetn),
            .s_axis_tvalid(axis_cat.tvalid), .s_axis_tready(), .s_axis_tdata(axis_cat.tdata[i*DATA_BITS+:DATA_BITS]), .s_axis_tkeep(axis_cat.tkeep[(i*DATA_BITS)/8+:DATA_BITS/8]), .s_axis_tlast(axis_cat.tlast),
            .m_axis_tvalid(), .m_axis_tready(rd_axis_ddr.tready), .m_axis_tdata(rd_axis_ddr.tdata[i*DATA_BITS+:DATA_BITS]), .m_axis_tkeep(rd_axis_ddr.tkeep[(i*DATA_BITS)/8+:DATA_BITS/8]), .m_axis_tlast()
        );
    end
end

endmodule

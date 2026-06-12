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

module store_unit #(
    parameter int unsigned              ADDR_BITS = HBM_ADDR_BITS,
    parameter int unsigned              DATA_BITS = HBM_DATA_BITS,
    parameter int unsigned              LEN_BITS = HBM_LEN_BITS,

    parameter int unsigned              OLEN_BITS,

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              CCROSS = 0
) (
    input  wire                         aclk,
    input  wire                         aresetn,
    input  wire                         dclk,
    input  wire                         dresetn,

    AXI4.master                         m_axi_hbm,

    output logic                        wr_done,
    AXI4S.slave                         wr_ctrl,
    
    AXI4S.slave                         s_axis
);

// Queue
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_ctrl_split ();
AXI4S #(.AXI4S_DATA_BITS(LEN_BITS)) q_last_in ();
AXI4S #(.AXI4S_DATA_BITS(LEN_BITS)) q_last_out ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma_in ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma_out ();

queue #(.QWIDTH(ADDR_BITS+LEN_BITS)) inst_queue_split (
    .aclk(aclk), .aresetn(aresetn), .s_axis(wr_ctrl), .m_axis(q_ctrl_split)
);

assign q_last_in.tvalid = q_ctrl_split.tvalid & q_dma_in.tready;
assign q_dma_in.tvalid  = q_ctrl_split.tvalid & q_last_in.tvalid;
assign q_ctrl_split.tready = q_last_in.tready & q_dma_in.tready;

assign q_last_in.tdata = q_ctrl_split.tdata[ADDR_BITS+:LEN_BITS];
assign q_dma_in.tdata  = q_ctrl_split.tdata;

queue #(.QWIDTH(LEN_BITS)) inst_queue_last (
    .aclk(aclk), .aresetn(aresetn), .s_axis(q_last_in), .m_axis(q_last_out)
);

logic dma_done;

logic dclk_int;
logic dresetn_int;

if(CCROSS == 1) begin
    axis_fifo_hbm_ctrl_cc inst_fifo_hbm_ctrl (
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aclk(dclk),
        .s_axis_tvalid(q_dma_in.tvalid),
        .s_axis_tready(q_dma_in.tready),
        .s_axis_tdata (q_dma_in.tdata),
        .m_axis_tvalid(q_dma_out.tvalid),
        .m_axis_tready(q_dma_out.tready),
        .m_axis_tdata (q_dma_out.tdata)
    );

    axis_cc_done inst_cc_done (
        .s_axis_aclk(dclk),
        .s_axis_aresetn(dresetn),
        .m_axis_aclk(aclk),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(dma_done),
        .s_axis_tready(),
        .m_axis_tvalid(wr_done),
        .m_axis_tready(1'b1)
    );

    assign dclk_int = dclk;
    assign dresetn_int = dresetn;
end
else begin
    queue #(.QWIDTH(ADDR_BITS+LEN_BITS)) inst_fifo_ctrl (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis(q_dma_in),
        .m_axis(q_dma_out)
    );

    assign wr_done = dma_done;

    assign dclk_int = aclk;
    assign dresetn_int = aresetn;
end

// DMA
AXI4S_PCKT axis_tmp ();
AXI4S_PCKT axis_dma ();

cdma_top #(
    .ADDR_BITS(ADDR_BITS),
    .LEN_BITS(LEN_BITS),
    .DATA_BITS(DATA_BITS),
    .CDMA_RD(0),
    .CDMA_WR(1)
) inst_dma_rd (
    .aclk(dclk_int),
    .aresetn(dresetn_int),

    .m_axi_ddr(m_axi_hbm),

    .rd_valid(1'b0),
    .rd_ready(),
    .rd_paddr('0),
    .rd_len('0),
    .rd_done(),

    .wr_valid(q_dma_out.tvalid),
    .wr_ready(q_dma_out.tready),
    .wr_paddr(q_dma_out.tdata[0+:ADDR_BITS]),
    .wr_len(q_dma_out.tdata[ADDR_BITS+:LEN_BITS]),
    .wr_done(dma_done),

    .s_axis_ddr(axis_dma),
    .m_axis_ddr(axis_tmp)
);
`AXISP_TIE_OFF_S(axis_tmp)

// REG stage 1
AXI4S_PCKT axis_dma_int ();

axisp_dma_reg inst_int_reg (
    .aclk(dclk_int), .aresetn(dresetn_int),
    .s_axis_tvalid(axis_dma_int.tvalid), .s_axis_tready(axis_dma_int.tready), .s_axis_tdata(axis_dma_int.tdata), .s_axis_tkeep(axis_dma_int.tkeep), .s_axis_tlast(axis_dma_int.tlast),
    .m_axis_tvalid(axis_dma.tvalid), .m_axis_tready(axis_dma.tready), .m_axis_tdata(axis_dma.tdata), .m_axis_tkeep(axis_dma.tkeep), .m_axis_tlast(axis_dma.tlast)
);

// CCROSS
AXI4S_PCKT axis_cc ();

if(CCROSS == 1) begin
    axis_fifo_hbm_data_cc inst_fifo_hbm_data (
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aclk(dclk_int),
        .s_axis_tvalid(axis_cc.tvalid),
        .s_axis_tready(axis_cc.tready),
        .s_axis_tdata (axis_cc.tdata),
        .s_axis_tkeep (axis_cc.tkeep),
        .s_axis_tlast (axis_cc.tlast),
        .m_axis_tvalid(axis_dma_int.tvalid),
        .m_axis_tready(axis_dma_int.tready),
        .m_axis_tdata (axis_dma_int.tdata),
        .m_axis_tkeep (axis_dma_int.tkeep),
        .m_axis_tlast (axis_dma_int.tlast)
    );
end else begin
    `AXISP_ASSIGN(axis_cc, axis_dma_int)
end

// DWC
AXI4S_PCKT #(.AXI4S_DATA_BITS(OLEN_BITS)) axis_dwc ();

axis_hdwc_wr inst_dwc_wr (
    .aclk(aclk), .aresetn(aresetn),
    .s_axis_tvalid(axis_dwc.tvalid), .s_axis_tready(axis_dwc.tready), .s_axis_tdata(axis_dwc.tdata), .s_axis_tkeep(axis_dwc.tkeep), .s_axis_tlast(axis_dwc.tlast),
    .m_axis_tvalid(axis_cc.tvalid), .m_axis_tready(axis_cc.tready), .m_axis_tdata(axis_cc.tdata), .m_axis_tkeep(axis_cc.tkeep), .m_axis_tlast(axis_cc.tlast)
);

// LAST
AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) axis_ldet ();

axis_ldet_wr inst_ldet_wr (.aclk(aclk), .aresetn(aresetn), .s_meta(q_last_out), .s_axis(axis_ldet), .m_axis(axis_dwc));

// REG
axis_reg_array_rtl #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(OLEN_BITS)) inst_reg_rd (.aclk(aclk), .aresetn(aresetn), .s_axis(s_axis), .m_axis(axis_ldet));

endmodule

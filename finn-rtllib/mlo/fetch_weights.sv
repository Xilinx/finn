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

module fetch_weights #(
    parameter int unsigned              PE,
    parameter int unsigned              SIMD,
    parameter int unsigned              MH,
    parameter int unsigned              MW,
    parameter int unsigned              N_REPS,
    parameter int unsigned              WEIGHT_WIDTH = 8,

    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              IDX_BITS = 16,

    parameter int unsigned              N_LAYERS,

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              EN_OREG = 1,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              DBG = 0,

    // Safely deducible parameters
    parameter                          DS_BITS_BA = (PE+7)/8 * 8,
	parameter                          WS_BITS_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
    parameter logic[ADDR_BITS-1:0]     LAYER_OFFS = ((MH*MW*WEIGHT_WIDTH+7)/8) & ~7 // 8-byte aligned
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    output logic                        m_done,

    // AXI
    output logic[ADDR_BITS-1:0]         m_axi_ddr_araddr,
    output logic[1:0]		            m_axi_ddr_arburst,
    output logic[3:0]		            m_axi_ddr_arcache,
    output logic[1:0]      		        m_axi_ddr_arid,
    output logic[7:0]		            m_axi_ddr_arlen,
    output logic[0:0]		            m_axi_ddr_arlock,
    output logic[2:0]		            m_axi_ddr_arprot,
    output logic[2:0]		            m_axi_ddr_arsize,
    input  logic			            m_axi_ddr_arready,
    output logic			            m_axi_ddr_arvalid,
    output logic[ADDR_BITS-1:0] 	    m_axi_ddr_awaddr,
    output logic[1:0]		            m_axi_ddr_awburst,
    output logic[3:0]		            m_axi_ddr_awcache,
    output logic[1:0]		            m_axi_ddr_awid,
    output logic[7:0]		            m_axi_ddr_awlen,
    output logic[0:0]		            m_axi_ddr_awlock,
    output logic[2:0]		            m_axi_ddr_awprot,
    output logic[2:0]		            m_axi_ddr_awsize,
    input  logic			            m_axi_ddr_awready,
    output logic			            m_axi_ddr_awvalid,
    input  logic[DATA_BITS-1:0] 	    m_axi_ddr_rdata,
    input  logic[1:0]      		        m_axi_ddr_rid,
    input  logic			            m_axi_ddr_rlast,
    input  logic[1:0]		            m_axi_ddr_rresp,
    output logic 			            m_axi_ddr_rready,
    input  logic			            m_axi_ddr_rvalid,
    output logic[DATA_BITS-1:0] 	    m_axi_ddr_wdata,
    output logic			            m_axi_ddr_wlast,
    output logic[DATA_BITS/8-1:0] 	    m_axi_ddr_wstrb,
    input  logic			            m_axi_ddr_wready,
    output logic			            m_axi_ddr_wvalid,
    input  logic[1:0]      		        m_axi_ddr_bid,
    input  logic[1:0]		            m_axi_ddr_bresp,
    output logic			            m_axi_ddr_bready,
    input  logic			            m_axi_ddr_bvalid,

    // Index
    input  logic                        s_idx_tvalid,
    output logic                        s_idx_tready,
    input  logic[IDX_BITS-1:0]          s_idx_tdata,

    // Stream
    // TODO: Should we reg this? Would be quite wide ...
    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready,
    output logic[WS_BITS_BA-1:0]        m_axis_tdata
);

// Offsets
logic [N_LAYERS-1:0][ADDR_BITS-1:0] l_offsets;
for(genvar i = 0; i < N_LAYERS; i++) begin
    assign l_offsets[i] = (i * LAYER_OFFS);
end

logic q_idx_out_tvalid, q_idx_out_tready;
logic [IDX_BITS-1:0] q_idx_out_tdata;
logic [ADDR_BITS-1:0] q_dma_addr;
logic [LEN_BITS-1:0] q_dma_len;

// Queues
Q_srl #(
    .depth(QDEPTH),
    .width(IDX_BITS)
) inst_queue_in (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(s_idx_tdata), .i_v(s_idx_tvalid), .i_r(s_idx_tready),
    .o_d(q_idx_out_tdata), .o_v(q_idx_out_tvalid), .o_r(q_idx_out_tready)
);

assign q_dma_addr = l_offsets[q_idx_out_tdata];
assign q_dma_len = ((MH*MW*WEIGHT_WIDTH+7)/8) & ~7;

// DMA
logic axis_dma_tvalid;
logic axis_dma_tready;
logic[DATA_BITS-1:0] axis_dma_tdata;
logic[DATA_BITS/8-1:0] axis_dma_tkeep;
logic axis_dma_tlast;

cdma_u_rd #(
    .DATA_BITS(DATA_BITS),
    .ADDR_BITS(ADDR_BITS),
    .LEN_BITS(LEN_BITS)
) inst_dma (
    .aclk(aclk), .aresetn(aresetn),

    .rd_valid(q_idx_out_tvalid), .rd_ready(q_idx_out_tready),
    .rd_paddr(q_dma_addr), .rd_len(q_dma_len),
    .rd_done(m_done),

    .m_axi_ddr_arvalid(m_axi_ddr_arvalid),
    .m_axi_ddr_arready(m_axi_ddr_arready),
    .m_axi_ddr_araddr(m_axi_ddr_araddr),
    .m_axi_ddr_arid(m_axi_ddr_arid),
    .m_axi_ddr_arlen(m_axi_ddr_arlen),
    .m_axi_ddr_arsize(m_axi_ddr_arsize),
    .m_axi_ddr_arburst(m_axi_ddr_arburst),
    .m_axi_ddr_arlock(m_axi_ddr_arlock),
    .m_axi_ddr_arcache(m_axi_ddr_arcache),
    .m_axi_ddr_arprot(m_axi_ddr_arprot),
    .m_axi_ddr_rvalid(m_axi_ddr_rvalid),
    .m_axi_ddr_rready(m_axi_ddr_rready),
    .m_axi_ddr_rdata(m_axi_ddr_rdata),
    .m_axi_ddr_rlast(m_axi_ddr_rlast),
    .m_axi_ddr_rid(m_axi_ddr_rid),
    .m_axi_ddr_rresp(m_axi_ddr_rresp),

    .m_axis_ddr_tvalid(axis_dma_tvalid),
    .m_axis_ddr_tready(axis_dma_tready),
    .m_axis_ddr_tdata(axis_dma_tdata),
    .m_axis_ddr_tkeep(axis_dma_tkeep),
    .m_axis_ddr_tlast(axis_dma_tlast)
);

// Width conversion
logic axis_dwc_tvalid;
logic axis_dwc_tready;
logic[DS_BITS_BA-1:0] axis_dwc_tdata;
logic[(DS_BITS_BA)/8-1:0] axis_dwc_tkeep;
logic axis_dwc_tlast;

axis_dwc #(
    .S_DATA_BITS(DATA_BITS), .M_DATA_BITS(DS_BITS_BA)
) inst_dwc (
    .aclk(aclk), .aresetn(aresetn),
    .s_axis_tvalid(axis_dma_tvalid), .s_axis_tready(axis_dma_tready), .s_axis_tdata(axis_dma_tdata), .s_axis_tkeep(axis_dma_tkeep), .s_axis_tlast(axis_dma_tlast),
    .m_axis_tvalid(axis_dwc_tvalid), .m_axis_tready(axis_dwc_tready), .m_axis_tdata(axis_dwc_tdata), .m_axis_tkeep(axis_dwc_tkeep), .m_axis_tlast(axis_dwc_tlast)
);

// Double buffer
logic axis_lwb_tvalid;
logic axis_lwb_tready;
logic[WS_BITS_BA-1:0] axis_lwb_tdata;

local_weight_buffer #(
    .PE(PE), .SIMD(SIMD), .MH(MH), .MW(MW), .N_REPS(N_REPS), .WEIGHT_WIDTH(WEIGHT_WIDTH), .DBG(DBG)
) inst_weight_buff (
    .clk(aclk), .rst(~aresetn),
    .ivld(axis_dwc_tvalid), .irdy(axis_dwc_tready), .idat(axis_dwc_tdata),
    .ovld(axis_lwb_tvalid), .ordy(axis_lwb_tready), .odat(axis_lwb_tdata)
);

// Reg slice
if(EN_OREG) begin
    axis_reg_array_rtl #(
        .DATA_BITS(WS_BITS_BA), .N_STAGES(N_DCPL_STGS)
    ) inst_oreg (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tvalid(axis_lwb_tvalid), .s_axis_tready(axis_lwb_tready), .s_axis_tdata(axis_lwb_tdata),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready), .m_axis_tdata(m_axis_tdata)
    );
end else begin
    assign m_axis_tvalid = axis_lwb_tvalid;
    assign axis_lwb_tready = m_axis_tready;
    assign m_axis_tdata = axis_lwb_tdata;
end

endmodule

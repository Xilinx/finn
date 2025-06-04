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

module cdma_top #(
    parameter integer                   BURST_LEN = 16,
    parameter integer                   DATA_BITS = HBM_DATA_BITS,
    parameter integer                   ADDR_BITS = HBM_ADDR_BITS,
    parameter integer                   ID_BITS = HBM_ID_BITS,
    parameter integer                   LEN_BITS = HBM_LEN_BITS,
    parameter integer                   CDMA_TYPE = 1, // (0: Aligned, 1: Unaglined)
    parameter integer                   CDMA_RD = 1,
    parameter integer                   CDMA_WR = 1
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    input  logic                        rd_valid,
    output logic                        rd_ready,
    input  logic [ADDR_BITS-1:0]        rd_paddr,
    input  logic [LEN_BITS-1:0]         rd_len,
    output logic                        rd_done,

    input  logic                        wr_valid,
    output logic                        wr_ready,
    input  logic [ADDR_BITS-1:0]        wr_paddr,
    input  logic [LEN_BITS-1:0]         wr_len,
    output logic                        wr_done,

    AXI4.master                         m_axi_ddr,

    AXI4S_PCKT.slave                    s_axis_ddr,
    AXI4S_PCKT.master                   m_axis_ddr
);

if(CDMA_TYPE == 0) begin

    // Aligned
    if(CDMA_RD == 1) begin
        cdma_a_rd #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_a_rd (
            .aclk(aclk),
            .aresetn(aresetn),

            .rd_valid(rd_valid),
            .rd_ready(rd_ready),
            .rd_paddr(rd_paddr),
            .rd_len(rd_len),
            .rd_done(rd_done),

            .m_axi_ddr_arvalid(m_axi_ddr.arvalid),
            .m_axi_ddr_arready(m_axi_ddr.arready),
            .m_axi_ddr_araddr(m_axi_ddr.araddr),
            .m_axi_ddr_arid(m_axi_ddr.arid),
            .m_axi_ddr_arlen(m_axi_ddr.arlen),
            .m_axi_ddr_arsize(m_axi_ddr.arsize),
            .m_axi_ddr_arburst(m_axi_ddr.arburst),
            .m_axi_ddr_arlock(m_axi_ddr.arlock),
            .m_axi_ddr_arcache(m_axi_ddr.arcache),
            .m_axi_ddr_arprot(m_axi_ddr.arprot),
            .m_axi_ddr_rvalid(m_axi_ddr.rvalid),
            .m_axi_ddr_rready(m_axi_ddr.rready),
            .m_axi_ddr_rdata(m_axi_ddr.rdata),
            .m_axi_ddr_rlast(m_axi_ddr.rlast),
            .m_axi_ddr_rid(m_axi_ddr.rid),
            .m_axi_ddr_rresp(m_axi_ddr.rresp),

            .m_axis_ddr(m_axis_ddr)
        );
    end
    else begin
        assign rd_done = 1'b0;

        assign m_axi_ddr.arvalid = 1'b0;
        assign m_axi_ddr.araddr = '0;
        assign m_axi_ddr.arid = '0;
        assign m_axi_ddr.arprot = '0;
        assign m_axi_ddr.arlen = '0;
        assign m_axi_ddr.arsize = '0;
        assign m_axi_ddr.arburst = '0;
        assign m_axi_ddr.arlock = '0;
        assign m_axi_ddr.arcache = '0;

        assign m_axi_ddr.rready = 1'b1;

        assign m_axis_ddr.tvalid = 1'b0;
        assign m_axis_ddr.tdata = '0;
        assign m_axis_ddr.tkeep = '0;
        assign m_axis_ddr.tlast = 1'b0;
    end

    if(CDMA_WR == 1) begin
        cdma_a_wr #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_a_wr (
            .aclk(aclk),
            .aresetn(aresetn),

            .wr_valid(wr_valid),
            .wr_ready(wr_ready),
            .wr_paddr(wr_paddr),
            .wr_len(wr_len),
            .wr_done(wr_done),

            .m_axi_ddr_awvalid(m_axi_ddr.awvalid),
            .m_axi_ddr_awready(m_axi_ddr.awready),
            .m_axi_ddr_awaddr(m_axi_ddr.awaddr),
            .m_axi_ddr_awid(m_axi_ddr.awid),
            .m_axi_ddr_awlen(m_axi_ddr.awlen),
            .m_axi_ddr_awsize(m_axi_ddr.awsize),
            .m_axi_ddr_awburst(m_axi_ddr.awburst),
            .m_axi_ddr_awlock(m_axi_ddr.awlock),
            .m_axi_ddr_awcache(m_axi_ddr.awcache),
            .m_axi_ddr_wdata(m_axi_ddr.wdata),
            .m_axi_ddr_wstrb(m_axi_ddr.wstrb),
            .m_axi_ddr_wlast(m_axi_ddr.wlast),
            .m_axi_ddr_wvalid(m_axi_ddr.wvalid),
            .m_axi_ddr_wready(m_axi_ddr.wready),
            .m_axi_ddr_bid(m_axi_ddr.bid),
            .m_axi_ddr_bresp(m_axi_ddr.bresp),
            .m_axi_ddr_bvalid(m_axi_ddr.bvalid),
            .m_axi_ddr_bready(m_axi_ddr.bready),

            .s_axis_ddr(s_axis_ddr)
        );
    end
    else begin
        assign wr_done = 1'b0;

        assign m_axi_ddr.awvalid = 1'b0;
        assign m_axi_ddr.awaddr = '0;
        assign m_axi_ddr.awid = '0;
        assign m_axi_ddr.awprot = '0;
        assign m_axi_ddr.awlen = '0;
        assign m_axi_ddr.awsize = '0;
        assign m_axi_ddr.awburst = '0;
        assign m_axi_ddr.awlock = '0;
        assign m_axi_ddr.awcache = '0;

        assign m_axi_ddr.wvalid = 1'b0;
        assign m_axi_ddr.wdata = '0;
        assign m_axi_ddr.wstrb = '0;
        assign m_axi_ddr.wlast = '0;

        assign m_axi_ddr.bready = 1'b1;

        assign s_axis_ddr.tready = 1'b1;
    end

end
else begin
    // Non-aligned
    if(CDMA_RD == 1) begin
        cdma_u_rd #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_u_rd (
            .aclk(aclk),
            .aresetn(aresetn),

            .rd_valid(rd_valid),
            .rd_ready(rd_ready),
            .rd_paddr(rd_paddr),
            .rd_len(rd_len),
            .rd_done(rd_done),

            .m_axi_ddr_arvalid(m_axi_ddr.arvalid),
            .m_axi_ddr_arready(m_axi_ddr.arready),
            .m_axi_ddr_araddr(m_axi_ddr.araddr),
            .m_axi_ddr_arid(m_axi_ddr.arid),
            .m_axi_ddr_arlen(m_axi_ddr.arlen),
            .m_axi_ddr_arsize(m_axi_ddr.arsize),
            .m_axi_ddr_arburst(m_axi_ddr.arburst),
            .m_axi_ddr_arlock(m_axi_ddr.arlock),
            .m_axi_ddr_arcache(m_axi_ddr.arcache),
            .m_axi_ddr_arprot(m_axi_ddr.arprot),
            .m_axi_ddr_rvalid(m_axi_ddr.rvalid),
            .m_axi_ddr_rready(m_axi_ddr.rready),
            .m_axi_ddr_rdata(m_axi_ddr.rdata),
            .m_axi_ddr_rlast(m_axi_ddr.rlast),
            .m_axi_ddr_rid(m_axi_ddr.rid),
            .m_axi_ddr_rresp(m_axi_ddr.rresp),

            .m_axis_ddr(m_axis_ddr)
        );
    end
    else begin
        assign rd_done = 1'b0;

        assign m_axi_ddr.arvalid = 1'b0;
        assign m_axi_ddr.araddr = '0;
        assign m_axi_ddr.arid = '0;
        assign m_axi_ddr.arprot = '0;
        assign m_axi_ddr.arlen = '0;
        assign m_axi_ddr.arsize = '0;
        assign m_axi_ddr.arburst = '0;
        assign m_axi_ddr.arlock = '0;
        assign m_axi_ddr.arcache = '0;

        assign m_axi_ddr.rready = 1'b1;

        assign m_axis_ddr.tvalid = 1'b0;
        assign m_axis_ddr.tdata = '0;
        assign m_axis_ddr.tkeep = '0;
        assign m_axis_ddr.tlast = 1'b0;
    end

    if(CDMA_WR == 1) begin
        cdma_u_wr #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_u_wr (
            .aclk(aclk),
            .aresetn(aresetn),

            .wr_valid(wr_valid),
            .wr_ready(wr_ready),
            .wr_paddr(wr_paddr),
            .wr_len(wr_len),
            .wr_done(wr_done),

            .m_axi_ddr_awvalid(m_axi_ddr.awvalid),
            .m_axi_ddr_awready(m_axi_ddr.awready),
            .m_axi_ddr_awaddr(m_axi_ddr.awaddr),
            .m_axi_ddr_awid(m_axi_ddr.awid),
            .m_axi_ddr_awlen(m_axi_ddr.awlen),
            .m_axi_ddr_awsize(m_axi_ddr.awsize),
            .m_axi_ddr_awburst(m_axi_ddr.awburst),
            .m_axi_ddr_awlock(m_axi_ddr.awlock),
            .m_axi_ddr_awcache(m_axi_ddr.awcache),
            .m_axi_ddr_wdata(m_axi_ddr.wdata),
            .m_axi_ddr_wstrb(m_axi_ddr.wstrb),
            .m_axi_ddr_wlast(m_axi_ddr.wlast),
            .m_axi_ddr_wvalid(m_axi_ddr.wvalid),
            .m_axi_ddr_wready(m_axi_ddr.wready),
            .m_axi_ddr_bid(m_axi_ddr.bid),
            .m_axi_ddr_bresp(m_axi_ddr.bresp),
            .m_axi_ddr_bvalid(m_axi_ddr.bvalid),
            .m_axi_ddr_bready(m_axi_ddr.bready),

            .s_axis_ddr(s_axis_ddr)
        );
    end
    else begin
        assign wr_done = 1'b0;

        assign m_axi_ddr.awvalid = 1'b0;
        assign m_axi_ddr.awaddr = '0;
        assign m_axi_ddr.awid = '0;
        assign m_axi_ddr.awprot = '0;
        assign m_axi_ddr.awlen = '0;
        assign m_axi_ddr.awsize = '0;
        assign m_axi_ddr.awburst = '0;
        assign m_axi_ddr.awlock = '0;
        assign m_axi_ddr.awcache = '0;

        assign m_axi_ddr.wvalid = 1'b0;
        assign m_axi_ddr.wdata = '0;
        assign m_axi_ddr.wstrb = '0;
        assign m_axi_ddr.wlast = '0;

        assign m_axi_ddr.bready = 1'b1;

        assign s_axis_ddr.tready = 1'b1;
    end
end


endmodule

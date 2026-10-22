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

module cdma_top #(
    int unsigned                        DATA_BITS,
    int unsigned                        ADDR_BITS,
    int unsigned                        LEN_BITS,
    int unsigned                        ID_BITS = 2,
    int unsigned                        BURST_LEN = 16,
    bit                                 PACED = 1,
    int unsigned                        N_OUTSTANDING = 2,
    int unsigned                        CDMA_TYPE = 1, // (0: Aligned, 1: Unaglined, 2: Datamover IP)
    int unsigned                        CDMA_RD = 1,
    int unsigned                        CDMA_WR = 1
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    input  logic                        rd_valid,
    output logic                        rd_ready,
    input  logic[ADDR_BITS-1:0]         rd_paddr,
    input  logic[LEN_BITS-1:0]          rd_len,
    output logic                        rd_done,

    input  logic                        wr_valid,
    output logic                        wr_ready,
    input  logic[ADDR_BITS-1:0]         wr_paddr,
    input  logic[LEN_BITS-1:0]          wr_len,
    output logic                        wr_done,

    AXI4.master                         m_axi_ddr,

    AXI4SF.slave                        s_axis_ddr,
    AXI4SF.master                       m_axis_ddr
);

case (CDMA_TYPE)
  0: begin
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
        assign m_axis_ddr.tuser = '0;
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

  1: begin
    // Non-aligned
    if(CDMA_RD == 1) begin
        axi_dma_rd_u #(
            .AXI_DATA_WIDTH(DATA_BITS),
            .AXI_ADDR_WIDTH(ADDR_BITS),
            .AXI_STRB_WIDTH(DATA_BITS/8),
            .AXI_MAX_BURST_LEN(BURST_LEN),
            .AXIS_DATA_WIDTH(DATA_BITS),
            .AXIS_KEEP_ENABLE(1),
            .AXIS_KEEP_WIDTH(DATA_BITS/8),
            .AXIS_LAST_ENABLE(1),
            .LEN_WIDTH(LEN_BITS),
            .AXI_ID_BITS(ID_BITS),
            .PACED(PACED),
            .N_OUTSTANDING(N_OUTSTANDING)
        ) inst_cdma_u_rd (
            .aclk(aclk),
            .aresetn(aresetn),

            .s_axis_read_desc_addr(rd_paddr),
            .s_axis_read_desc_len(rd_len),
            .s_axis_read_desc_valid(rd_valid),
            .s_axis_read_desc_ready(rd_ready),

            .m_axis_read_desc_status_valid(rd_done),

            .m_axis_read_data_tdata(m_axis_ddr.tdata),
            .m_axis_read_data_tkeep(m_axis_ddr.tkeep),
            .m_axis_read_data_tvalid(m_axis_ddr.tvalid),
            .m_axis_read_data_tready(m_axis_ddr.tready),
            .m_axis_read_data_tlast(m_axis_ddr.tlast),

            .m_axi_arvalid(m_axi_ddr.arvalid),
            .m_axi_arready(m_axi_ddr.arready),
            .m_axi_araddr(m_axi_ddr.araddr),
            .m_axi_arid(m_axi_ddr.arid),
            .m_axi_arlen(m_axi_ddr.arlen),
            .m_axi_arsize(m_axi_ddr.arsize),
            .m_axi_arburst(m_axi_ddr.arburst),
            .m_axi_arlock(m_axi_ddr.arlock),
            .m_axi_arcache(m_axi_ddr.arcache),
            .m_axi_arprot(m_axi_ddr.arprot),
            .m_axi_rvalid(m_axi_ddr.rvalid),
            .m_axi_rready(m_axi_ddr.rready),
            .m_axi_rdata(m_axi_ddr.rdata),
            .m_axi_rlast(m_axi_ddr.rlast),
            .m_axi_rid(m_axi_ddr.rid),
            .m_axi_rresp(m_axi_ddr.rresp)
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
        assign m_axis_ddr.tuser = '0;
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

            .s_axis_ddr_tvalid(s_axis_ddr.tvalid),
            .s_axis_ddr_tready(s_axis_ddr.tready),
            .s_axis_ddr_tdata(s_axis_ddr.tdata),
            .s_axis_ddr_tkeep(s_axis_ddr.tkeep),
            .s_axis_ddr_tlast(s_axis_ddr.tlast)
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

  2: begin
    // Datamover IP
    if(CDMA_RD == 1) begin
        cdma_x_rd #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_x_rd (
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
        assign m_axis_ddr.tuser = '0;
        assign m_axis_ddr.tlast = 1'b0;
    end

    if(CDMA_WR == 1) begin
        cdma_x_wr #(
            .DATA_BITS(DATA_BITS),
            .ADDR_BITS(ADDR_BITS),
            .ID_BITS(ID_BITS),
            .LEN_BITS(LEN_BITS),
            .BURST_LEN(BURST_LEN)
        ) inst_cdma_x_wr (
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

  default: begin
    $fatal(1, "Invalid CDMA_TYPE=%0d. Allowed values are 0, 1, or 2.", CDMA_TYPE);
  end
endcase

endmodule

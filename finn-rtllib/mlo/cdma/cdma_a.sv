/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

import iwTypes::*;

/**
 * @brief   Aligned CDMA top level
 *
 * The aligned CDMA top level. Contains read and write DMA engines. 
 * Outstanding queues at the input. Low resource overhead.
 *
 *  @param BURST_LEN    Maximum burst length size
 *  @param DATA_BITS    Size of the data bus (both AXI and stream)
 *  @param ADDR_BITS    Size of the address bits
 *  @param ID_BITS      Size of the ID bits
 */
module cdma_a #(
    parameter integer                   BURST_LEN = 16,
    parameter integer                   DATA_BITS = AXI_DATA_BITS,
    parameter integer                   ADDR_BITS = AXI_ADDR_BITS,
    parameter integer                   ID_BITS = AXI_ID_BITS,
    parameter integer                   LEN_BITS = 32,
    parameter integer                   BURST_OUTSTANDING = 64
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

localparam integer DCPL_DEPTH = 4;

// RD ------------------------------------------------------------------------------------------
logic [LEN_BITS-1:0] rd_len_int;
logic [ADDR_BITS-1:0] rd_paddr_int;
logic rd_valid_int, rd_ready_int;
logic rd_done_int;

Q_srl #(
    .depth(DCPL_DEPTH), 
    .width(ADDR_BITS+LEN_BITS)
) inst_q_rd (
    .clock(aclk),
    .reset(!aresetn),
    .count(),
    .maxcount(),
    .i_d({rd_len, rd_paddr}),
    .i_v(rd_valid),
    .i_r(rd_ready),
    .o_d({rd_len_int, rd_paddr_int}),
    .o_v(rd_valid_int),
    .o_r(rd_ready_int)
);

always_ff @(posedge aclk) begin
    if(~aresetn)
        rd_done <= 1'b0;
    else
        rd_done <= rd_done_int;
end

// WR ------------------------------------------------------------------------------------------
logic [LEN_BITS-1:0] wr_len_int;
logic [ADDR_BITS-1:0] wr_paddr_int;
logic wr_valid_int, wr_ready_int;
logic wr_done_int;

Q_srl #(
    .depth(DCPL_DEPTH), 
    .width(ADDR_BITS+LEN_BITS)
) inst_q_wr (
    .clock(aclk),
    .reset(!aresetn),
    .count(),
    .maxcount(),
    .i_d({wr_len, wr_paddr}),
    .i_v(wr_valid),
    .i_r(wr_ready),
    .o_d({wr_len_int, wr_paddr_int}),
    .o_v(wr_valid_int),
    .o_r(wr_ready_int)
);

always_ff @(posedge aclk) begin
    if(~aresetn)
        wr_done <= 1'b0;
    else
        wr_done <= wr_done_int;
end

// 
// CDMA
//

// RD channel
axi_dma_rd_a #(
    .BURST_LEN(BURST_LEN),
    .DATA_BITS(DATA_BITS),
    .ADDR_BITS(ADDR_BITS),
    .ID_BITS(ID_BITS),
    .MAX_OUTSTANDING(BURST_OUTSTANDING)
) axi_dma_rd_inst (
    .aclk(aclk),
    .aresetn(aresetn),

    // CS
    .ctrl_valid(rd_valid_int),
    .stat_ready(rd_ready_int),
    .ctrl_addr(rd_paddr_int),
    .ctrl_len(rd_len_int),
    .ctrl_ctl(1'b1),
    .stat_done(rd_done_int),

    // AXI
    .arvalid(m_axi_ddr.arvalid),
    .arready(m_axi_ddr.arready),
    .araddr(m_axi_ddr.araddr),
    .arid(m_axi_ddr.arid),
    .arlen(m_axi_ddr.arlen),
    .arsize(m_axi_ddr.arsize),
    .arburst(m_axi_ddr.arburst),
    .arlock(m_axi_ddr.arlock),
    .arcache(m_axi_ddr.arcache),
    .arprot(m_axi_ddr.arprot),
    .rvalid(m_axi_ddr.rvalid),
    .rready(m_axi_ddr.rready),
    .rdata(m_axi_ddr.rdata),
    .rlast(m_axi_ddr.rlast),
    .rid(m_axi_ddr.rid),
    .rresp(m_axi_ddr.rresp),

    // AXIS
    .axis_out_tdata(m_axis_ddr.tdata),
    .axis_out_tkeep(m_axis_ddr.tkeep),
    .axis_out_tvalid(m_axis_ddr.tvalid),
    .axis_out_tready(m_axis_ddr.tready),
    .axis_out_tlast(m_axis_ddr.tlast)
);
assign m_axis_ddr.tuser = '0;

// WR channel
axi_dma_wr_a #(
    .BURST_LEN(BURST_LEN),
    .DATA_BITS(DATA_BITS),
    .ADDR_BITS(ADDR_BITS),
    .ID_BITS(ID_BITS),
    .MAX_OUTSTANDING(BURST_OUTSTANDING)
) axi_dma_wr_inst (
    .aclk(aclk),
    .aresetn(aresetn),

    // CS
    .ctrl_valid(wr_valid_int),
    .stat_ready(wr_ready_int),
    .ctrl_addr(wr_paddr_int),
    .ctrl_len(wr_len_int),
    .ctrl_ctl(1'b1),
    .stat_done(wr_done_int),

    // AXI
    .awvalid(m_axi_ddr.awvalid),
    .awready(m_axi_ddr.awready),
    .awaddr(m_axi_ddr.awaddr),
    .awid(m_axi_ddr.awid),
    .awlen(m_axi_ddr.awlen),
    .awsize(m_axi_ddr.awsize),
    .awburst(m_axi_ddr.awburst),
    .awlock(m_axi_ddr.awlock),
    .awcache(m_axi_ddr.awcache),
    .wdata(m_axi_ddr.wdata),
    .wstrb(m_axi_ddr.wstrb),
    .wlast(m_axi_ddr.wlast),
    .wvalid(m_axi_ddr.wvalid),
    .wready(m_axi_ddr.wready),
    .bid(m_axi_ddr.bid),
    .bresp(m_axi_ddr.bresp),
    .bvalid(m_axi_ddr.bvalid),
    .bready(m_axi_ddr.bready),

    // AXIS
    .axis_in_tdata(s_axis_ddr.tdata),
    .axis_in_tkeep(s_axis_ddr.tkeep),
    .axis_in_tvalid(s_axis_ddr.tvalid),
    .axis_in_tready(s_axis_ddr.tready),
    .axis_in_tlast(s_axis_ddr.tlast)
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_A

`endif

endmodule
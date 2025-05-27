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
module cdma_u_wr #(
    parameter integer                   BURST_LEN = 16,
    parameter integer                   DATA_BITS = HBM_DATA_BITS,
    parameter integer                   ADDR_BITS = HBM_ADDR_BITS,
    parameter integer                   LEN_BITS = HBM_LEN_BITS,
    parameter integer                   ID_BITS = HBM_ID_BITS,
    parameter integer                   BURST_OUTSTANDING = 64
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    // CS
    input  logic                        wr_valid,
    output logic                        wr_ready,
    input  logic[ADDR_BITS-1:0]         wr_paddr,
    input  logic[LEN_BITS-1:0]          wr_len,
    output logic                        wr_done,

    // AXI4 master interface
    output wire                         m_axi_ddr_awvalid,
    input  wire                         m_axi_ddr_awready,
    output wire [ADDR_BITS-1:0]         m_axi_ddr_awaddr,
    output wire [ID_BITS-1:0]           m_axi_ddr_awid,
    output wire [7:0]                   m_axi_ddr_awlen,
    output wire [2:0]                   m_axi_ddr_awsize,
    output wire [1:0]                   m_axi_ddr_awburst,
    output wire [0:0]                   m_axi_ddr_awlock,
    output wire [3:0]                   m_axi_ddr_awcache,
    output wire [DATA_BITS-1:0]         m_axi_ddr_wdata,
    output wire [DATA_BITS/8-1:0]       m_axi_ddr_wstrb,
    output wire                         m_axi_ddr_wlast,
    output wire                         m_axi_ddr_wvalid,
    input  wire                         m_axi_ddr_wready,
    input  wire [ID_BITS-1:0]           m_axi_ddr_bid,
    input  wire [1:0]                   m_axi_ddr_bresp,
    input  wire                         m_axi_ddr_bvalid,
    output wire                         m_axi_ddr_bready,

    // AXI4S
    AXI4S_PCKT.slave                    s_axis_ddr
);

localparam integer DCPL_DEPTH = 4;

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

// WR channel
axi_dma_wr_u #(
    .AXI_DATA_WIDTH(DATA_BITS),
    .AXI_ADDR_WIDTH(ADDR_BITS),
    .AXI_STRB_WIDTH(DATA_BITS/8),
    .AXI_MAX_BURST_LEN(BURST_LEN),
    .AXIS_DATA_WIDTH(DATA_BITS),
    .AXIS_KEEP_ENABLE(1),
    .AXIS_KEEP_WIDTH(DATA_BITS/8),
    .AXIS_LAST_ENABLE(0),
    .LEN_WIDTH(LEN_BITS),
    .AXI_ID_BITS(ID_BITS)
)
axi_dma_wr_inst (
    .aclk(aclk),
    .aresetn(aresetn),

    /*
     * AXI write descriptor input
     */
    .s_axis_write_desc_addr(wr_paddr_int),
    .s_axis_write_desc_len(wr_len_int),
    .s_axis_write_desc_valid(wr_valid_int),
    .s_axis_write_desc_ready(wr_ready_int),

    /*
     * AXI write descriptor status output
     */
    .m_axis_write_desc_status_valid(wr_done_int),

    /*
     * AXI stream write data input
     */
    .s_axis_write_data_tdata(s_axis_ddr.tdata),
    .s_axis_write_data_tkeep(s_axis_ddr.tkeep),
    .s_axis_write_data_tvalid(s_axis_ddr.tvalid),
    .s_axis_write_data_tready(s_axis_ddr.tready),
    .s_axis_write_data_tlast(s_axis_ddr.tlast),

    /*
     * AXI master interface
     */
    .m_axi_awid(m_axi_ddr_awid),
    .m_axi_awaddr(m_axi_ddr_awaddr),
    .m_axi_awlen(m_axi_ddr_awlen),
    .m_axi_awsize(m_axi_ddr_awsize),
    .m_axi_awburst(m_axi_ddr_awburst),
    .m_axi_awlock(m_axi_ddr_awlock),
    .m_axi_awcache(m_axi_ddr_awcache),
    .m_axi_awvalid(m_axi_ddr_awvalid),
    .m_axi_awready(m_axi_ddr_awready),
    .m_axi_wdata(m_axi_ddr_wdata),
    .m_axi_wstrb(m_axi_ddr_wstrb),
    .m_axi_wlast(m_axi_ddr_wlast),
    .m_axi_wvalid(m_axi_ddr_wvalid),
    .m_axi_wready(m_axi_ddr_wready),
    .m_axi_bid(m_axi_ddr_bid),
    .m_axi_bresp(m_axi_ddr_bresp),
    .m_axi_bvalid(m_axi_ddr_bvalid),
    .m_axi_bready(m_axi_ddr_bready)
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_A_WR

`endif

endmodule

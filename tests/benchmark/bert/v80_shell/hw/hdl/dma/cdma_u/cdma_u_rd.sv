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
module cdma_u_rd #(
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
    input  logic                        rd_valid,
    output logic                        rd_ready,
    input  logic[ADDR_BITS-1:0]         rd_paddr,
    input  logic[LEN_BITS-1:0]          rd_len,
    output logic                        rd_done,

    // AXI4 master interface
    output wire                         m_axi_ddr_arvalid,
    input  wire                         m_axi_ddr_arready,
    output wire [ADDR_BITS-1:0]         m_axi_ddr_araddr,
    output wire [ID_BITS-1:0]           m_axi_ddr_arid,
    output wire [7:0]                   m_axi_ddr_arlen,
    output wire [2:0]                   m_axi_ddr_arsize,
    output wire [1:0]                   m_axi_ddr_arburst,
    output wire [0:0]                   m_axi_ddr_arlock,
    output wire [3:0]                   m_axi_ddr_arcache,
    output wire [2:0]                   m_axi_ddr_arprot,
    input  wire                         m_axi_ddr_rvalid,
    output wire                         m_axi_ddr_rready,
    input  wire [DATA_BITS-1:0]         m_axi_ddr_rdata,
    input  wire                         m_axi_ddr_rlast,
    input  wire [ID_BITS-1:0]           m_axi_ddr_rid,
    input  wire [1:0]                   m_axi_ddr_rresp,

    // AXI4S
    AXI4S_PCKT.master                   m_axis_ddr
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

//
// CDMA
//

// RD channel
axi_dma_rd_u #(
    .AXI_DATA_WIDTH(DATA_BITS),
    .AXI_ADDR_WIDTH(ADDR_BITS),
    .AXI_STRB_WIDTH(DATA_BITS/8),
    .AXI_MAX_BURST_LEN(BURST_LEN),
    .AXIS_DATA_WIDTH(DATA_BITS),
    .AXIS_KEEP_ENABLE(1),
    .AXIS_KEEP_WIDTH(DATA_BITS/8),
    .AXIS_LAST_ENABLE(1'b1),
    .LEN_WIDTH(LEN_BITS),
    .AXI_ID_BITS(ID_BITS)
)
axi_dma_rd_inst (
    .aclk(aclk),
    .aresetn(aresetn),

    /*
     * AXI read descriptor input
     */
    .s_axis_read_desc_addr(rd_paddr_int),
    .s_axis_read_desc_len(rd_len_int),
    .s_axis_read_desc_valid(rd_valid_int),
    .s_axis_read_desc_ready(rd_ready_int),

    /*
     * AXI read descriptor status output
     */
    .m_axis_read_desc_status_valid(rd_done_int),

    /*
     * AXI stream read data output
     */
    .m_axis_read_data_tdata(m_axis_ddr.tdata),
    .m_axis_read_data_tkeep(m_axis_ddr.tkeep),
    .m_axis_read_data_tvalid(m_axis_ddr.tvalid),
    .m_axis_read_data_tready(m_axis_ddr.tready),
    .m_axis_read_data_tlast(m_axis_ddr.tlast),

    /*
     * AXI master interface
     */
    .m_axi_arid(m_axi_ddr_arid),
    .m_axi_araddr(m_axi_ddr_araddr),
    .m_axi_arlen(m_axi_ddr_arlen),
    .m_axi_arsize(m_axi_ddr_arsize),
    .m_axi_arburst(m_axi_ddr_arburst),
    .m_axi_arlock(m_axi_ddr_arlock),
    .m_axi_arcache(m_axi_ddr_arcache),
    .m_axi_arprot(m_axi_ddr_arprot),
    .m_axi_arvalid(m_axi_ddr_arvalid),
    .m_axi_arready(m_axi_ddr_arready),
    .m_axi_rid(m_axi_ddr_rid),
    .m_axi_rdata(m_axi_ddr_rdata),
    .m_axi_rresp(m_axi_ddr_rresp),
    .m_axi_rlast(m_axi_ddr_rlast),
    .m_axi_rvalid(m_axi_ddr_rvalid),
    .m_axi_rready(m_axi_ddr_rready)
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_U_RD

`endif

endmodule

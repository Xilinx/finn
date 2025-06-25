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

/**
 * @brief   Unaligned CDMA top level
 *
 * The unaligned CDMA top level. Contains read and write DMA engines.
 * Outstanding queues at the input. High resource overhead.
 *
 *  @param BURST_LEN    Maximum burst length size
 *  @param DATA_BITS    Size of the data bus (both AXI and stream)
 *  @param ADDR_BITS    Size of the address bits
 *  @param ID_BITS      Size of the ID bits
 */
module cdma_x_rd #(
    parameter integer                   BURST_LEN = 64,
    parameter integer                   DATA_BITS = 256,
    parameter integer                   ADDR_BITS = 64,
    parameter integer                   ID_BITS = 2,
    parameter integer                   LEN_BITS = 32
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    input  logic                        rd_valid,
    output logic                        rd_ready,
    input  logic[ADDR_BITS-1:0]         rd_paddr,
    input  logic[LEN_BITS-1:0]          rd_len,
    output logic                        rd_done,

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

    AXI4SF.m                            m_axis_ddr
);

logic mm2s_error;
logic [7:0] rd_sts;

logic [103:0] rd_req;

// RD
//assign rd_req = {8'h0, 24'h0, rd_paddr, 1'b1, 1'b1, 6'h0, 1'b1, rd_len[22:0]};
assign rd_req = {8'h0, rd_paddr, 1'b1, 1'b1, 6'h0, 1'b1, rd_len[22:0]};

cdma_datamover_rd inst_cdma_datamover (
    // RD clk
    .m_axi_mm2s_aclk(aclk),// : IN STD_LOGIC;
    .m_axi_mm2s_aresetn(aresetn), //: IN STD_LOGIC;
    .m_axis_mm2s_cmdsts_aclk(aclk), //: IN STD_LOGIC;
    .m_axis_mm2s_cmdsts_aresetn(aresetn), //: IN STD_LOGIC;
    .mm2s_err(mm2s_error), //: OUT STD_LOGIC;
    // RD cmd
    .s_axis_mm2s_cmd_tvalid(rd_valid), //: IN STD_LOGIC;
    .s_axis_mm2s_cmd_tready(rd_ready), //: OUT STD_LOGIC;
    .s_axis_mm2s_cmd_tdata(rd_req), //: IN STD_LOGIC_VECTOR(103 DOWNTO 0);
    // RD sts
    .m_axis_mm2s_sts_tvalid(rd_done), //: OUT STD_LOGIC;
    .m_axis_mm2s_sts_tready(1'b1), //: IN STD_LOGIC;
    .m_axis_mm2s_sts_tdata(rd_sts), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axis_mm2s_sts_tkeep(), //: OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    .m_axis_mm2s_sts_tlast(), //: OUT STD_LOGIC;
    // RD channel AXI
    .m_axi_mm2s_arid(m_axi_ddr.arid), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_araddr(m_axi_ddr.araddr), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axi_mm2s_arlen(m_axi_ddr.arlen), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axi_mm2s_arsize(m_axi_ddr.arsize), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_mm2s_arburst(m_axi_ddr.arburst), //: OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_mm2s_arprot(m_axi_ddr.arprot), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_mm2s_arcache(m_axi_ddr.arcache), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_aruser(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_arvalid(m_axi_ddr.arvalid), //: OUT STD_LOGIC;
    .m_axi_mm2s_arready(m_axi_ddr.arready), //: IN STD_LOGIC;
    .m_axi_mm2s_rdata(m_axi_ddr.rdata), //: IN STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axi_mm2s_rresp(m_axi_ddr.rresp), //: IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_mm2s_rlast(m_axi_ddr.rlast), //: IN STD_LOGIC;
    .m_axi_mm2s_rvalid(m_axi_ddr.rvalid), //: IN STD_LOGIC;
    .m_axi_mm2s_rready(m_axi_ddr.rready), //: OUT STD_LOGIC;
    // RD channel AXIS
    .m_axis_mm2s_tdata(m_axis_ddr.tdata), //: OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axis_mm2s_tkeep(m_axis_ddr.tkeep), //: OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    .m_axis_mm2s_tlast(m_axis_ddr.tlast), //: OUT STD_LOGIC;
    .m_axis_mm2s_tvalid(m_axis_ddr.tvalid), //: OUT STD_LOGIC;
    .m_axis_mm2s_tready(m_axis_ddr.tready) //: IN STD_LOGIC;
);
assign m_axis_ddr.tuser = '0;

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_X_RD

`endif

endmodule

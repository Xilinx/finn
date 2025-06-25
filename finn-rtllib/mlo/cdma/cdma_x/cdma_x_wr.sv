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
module cdma_x_wr #(
    parameter integer                   BURST_LEN = 64,
    parameter integer                   DATA_BITS = 256,
    parameter integer                   ADDR_BITS = 64,
    parameter integer                   ID_BITS = 2,
    parameter integer                   LEN_BITS = 32
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    input  logic                        wr_valid,
    output logic                        wr_ready,
    input  logic[ADDR_BITS-1:0]         wr_paddr,
    input  logic[LEN_BITS-1:0]          wr_len,
    output logic                        wr_done,

    AXI4.m                              m_axi_ddr,

    AXI4SF.s                            s_axis_ddr
);

logic s2mm_error;
logic [7:0] wr_sts;

logic [103:0] wr_req;

// WR
//assign wr_req = {8'h0, 24'h0, wr_paddr, 1'b1, 1'b1, 6'h0, 1'b1, wr_len[22:0]};
assign wr_req = {8'h0, wr_paddr, 1'b1, 1'b1, 6'h0, 1'b1, wr_len[22:0]};

cdma_datamover_wr inst_cdma_datamover (
    // WR clk
    .m_axi_s2mm_aclk(aclk), //: IN STD_LOGIC;
    .m_axi_s2mm_aresetn(aresetn), //: IN STD_LOGIC;
    .m_axis_s2mm_cmdsts_awclk(aclk), //: IN STD_LOGIC;
    .m_axis_s2mm_cmdsts_aresetn(aresetn), //: IN STD_LOGIC;
    .s2mm_err(s2mm_error), //: OUT STD_LOGIC;
    // WR cmd
    .s_axis_s2mm_cmd_tvalid(wr_valid), //: IN STD_LOGIC;
    .s_axis_s2mm_cmd_tready(wr_ready), //: OUT STD_LOGIC;
    .s_axis_s2mm_cmd_tdata(wr_req), //: IN STD_LOGIC_VECTOR(103 DOWNTO 0);
    // WR sts
    .m_axis_s2mm_sts_tvalid(wr_done), //: OUT STD_LOGIC;
    .m_axis_s2mm_sts_tready(1'b1), //: IN STD_LOGIC;
    .m_axis_s2mm_sts_tdata(wr_sts), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axis_s2mm_sts_tkeep(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axis_s2mm_sts_tlast(), //: OUT STD_LOGIC;
    // WR channel AXI
    .m_axi_s2mm_awid(m_axi_ddr.awid), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awaddr(m_axi_ddr.awaddr), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axi_s2mm_awlen(m_axi_ddr.awlen), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axi_s2mm_awsize(m_axi_ddr.awsize), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awburst(m_axi_ddr.awburst), //: OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_awprot(m_axi_ddr.awprot), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awcache(m_axi_ddr.awcache), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awuser(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awvalid(m_axi_ddr.awvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_awready(m_axi_ddr.awready), //: IN STD_LOGIC;
    .m_axi_s2mm_wdata(m_axi_ddr.wdata), //: OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axi_s2mm_wstrb(m_axi_ddr.wstrb), //: OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    .m_axi_s2mm_wlast(m_axi_ddr.wlast), //: OUT STD_LOGIC;
    .m_axi_s2mm_wvalid(m_axi_ddr.wvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_wready(m_axi_ddr.wready), //: IN STD_LOGIC;
    .m_axi_s2mm_bresp(m_axi_ddr.bresp), //: IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_bvalid(m_axi_ddr.bvalid), //: IN STD_LOGIC;
    .m_axi_s2mm_bready(m_axi_ddr.bready), //: OUT STD_LOGIC;
    // WR channel AXIS
    .s_axis_s2mm_tdata(s_axis_ddr.tdata), //: IN STD_LOGIC_VECTOR(511 DOWNTO 0);
    .s_axis_s2mm_tkeep(s_axis_ddr.tkeep), //: IN STD_LOGIC_VECTOR(63 DOWNTO 0);
    .s_axis_s2mm_tlast(s_axis_ddr.tlast), //: IN STD_LOGIC;
    .s_axis_s2mm_tvalid(s_axis_ddr.tvalid), //: IN STD_LOGIC;
    .s_axis_s2mm_tready(s_axis_ddr.tready) //: OUT STD_LOGIC;
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_X_WR

`endif

endmodule

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
    parameter integer                   DATA_BITS = HBM_DATA_BITS,
    parameter integer                   ADDR_BITS = HBM_ADDR_BITS,
    parameter integer                   LEN_BITS = HBM_LEN_BITS,
    parameter integer                   ID_BITS = HBM_ID_BITS
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    input  logic                        wr_valid,
    output logic                        wr_ready,
    input  logic[ADDR_BITS-1:0]         wr_paddr,
    input  logic[LEN_BITS-1:0]          wr_len,
    output logic                        wr_done,

    output wire                         m_axi_ddr_awvalid,
    input  wire                         m_axi_ddr_awready,
    output wire [ADDR_BITS-1:0]         m_axi_ddr_awaddr,
    output wire [ID_BITS-1:0]           m_axi_ddr_awid,
    output wire [7:0]                   m_axi_ddr_awlen,
    output wire [2:0]                   m_axi_ddr_awsize,
    output wire [1:0]                   m_axi_ddr_awburst,
    output wire [0:0]                   m_axi_ddr_awlock,
    output wire [3:0]                   m_axi_ddr_awcache,
    output wire [2:0]                   m_axi_ddr_awprot,
    output wire [DATA_BITS-1:0]         m_axi_ddr_wdata,
    output wire [DATA_BITS/8-1:0]       m_axi_ddr_wstrb,
    output wire                         m_axi_ddr_wlast,
    output wire                         m_axi_ddr_wvalid,
    input  wire                         m_axi_ddr_wready,
    input  wire [ID_BITS-1:0]           m_axi_ddr_bid,
    input  wire [1:0]                   m_axi_ddr_bresp,
    input  wire                         m_axi_ddr_bvalid,
    output wire                         m_axi_ddr_bready,

    input  wire                         m_axi_ddr_rvalid,
    output wire                         m_axi_ddr_rready,
    input  wire [DATA_BITS-1:0]         m_axi_ddr_rdata,
    input  wire                         m_axi_ddr_rlast,
    input  wire [ID_BITS-1:0]           m_axi_ddr_rid,
    input  wire [1:0]                   m_axi_ddr_rresp,

    AXI4S_PCKT.slave                    s_axis_ddr
);

localparam integer MAX_DMA_TRANSFER = 32'd4194304;

typedef enum logic[0:0] {ST_IDLE, ST_SEND} state_t;
state_t state_C = ST_IDLE, state_N;

logic [ADDR_BITS-1:0] addr_C = '0, addr_N;
logic [LEN_BITS-1:0] len_C = '0, len_N;

logic wr_valid_int, wr_ready_int;
logic [ADDR_BITS-1:0] wr_paddr_int;
logic [LEN_BITS-1:0] wr_len_int;

logic dma_valid, dma_ready;
logic eof;

Q_srl #(
    .depth(8),
    .width(LEN_BITS+ADDR_BITS)
) isnt_dma_wr_fifo (
    .clock(aclk),
    .reset(~aresetn),
    .i_d({wr_len, wr_paddr}),
    .i_v(wr_valid),
    .i_r(wr_ready),
    .o_d({wr_len_int, wr_paddr_int}),
    .o_v(wr_valid_int),
    .o_r(wr_ready_int)
);

// REG
always_ff @(posedge aclk) begin
    if(~aresetn) begin
        state_C <= ST_IDLE;

        addr_C <= 'X;
        len_C <= 'X;
    end 
    else begin
        state_C <= state_N;

        addr_C <= addr_N;
        len_C <= len_N;
    end
end

// NSL
always_comb begin
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = wr_valid_int ? ST_SEND : ST_IDLE;

        ST_SEND:
            state_N = ((len_C <= MAX_DMA_TRANSFER) && (dma_valid && dma_ready)) ? ST_IDLE : ST_SEND; 

    endcase
end

// DP
always_comb begin
    addr_N = addr_C;
    len_N = len_C;
    
    wr_ready_int = 1'b0;
    dma_valid = 1'b0;

    eof = (len_C <= MAX_DMA_TRANSFER);

    case (state_C)
        ST_IDLE: begin
            wr_ready_int = 1'b1;
            if(wr_valid_int) begin
                addr_N = wr_paddr_int;
                len_N = wr_len_int;
            end
        end

        ST_SEND: begin
            dma_valid = 1'b1;
            if(dma_valid && dma_ready) begin
                addr_N = addr_C + MAX_DMA_TRANSFER;
                len_N = len_C - MAX_DMA_TRANSFER;
            end
        end
        
    endcase

end

//
// DMA
//

logic s2mm_error;
logic [7:0] wr_sts;

logic [103:0] wr_req;

// WR
//assign wr_req = {8'h0, 24'h0, wr_paddr, 1'b1, 1'b1, 6'h0, 1'b1, wr_len[22:0]};
//assign wr_req = {8'h0, wr_paddr, 1'b1, 1'b1, 6'h0, 1'b1, wr_len[22:0]};
assign wr_req = {8'h0, addr_C, 1'b1, eof, 6'h0, 1'b1, ((len_C <= MAX_DMA_TRANSFER) ? len_C[22:0] : MAX_DMA_TRANSFER)};

cdma_datamover_wr inst_cdma_datamover (
    // WR clk
    .m_axi_s2mm_aclk(aclk), //: IN STD_LOGIC;
    .m_axi_s2mm_aresetn(aresetn), //: IN STD_LOGIC;
    .m_axis_s2mm_cmdsts_awclk(aclk), //: IN STD_LOGIC;
    .m_axis_s2mm_cmdsts_aresetn(aresetn), //: IN STD_LOGIC;
    .s2mm_err(s2mm_error), //: OUT STD_LOGIC;
    // WR cmd
    .s_axis_s2mm_cmd_tvalid(dma_valid), //: IN STD_LOGIC;
    .s_axis_s2mm_cmd_tready(dma_ready), //: OUT STD_LOGIC;
    .s_axis_s2mm_cmd_tdata(wr_req), //: IN STD_LOGIC_VECTOR(103 DOWNTO 0);
    // WR sts
    .m_axis_s2mm_sts_tvalid(wr_done), //: OUT STD_LOGIC;
    .m_axis_s2mm_sts_tready(1'b1), //: IN STD_LOGIC;
    .m_axis_s2mm_sts_tdata(wr_sts), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axis_s2mm_sts_tkeep(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axis_s2mm_sts_tlast(), //: OUT STD_LOGIC;
    // WR channel AXI
    .m_axi_s2mm_awid(m_axi_ddr_awid), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awaddr(m_axi_ddr_awaddr), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axi_s2mm_awlen(m_axi_ddr_awlen), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axi_s2mm_awsize(m_axi_ddr_awsize), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awburst(m_axi_ddr_awburst), //: OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_awprot(m_axi_ddr_awprot), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awcache(m_axi_ddr_awcache), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awuser(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awvalid(m_axi_ddr_awvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_awready(m_axi_ddr_awready), //: IN STD_LOGIC;
    .m_axi_s2mm_wdata(m_axi_ddr_wdata), //: OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axi_s2mm_wstrb(m_axi_ddr_wstrb), //: OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    .m_axi_s2mm_wlast(m_axi_ddr_wlast), //: OUT STD_LOGIC;
    .m_axi_s2mm_wvalid(m_axi_ddr_wvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_wready(m_axi_ddr_wready), //: IN STD_LOGIC;
    .m_axi_s2mm_bresp(m_axi_ddr_bresp), //: IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_bvalid(m_axi_ddr_bvalid), //: IN STD_LOGIC;
    .m_axi_s2mm_bready(m_axi_ddr_bready), //: OUT STD_LOGIC;
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
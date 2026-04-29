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

module fetch_weights #(
    int unsigned              PE,
    int unsigned              SIMD,
    int unsigned              MH,
    int unsigned              MW,
    int unsigned              N_REPS,
    int unsigned              WEIGHT_WIDTH = 8,

    int unsigned              ADDR_BITS = 64,
    int unsigned              DATA_BITS = 256,
    int unsigned              LEN_BITS = 32,
    int unsigned              IDX_BITS = 16,

    int unsigned              N_LAYERS,

    int unsigned              QDEPTH = 8,
    int unsigned              EN_OREG = 1,
    int unsigned              N_DCPL_STGS = 1,
    int unsigned              DBG = 0,

    // Safely deducible parameters
    int unsigned              DS_BITS_BA = (SIMD*WEIGHT_WIDTH+7)/8 * 8,
	int unsigned              WS_BITS_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
    logic[ADDR_BITS-1:0]      LAYER_OFFS = ((MH*MW*WEIGHT_WIDTH+7)/8) & ~7 // 8-byte aligned
) (
    input  logic                        aclk,
    input  logic                        aresetn,

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

localparam int unsigned WMAT_SIZE = ((MH*MW*WEIGHT_WIDTH+7)/8) & ~7;

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
assign q_dma_len = WMAT_SIZE;

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

axis_fifo_adapter #(
    .S_DATA_WIDTH(DATA_BITS), .M_DATA_WIDTH(DS_BITS_BA)
) inst_dwc (
    .clk(aclk), .rst(~aresetn),
    .pause_req('0), .s_axis_tid('0), .s_axis_tdest('0), .s_axis_tuser('0),
    .s_axis_tvalid(axis_dma_tvalid), .s_axis_tready(axis_dma_tready), .s_axis_tdata(axis_dma_tdata), .s_axis_tkeep(axis_dma_tkeep), .s_axis_tlast(axis_dma_tlast),
    .pause_ack(), .m_axis_tid(), .m_axis_tdest(), .m_axis_tuser(),
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
    skid #(
        .DATA_WIDTH(WS_BITS_BA), .FEED_STAGES(N_DCPL_STGS)
    ) inst_oreg (
        .clk(aclk), .rst(~aresetn),
        .ivld(axis_lwb_tvalid), .irdy(axis_lwb_tready), .idat(axis_lwb_tdata),
        .ovld(m_axis_tvalid), .ordy(m_axis_tready), .odat(m_axis_tdata)
    );
end else begin
    assign m_axis_tvalid = axis_lwb_tvalid;
    assign axis_lwb_tready = m_axis_tready;
    assign m_axis_tdata = axis_lwb_tdata;
end

endmodule

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

module loop_control #(
    // COMPILER SET, this is the size of the global in, global out frames
    int unsigned FM_SIZE,
    // COMPILER SET, number of layers
    int unsigned N_LAYERS,
    // COMPILER SET? Input and output core bus widths
    int unsigned ILEN_BITS,
    int unsigned OLEN_BITS,

    // These can be as is
    int unsigned IDX_BITS,
    int unsigned ADDR_BITS,
    int unsigned DATA_BITS,
    int unsigned LEN_BITS
) (
    input  logic                aclk,
    input  logic                aresetn,

    // AXI4 master interface for m_axi_hbm
    output [ADDR_BITS-1:0]      m_axi_hbm_araddr,
    output [1:0]                m_axi_hbm_arburst,
    output [3:0]                m_axi_hbm_arcache,
    output [1:0]                m_axi_hbm_arid,
    output [7:0]                m_axi_hbm_arlen,
    output                      m_axi_hbm_arlock,
    output [2:0]                m_axi_hbm_arprot,
    output [2:0]                m_axi_hbm_arsize,
    input                       m_axi_hbm_arready,
    output                      m_axi_hbm_arvalid,
    output [ADDR_BITS-1:0]      m_axi_hbm_awaddr,
    output [1:0]                m_axi_hbm_awburst,
    output [3:0]                m_axi_hbm_awcache,
    output [1:0]                m_axi_hbm_awid,
    output [7:0]                m_axi_hbm_awlen,
    output                      m_axi_hbm_awlock,
    output [2:0]                m_axi_hbm_awprot,
    output [2:0]                m_axi_hbm_awsize,
    input                       m_axi_hbm_awready,
    output                      m_axi_hbm_awvalid,
    input  [DATA_BITS-1:0]      m_axi_hbm_rdata,
    input  [1:0]                m_axi_hbm_rid,
    input                       m_axi_hbm_rlast,
    input  [1:0]                m_axi_hbm_rresp,
    output                      m_axi_hbm_rready,
    input                       m_axi_hbm_rvalid,
    output [DATA_BITS-1:0]      m_axi_hbm_wdata,
    output                      m_axi_hbm_wlast,
    output [DATA_BITS/8-1:0]    m_axi_hbm_wstrb,
    input                       m_axi_hbm_wready,
    output                      m_axi_hbm_wvalid,
    input  [1:0]                m_axi_hbm_bid,
    input  [1:0]                m_axi_hbm_bresp,
    output                      m_axi_hbm_bready,
    input                       m_axi_hbm_bvalid,

    // AXI4S master interface for core_in
    output [ILEN_BITS-1:0]      m_axis_core_tdata,
    output                      m_axis_core_tvalid,
    input                       m_axis_core_tready,

    // AXI4S slave interface for core_out
    input  [OLEN_BITS-1:0]      s_axis_core_tdata,
    input                       s_axis_core_tvalid,
    output                      s_axis_core_tready,

    // AXI4S master interface for core_in_fw_idx
    output [IDX_BITS-1:0]       m_idx_tdata,
    output                      m_idx_tvalid,
    input                       m_idx_tready,

    // AXI4S slave interface for core_out_fw_idx
    input  [IDX_BITS-1:0]       s_idx_tdata,
    input                       s_idx_tvalid,
    output                      s_idx_tready,

    // Activation signals
    input  [ILEN_BITS-1:0]      s_axis_fs_tdata,
    input                       s_axis_fs_tvalid,
    output                      s_axis_fs_tready,

    output [OLEN_BITS-1:0]      m_axis_se_tdata,
    output                      m_axis_se_tvalid,
    input                       m_axis_se_tready
);

logic idx_if_in_tvalid, idx_if_in_tready;
logic [IDX_BITS-1:0] idx_if_in_tdata;
logic idx_if_out_tvalid, idx_if_out_tready;
logic [IDX_BITS-1:0] idx_if_out_tdata;

logic axis_if_in_tvalid, axis_if_in_tready;
logic [OLEN_BITS-1:0] axis_if_in_tdata;
logic axis_if_out_tvalid, axis_if_out_tready;
logic [ILEN_BITS-1:0] axis_if_out_tdata;

// ================-----------------------------------------------------------------
// Mux (input to the core)
// ================-----------------------------------------------------------------

mux #(
    .IDX_BITS(IDX_BITS),
    .FM_SIZE(FM_SIZE),
    .ILEN_BITS(ILEN_BITS)
) inst_mux_in (
    .aclk(aclk),
    .aresetn(aresetn),

    // Idx
    .s_idx_tvalid(idx_if_out_tvalid),
    .s_idx_tready(idx_if_out_tready),
    .s_idx_tdata (idx_if_out_tdata),

    .m_idx_tvalid(m_idx_tvalid),
    .m_idx_tready(m_idx_tready),
    .m_idx_tdata (m_idx_tdata),

    // Data
    .s_axis_fs_tvalid(s_axis_fs_tvalid),
    .s_axis_fs_tready(s_axis_fs_tready),
    .s_axis_fs_tdata (s_axis_fs_tdata),

    .s_axis_if_tvalid(axis_if_out_tvalid),
    .s_axis_if_tready(axis_if_out_tready),
    .s_axis_if_tdata (axis_if_out_tdata),

    .m_axis_tvalid(m_axis_core_tvalid),
    .m_axis_tready(m_axis_core_tready),
    .m_axis_tdata (m_axis_core_tdata)
);

// ================-----------------------------------------------------------------
// Demux (output of the core)
// ================-----------------------------------------------------------------

demux #(
    .N_LAYERS(N_LAYERS),
    .IDX_BITS(IDX_BITS),
    .FM_SIZE(FM_SIZE),
    .OLEN_BITS(OLEN_BITS)
) inst_mux_out (
    .aclk(aclk),
    .aresetn(aresetn),

    // Idx
    .s_idx_tvalid(s_idx_tvalid),
    .s_idx_tready(s_idx_tready),
    .s_idx_tdata (s_idx_tdata),

    .m_idx_tvalid(idx_if_in_tvalid),
    .m_idx_tready(idx_if_in_tready),
    .m_idx_tdata (idx_if_in_tdata),

    // Data
    .s_axis_tvalid(s_axis_core_tvalid),
    .s_axis_tready(s_axis_core_tready),
    .s_axis_tdata (s_axis_core_tdata),

    .m_axis_if_tvalid(axis_if_in_tvalid),
    .m_axis_if_tready(axis_if_in_tready),
    .m_axis_if_tdata (axis_if_in_tdata),

    .m_axis_se_tvalid(m_axis_se_tvalid),
    .m_axis_se_tready(m_axis_se_tready),
    .m_axis_se_tdata (m_axis_se_tdata)
);

//  ================-----------------------------------------------------------------
//  Intermediate frames
//  ================-----------------------------------------------------------------

intermediate_frames #(
    .FM_SIZE(FM_SIZE),
    .ILEN_BITS(ILEN_BITS),
    .OLEN_BITS(OLEN_BITS),
    .ADDR_BITS(ADDR_BITS),
    .DATA_BITS(DATA_BITS),
    .LEN_BITS(LEN_BITS),
    .IDX_BITS(IDX_BITS)
) inst_intermediate_frames (
    .aclk(aclk),
    .aresetn(aresetn),

    // MM
    .m_axi_ddr_arvalid(m_axi_hbm_arvalid),
    .m_axi_ddr_arready(m_axi_hbm_arready),
    .m_axi_ddr_araddr(m_axi_hbm_araddr),
    .m_axi_ddr_arid(m_axi_hbm_arid),
    .m_axi_ddr_arlen(m_axi_hbm_arlen),
    .m_axi_ddr_arsize(m_axi_hbm_arsize),
    .m_axi_ddr_arburst(m_axi_hbm_arburst),
    .m_axi_ddr_arlock(m_axi_hbm_arlock),
    .m_axi_ddr_arcache(m_axi_hbm_arcache),
    .m_axi_ddr_arprot(m_axi_hbm_arprot),
    .m_axi_ddr_rvalid(m_axi_hbm_rvalid),
    .m_axi_ddr_rready(m_axi_hbm_rready),
    .m_axi_ddr_rdata(m_axi_hbm_rdata),
    .m_axi_ddr_rlast(m_axi_hbm_rlast),
    .m_axi_ddr_rid(m_axi_hbm_rid),
    .m_axi_ddr_rresp(m_axi_hbm_rresp),
    .m_axi_ddr_awvalid(m_axi_hbm_awvalid),
    .m_axi_ddr_awready(m_axi_hbm_awready),
    .m_axi_ddr_awaddr(m_axi_hbm_awaddr),
    .m_axi_ddr_awid(m_axi_hbm_awid),
    .m_axi_ddr_awlen(m_axi_hbm_awlen),
    .m_axi_ddr_awsize(m_axi_hbm_awsize),
    .m_axi_ddr_awburst(m_axi_hbm_awburst),
    .m_axi_ddr_awlock(m_axi_hbm_awlock),
    .m_axi_ddr_awcache(m_axi_hbm_awcache),
    .m_axi_ddr_wdata(m_axi_hbm_wdata),
    .m_axi_ddr_wstrb(m_axi_hbm_wstrb),
    .m_axi_ddr_wlast(m_axi_hbm_wlast),
    .m_axi_ddr_wvalid(m_axi_hbm_wvalid),
    .m_axi_ddr_wready(m_axi_hbm_wready),
    .m_axi_ddr_bid(m_axi_hbm_bid),
    .m_axi_ddr_bresp(m_axi_hbm_bresp),
    .m_axi_ddr_bvalid(m_axi_hbm_bvalid),
    .m_axi_ddr_bready(m_axi_hbm_bready),

    // Idx
    .s_idx_tvalid(idx_if_in_tvalid),
    .s_idx_tready(idx_if_in_tready),
    .s_idx_tdata (idx_if_in_tdata),

    .m_idx_tvalid(idx_if_out_tvalid),
    .m_idx_tready(idx_if_out_tready),
    .m_idx_tdata (idx_if_out_tdata),

    // Data
    .s_axis_tvalid(axis_if_in_tvalid),
    .s_axis_tready(axis_if_in_tready),
    .s_axis_tdata (axis_if_in_tdata),

    .m_axis_tvalid(axis_if_out_tvalid),
    .m_axis_tready(axis_if_out_tready),
    .m_axis_tdata (axis_if_out_tdata)
);

endmodule

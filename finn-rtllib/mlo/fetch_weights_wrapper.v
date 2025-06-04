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
 * @brief	Verilog AXI-lite wrapper for MVU & VVU.
 *****************************************************************************/

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	MW = $MW$,
	parameter	MH = $MH$,
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
    parameter   N_REPS = $N_REPS$,
	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,

    parameter   ADDR_BITS = 64,
    parameter   DATA_BITS = 256,
    parameter   LEN_BITS = 32,
    parameter   CNT_BITS = 16,

    parameter   ADDR_WEIGHTS = $ADDR_WEIGHTS$,
    parameter   LAYER_OFFS = $LAYER_OFFS$,
    parameter   N_MAX_LAYERS = $N_MAX_LAYERS$,

	// Safely deducible parameters
	parameter	WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in1_V:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

    // Completion
    output wire                                out_done,

    // AXI
    output wire[ADDR_BITS-1:0]                 axi_mm_araddr,
    output wire[1:0]		                    axi_mm_arburst,
    output wire[3:0]		                    axi_mm_arcache,
    output wire[1:0]      		                axi_mm_arid,
    output wire[7:0]		                    axi_mm_arlen,
    output wire[0:0]		                    axi_mm_arlock,
    output wire[2:0]		                    axi_mm_arprot,
    output wire[2:0]		                    axi_mm_arsize,
    input  wire			                    axi_mm_arready,
    output wire			                    axi_mm_arvalid,
    output wire[ADDR_BITS-1:0] 	            axi_mm_awaddr,
    output wire[1:0]		                    axi_mm_awburst,
    output wire[3:0]		                    axi_mm_awcache,
    output wire[1:0]		                    axi_mm_awid,
    output wire[7:0]		                    axi_mm_awlen,
    output wire[0:0]		                    axi_mm_awlock,
    output wire[2:0]		                    axi_mm_awprot,
    output wire[2:0]		                    axi_mm_awsize,
    input  wire			                    axi_mm_awready,
    output wire			                    axi_mm_awvalid,
    input  wire[DATA_BITS-1:0] 	            axi_mm_rdata,
    input  wire[1:0]      		                axi_mm_rid,
    input  wire			                    axi_mm_rlast,
    input  wire[1:0]		                    axi_mm_rresp,
    output wire 			                    axi_mm_rready,
    input  wire			                    axi_mm_rvalid,
    output wire[DATA_BITS-1:0] 	            axi_mm_wdata,
    output wire			                    axi_mm_wlast,
    output wire[DATA_BITS/8-1:0] 	            axi_mm_wstrb,
    input  wire			                    axi_mm_wready,
    output wire			                    axi_mm_wvalid,
    input  wire[1:0]      		                axi_mm_bid,
    input  wire[1:0]		                    axi_mm_bresp,
    output wire			                    axi_mm_bready,
    input  wire			                    axi_mm_bvalid,

    // Index
    input  wire                                in_idx0_V_tvalid,
    output wire                                in_idx0_V_tready,
    input  wire[2*CNT_BITS-1:0]                in_idx0_V_tdata,

    // Stream
    // TODO: Should we reg this? Would be quite wide ...
    output wire                                out0_V_TVALID,
    input  wire                                out0_V_TREADY,
    output wire[WEIGHT_STREAM_WIDTH_BA-1:0]    out0_V_TDATA
);


fetch_weights #(
    .PE(PE), .SIMD(SIMD), .MH(MH), .MW(MW), .N_REPS(N_REPS),
    .WEIGHT_WIDTH(WEIGHT_WIDTH),
    .ADDR_BITS(ADDR_BITS), .DATA_BITS(DATA_BITS), .LEN_BITS(LEN_BITS), .CNT_BITS(CNT_BITS),
    .ADDR_WEIGHTS(ADDR_WEIGHTS), .LAYER_OFFS(LAYER_OFFS), .N_MAX_LAYERS(N_MAX_LAYERS)
) inst (
    .aclk(ap_clk), .aresetn(ap_rst_n),
    .m_done(out_done),

    .m_axi_hbm_araddr(axi_mm_araddr),
    .m_axi_hbm_arburst(axi_mm_arburst),
    .m_axi_hbm_arcache(axi_mm_arcache),
    .m_axi_hbm_arid(axi_mm_arid),
    .m_axi_hbm_arlen(axi_mm_arlen),
    .m_axi_hbm_arlock(axi_mm_arlock),
    .m_axi_hbm_arprot(axi_mm_arprot),
    .m_axi_hbm_arsize(axi_mm_arsize),
    .m_axi_hbm_arready(axi_mm_arready),
    .m_axi_hbm_arvalid(axi_mm_arvalid),
    .m_axi_hbm_awaddr(axi_mm_awaddr),
    .m_axi_hbm_awburst(axi_mm_awburst),
    .m_axi_hbm_awcache(axi_mm_awcache),
    .m_axi_hbm_awid(axi_mm_awid),
    .m_axi_hbm_awlen(axi_mm_awlen),
    .m_axi_hbm_awlock(axi_mm_awlock),
    .m_axi_hbm_awprot(axi_mm_awprot),
    .m_axi_hbm_awsize(axi_mm_awsize),
    .m_axi_hbm_awready(axi_mm_awready),
    .m_axi_hbm_awvalid(axi_mm_awvalid),
    .m_axi_hbm_rdata(axi_mm_rdata),
    .m_axi_hbm_rid(axi_mm_rid),
    .m_axi_hbm_rlast(axi_mm_rlast),
    .m_axi_hbm_rresp(axi_mm_rresp),
    .m_axi_hbm_rready(axi_mm_rready),
    .m_axi_hbm_rvalid(axi_mm_rvalid),
    .m_axi_hbm_wdata(axi_mm_wdata),
    .m_axi_hbm_wlast(axi_mm_wlast),
    .m_axi_hbm_wstrb(axi_mm_wstrb),
    .m_axi_hbm_wready(axi_mm_wready),
    .m_axi_hbm_wvalid(axi_mm_wvalid),
    .m_axi_hbm_bid(axi_mm_bid),
    .m_axi_hbm_bresp(axi_mm_bresp),
    .m_axi_hbm_bready(axi_mm_bready),
    .m_axi_hbm_bvalid(axi_mm_bvalid),

    .s_idx_tvalid(in_idx0_V_TVALID),
    .s_idx_tready(in_idx0_V_TREADY),
    .s_idx_tdata (in_idx0_V_TDATA),

    .s_idx_tvalid(out0_V_TVALID),
    .s_idx_tready(out0_V_TREADY),
    .s_idx_tdata (out0_V_TDATA)
);

endmodule // $MODULE_NAME_AXI_WRAPPER$

/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	Feature map padding.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module fmpadding_axi #(
	int unsigned  XCOUNTER_BITS,
	int unsigned  YCOUNTER_BITS,
	int unsigned  NUM_CHANNELS,
	int unsigned  SIMD,
	int unsigned  ELEM_BITS,
	int unsigned  INIT_XON,
	int unsigned  INIT_XOFF,
	int unsigned  INIT_XEND,
	int unsigned  INIT_YON,
	int unsigned  INIT_YOFF,
	int unsigned  INIT_YEND,

	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	       s_axilite_AWVALID,
	output	       s_axilite_AWREADY,
	input	[4:0]  s_axilite_AWADDR,

	input	        s_axilite_WVALID,
	output	        s_axilite_WREADY,
	input	[31:0]  s_axilite_WDATA,
	input	[ 3:0]  s_axilite_WSTRB,

	output	       s_axilite_BVALID,
	input	       s_axilite_BREADY,
	output	[1:0]  s_axilite_BRESP,

	// Reading
	input	       s_axilite_ARVALID,
	output	       s_axilite_ARREADY,
	input	[4:0]  s_axilite_ARADDR,

	output	        s_axilite_RVALID,
	input	        s_axilite_RREADY,
	output	[31:0]  s_axilite_RDATA,
	output	[ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_tdata
);

	// AXI-Lite Adapter
	uwire         we;
	uwire [ 4:0]  wa;
	uwire [31:0]  wd;
	axi2we #(.ADDR_BITS(5)) axilight_adapter (
		.ap_clk, .ap_rst_n,

		.s_axilite_AWVALID, .s_axilite_AWREADY, .s_axilite_AWADDR,
		.s_axilite_WVALID, .s_axilite_WREADY, .s_axilite_WDATA, .s_axilite_WSTRB,
		.s_axilite_BVALID, .s_axilite_BREADY, .s_axilite_BRESP,

		.s_axilite_ARVALID, .s_axilite_ARREADY, .s_axilite_ARADDR,
		.s_axilite_RVALID, .s_axilite_RREADY, .s_axilite_RDATA, .s_axilite_RRESP,

		.we, .wa, .wd
	);

	// Actual Padding
	fmpadding #(
		.XCOUNTER_BITS(XCOUNTER_BITS), .YCOUNTER_BITS(YCOUNTER_BITS),
		.NUM_CHANNELS(NUM_CHANNELS), .SIMD(SIMD),
		.INIT_XON(INIT_XON), .INIT_XOFF(INIT_XOFF), .INIT_XEND(INIT_XEND),
		.INIT_YON(INIT_YON), .INIT_YOFF(INIT_YOFF), .INIT_YEND(INIT_YEND),
		.ELEM_BITS(ELEM_BITS)
	) padding (
		.ap_clk, .ap_rst_n,

		.we, .wa, .wd,

		.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
		.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
	);

endmodule : fmpadding_axi

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
 * @brief	IPI-compatible Verilog wrapper for thresholding_axi module.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module thresholding_axi_wrapper #(
	parameter  N,	// output precision
	parameter  M,	// input/threshold precision
	parameter  C,	// Channels
	parameter  BIAS = 0,  // offsetting the output [0, 2^N-1) -> [-BIAS, 2^N-1 - BIAS)

	localparam  C_BITS = $clog2(C),
	localparam  O_BITS = BIAS <= 0?
		/* unsigned */ $clog2(2**N-BIAS) :
		/* signed */ 1+$clog2(BIAS >= 2**(N-1)? BIAS : 2**N-BIAS)
)(
	//- Global Control ------------------
	input	ap_clk,
	input	ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	                s_axilite_AWVALID,
	output	                s_axilite_AWREADY,
	input	[C_BITS+N-1:0]  s_axilite_AWADDR,

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
	input	[0:0]  s_axilite_ARADDR,

	output	        s_axilite_RVALID,
	input	        s_axilite_RREADY,
	output	[31:0]  s_axilite_RDATA,
	output	[ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	s_axis_tready,
	input	s_axis_tvalid,
	input	[((M+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	m_axis_tready,
	output	m_axis_tvalid,
	output	[((O_BITS+7)/8)*8-1:0]  m_axis_tdata
);

	thresholding_axi #(.N(N), .M(M), .C(C), .BIAS(BIAS)) inst (
		//- Global Control ------------------
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		//- AXI Lite ------------------------
		// Writing
		.s_axilite_AWVALID(s_axilite_AWVALID),
		.s_axilite_AWREADY(s_axilite_AWREADY),
		.s_axilite_AWADDR(s_axilite_AWADDR),

		.s_axilite_WVALID(s_axilite_WVALID),
		.s_axilite_WREADY(s_axilite_WREADY),
		.s_axilite_WDATA(s_axilite_WDATA),
		.s_axilite_WSTRB(s_axilite_WSTRB),

		.s_axilite_BVALID(s_axilite_BVALID),
		.s_axilite_BREADY(s_axilite_BREADY),
		.s_axilite_BRESP(s_axilite_BRESP),

		// Reading
		.s_axilite_ARVALID(s_axilite_ARVALID),
		.s_axilite_ARREADY(s_axilite_ARREADY),
		.s_axilite_ARADDR(s_axilite_ARADDR),

		.s_axilite_RVALID(s_axilite_RVALID),
		.s_axilite_RREADY(s_axilite_RREADY),
		.s_axilite_RDATA(s_axilite_RDATA),
		.s_axilite_RRESP(s_axilite_RRESP),

		//- AXI Stream - Input --------------
		.s_axis_tready(s_axis_tready),
		.s_axis_tvalid(s_axis_tvalid),
		.s_axis_tdata(s_axis_tdata),

		//- AXI Stream - Output -------------
		.m_axis_tready(m_axis_tready),
		.m_axis_tvalid(m_axis_tvalid),
		.m_axis_tdata(m_axis_tdata)
	);

endmodule : thresholding_axi_wrapper

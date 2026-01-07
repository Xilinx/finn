/**
 * Copyright (C) 2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of FINN nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
  */

module $MODULE_NAME$_memstream_wrapper #(
	parameter  SETS = $SETS$,
	parameter  DEPTH = $DEPTH$,
	parameter  WIDTH = $WIDTH$,

	parameter  INIT_FILE = "$INIT_FILE$",
	parameter  RAM_STYLE = "$RAM_STYLE$",
	parameter  PUMPED_MEMORY = $PUMPED_MEMORY$,

	parameter  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2,
	parameter  SET_BITS = SETS > 2? $clog2(SETS) : 1
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	input	ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// AXI-lite Write
	output	s_axilite_AWREADY,
	input	s_axilite_AWVALID,
	input	[2:0]  s_axilite_AWPROT,
	input	[AXILITE_ADDR_WIDTH-1:0]  s_axilite_AWADDR,

	output	s_axilite_WREADY,
	input	s_axilite_WVALID,
	input	[31:0]  s_axilite_WDATA,
	input	[ 3:0]  s_axilite_WSTRB,

	input	s_axilite_BREADY,
	output	s_axilite_BVALID,
	output	[1:0]  s_axilite_BRESP,

	// AXI-lite Read
	output	s_axilite_ARREADY,
	input	s_axilite_ARVALID,
	input	[2:0]  s_axilite_ARPROT,
	input	[AXILITE_ADDR_WIDTH-1:0]  s_axilite_ARADDR,

	input	s_axilite_RREADY,
	output	s_axilite_RVALID,
	output	[ 1:0]  s_axilite_RRESP,
	output	[31:0]  s_axilite_RDATA,

	// Set selector stream (ignored for SETS = 1)
	output	s_axis_0_tready,
	input	s_axis_0_tvalid,
	input	[SET_BITS-1:0]  s_axis_0_tdata,

	// Continuous output stream
	input	m_axis_0_tready,
	output	m_axis_0_tvalid,
	output	[((WIDTH+7)/8)*8-1:0]  m_axis_0_tdata
);

	memstream_axi #(
		.SETS(SETS),
		.DEPTH(DEPTH), .WIDTH(WIDTH),
		.INIT_FILE(INIT_FILE),
		.RAM_STYLE(RAM_STYLE),
		.PUMPED_MEMORY(PUMPED_MEMORY)
	) core (
		.clk(ap_clk), .clk2x(ap_clk2x), .rst(!ap_rst_n),

		// AXI-lite Write
		.awready(s_axilite_AWREADY),
		.awvalid(s_axilite_AWVALID),
		.awprot(s_axilite_AWPROT),
		.awaddr(s_axilite_AWADDR),
		.wready(s_axilite_WREADY),
		.wvalid(s_axilite_WVALID),
		.wdata(s_axilite_WDATA),
		.wstrb(s_axilite_WSTRB),
		.bready(s_axilite_BREADY),
		.bvalid(s_axilite_BVALID),
		.bresp(s_axilite_BRESP),

		// AXI-lite Read
		.arready(s_axilite_ARREADY),
		.arvalid(s_axilite_ARVALID),
		.arprot(s_axilite_ARPROT),
		.araddr(s_axilite_ARADDR),
		.rready(s_axilite_RREADY),
		.rvalid(s_axilite_RVALID),
		.rresp(s_axilite_RRESP),
		.rdata(s_axilite_RDATA),

		// Set selector stream (ignored for SETS = 1)
		.s_axis_0_tready(s_axis_0_tready),
		.s_axis_0_tvalid(s_axis_0_tvalid),
		.s_axis_0_tdata(s_axis_0_tdata),

		// Continuous output stream
		.m_axis_0_tready(m_axis_0_tready),
		.m_axis_0_tvalid(m_axis_0_tvalid),
		.m_axis_0_tdata(m_axis_0_tdata)
	);

endmodule // $MODULE_NAME$_memstream_wrapper

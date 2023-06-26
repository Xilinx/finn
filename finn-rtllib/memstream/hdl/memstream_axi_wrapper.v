/**
 * Copyright (c) 2023, Xilinx
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
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 */

module memstream_axi_wrapper #(
	parameter  DEPTH = 512,
	parameter  WIDTH = 32,

	parameter  INIT_FILE = "",
	parameter  RAM_STYLE = "auto",

	parameter  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axis_0, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// AXI-lite Write
	output	awready,
	input	awvalid,
	input	[2:0]  awprot,
	input	[AXILITE_ADDR_WIDTH-1:0]  awaddr,

	output	wready,
	input	wvalid,
	input	[31:0]  wdata,
	input	[ 3:0]  wstrb,

	input	bready,
	output	bvalid,
	output	[1:0]  bresp,

	// AXI-lite Read
	output	arready,
	input	arvalid,
	input	[2:0]  arprot,
	input	[AXILITE_ADDR_WIDTH-1:0]  araddr,

	input	rready,
	output	rvalid,
	output	[ 1:0]  rresp,
	output	[31:0]  rdata,

	// Continuous output stream
	input	m_axis_0_tready,
	output	m_axis_0_tvalid,
	output	[((WIDTH+7)/8)*8-1:0]  m_axis_0_tdata
);

	localparam  INIT_FILTERED =
`ifdef SYNTHESIS
		RAM_STYLE == "ultra"? "" :
`endif
		INIT_FILE;

	memstream_axi #(
		.DEPTH(DEPTH), .WIDTH(WIDTH),
		.INIT_FILE(INIT_FILTERED),
		.RAM_STYLE(RAM_STYLE)
	) core (
		.clk(ap_clk), .rst(!ap_rst_n),

		// AXI-lite Write
		.awready(awready),
		.awvalid(awvalid),
		.awprot(awprot),
		.awaddr(awaddr),
		.wready(wready),
		.wvalid(wvalid),
		.wdata(wdata),
		.wstrb(wstrb),
		.bready(bready),
		.bvalid(bvalid),
		.bresp(bresp),

		// AXI-lite Read
		.arready(arready),
		.arvalid(arvalid),
		.arprot(arprot),
		.araddr(araddr),
		.rready(rready),
		.rvalid(rvalid),
		.rresp(rresp),
		.rdata(rdata),

		// Continuous output stream
		.m_axis_0_tready(m_axis_0_tready),
		.m_axis_0_tvalid(m_axis_0_tvalid),
		.m_axis_0_tdata(m_axis_0_tdata)
	);

endmodule : memstream_axi_wrapper

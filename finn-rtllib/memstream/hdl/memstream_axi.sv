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

module memstream_axi #(
	int unsigned  DEPTH,
	int unsigned  WIDTH,

	parameter  INIT_FILE = "",
	parameter  RAM_STYLE = "auto",

	localparam int unsigned  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// AXI-lite Write
	output	logic  awready,
	input	logic  awvalid,
	input	logic [2:0]  awprot,
	input	logic [AXILITE_ADDR_WIDTH-1:0]  awaddr,

	output	logic  wready,
	input	logic  wvalid,
	input	logic [31:0]  wdata,
	input	logic [ 3:0]  wstrb,

	input	logic  bready,
	output	logic  bvalid,
	output	logic [1:0]  bresp,

	// AXI-lite Read
	output	logic  arready,
	input	logic  arvalid,
	input	logic [2:0]  arprot,
	input	logic [AXILITE_ADDR_WIDTH-1:0]  araddr,

	input	logic  rready,
	output	logic  rvalid,
	output	logic [ 1:0]  rresp,
	output	logic [31:0]  rdata,

	// Continuous output stream
	input	logic  m_axis_0_tready,
	output	logic  m_axis_0_tvalid,
	output	logic [((WIDTH+7)/8)*8-1:0]  m_axis_0_tdata
);

	//-----------------------------------------------------------------------
	// AXI-lite to ap_memory Adapter
	uwire [31:0]  config_address;
	uwire  config_ce;
	uwire  config_we;
	uwire  config_rack;
	uwire [WIDTH-1:0]  config_d0;
	uwire [WIDTH-1:0]  config_q0;
	axi4lite_if #(
		.ADDR_WIDTH(AXILITE_ADDR_WIDTH),
		.DATA_WIDTH(32),
		.IP_DATA_WIDTH(WIDTH)
	) config_if (
		.aclk(clk), .aresetn(!rst),

		// Write Channels
		.awready, .awvalid, .awaddr, .awprot,
		.wready,  .wvalid,  .wdata,  .wstrb,
		.bready,  .bvalid,  .bresp,

		// Read Channels
		.arready, .arvalid, .araddr, .arprot,
		.rready,  .rvalid,  .rresp,  .rdata,

		// IP-side Interface
		.ip_en(config_ce),
		.ip_wen(config_we),
		.ip_addr(config_address),
		.ip_wdata(config_d0),
		.ip_rack(config_rack),
		.ip_rdata(config_q0)
	);

	//-----------------------------------------------------------------------
	// Streaming Memory Backend
	memstream #(
		.DEPTH(DEPTH),
		.WIDTH(WIDTH),
		.INIT_FILE(INIT_FILE),
		.RAM_STYLE(RAM_STYLE)
	) mem (
		.clk, .rst,

		.config_address,
		.config_ce,
		.config_we,
		.config_d0,
		.config_q0,
		.config_rack,

		.ordy(m_axis_0_tready),
		.ovld(m_axis_0_tvalid),
		.odat(m_axis_0_tdata[WIDTH-1:0])
	);
	if($bits(m_axis_0_tdata) > WIDTH) begin
		assign	m_axis_0_tdata[$left(m_axis_0_tdata):WIDTH] = '0;
	end

endmodule : memstream_axi

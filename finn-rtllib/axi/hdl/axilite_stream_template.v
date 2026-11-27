/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************
 * @brief	Verilog wrapper of axilite_stream for use as a block design module
 *	reference, which must not be a SystemVerilog module.
 ***************************************************************************/

module $MODULE_NAME$ #(
	parameter  ADDR_WIDTH = $ADDR_WIDTH$,
	parameter  IP_DATA_WIDTH = $IP_DATA_WIDTH$,
	parameter  IP_ADDR_WIDTH = $IP_ADDR_WIDTH$,

	parameter  REQ_WIDTH = (IP_DATA_WIDTH + IP_ADDR_WIDTH + 1 + 7)/8 * 8,
	parameter  RSP_WIDTH = (IP_DATA_WIDTH + 7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:m_axis_req:s_axis_rsp, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// AXI-lite Write
	output	s_axilite_AWREADY,
	input	s_axilite_AWVALID,
	input	[2:0]  s_axilite_AWPROT,
	input	[ADDR_WIDTH-1:0]  s_axilite_AWADDR,

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
	input	[ADDR_WIDTH-1:0]  s_axilite_ARADDR,

	input	s_axilite_RREADY,
	output	s_axilite_RVALID,
	output	[ 1:0]  s_axilite_RRESP,
	output	[31:0]  s_axilite_RDATA,

	// Request stream
	input	m_axis_req_tready,
	output	m_axis_req_tvalid,
	output	[REQ_WIDTH-1:0]  m_axis_req_tdata,

	// Response stream
	output	s_axis_rsp_tready,
	input	s_axis_rsp_tvalid,
	input	[RSP_WIDTH-1:0]  s_axis_rsp_tdata
);

	axilite_stream #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.IP_DATA_WIDTH(IP_DATA_WIDTH),
		.IP_ADDR_WIDTH(IP_ADDR_WIDTH)
	) core (
		.aclk(ap_clk), .aresetn(ap_rst_n),

		.s_axilite_AWREADY(s_axilite_AWREADY), .s_axilite_AWVALID(s_axilite_AWVALID),
		.s_axilite_AWPROT(s_axilite_AWPROT),   .s_axilite_AWADDR(s_axilite_AWADDR),
		.s_axilite_WREADY(s_axilite_WREADY),   .s_axilite_WVALID(s_axilite_WVALID),
		.s_axilite_WSTRB(s_axilite_WSTRB),     .s_axilite_WDATA(s_axilite_WDATA),
		.s_axilite_BREADY(s_axilite_BREADY),   .s_axilite_BVALID(s_axilite_BVALID),
		.s_axilite_BRESP(s_axilite_BRESP),
		.s_axilite_ARREADY(s_axilite_ARREADY), .s_axilite_ARVALID(s_axilite_ARVALID),
		.s_axilite_ARPROT(s_axilite_ARPROT),   .s_axilite_ARADDR(s_axilite_ARADDR),
		.s_axilite_RREADY(s_axilite_RREADY),   .s_axilite_RVALID(s_axilite_RVALID),
		.s_axilite_RRESP(s_axilite_RRESP),     .s_axilite_RDATA(s_axilite_RDATA),

		.m_axis_req_tdata(m_axis_req_tdata), .m_axis_req_tvalid(m_axis_req_tvalid), .m_axis_req_tready(m_axis_req_tready),
		.s_axis_rsp_tdata(s_axis_rsp_tdata), .s_axis_rsp_tvalid(s_axis_rsp_tvalid), .s_axis_rsp_tready(s_axis_rsp_tready)
	);

endmodule // $MODULE_NAME$

/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************
 * @brief	AXI-Lite slave issuing its accesses as request and response streams.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *  Translates word accesses on an AXI-Lite slave (see axilite) into a
 *  request stream towards a kernel with a stream interface for its
 *  configuration, such as an HLS kernel. A request carries, from the LSB:
 *     { we, addr[IP_ADDR_WIDTH], val[IP_DATA_WIDTH] }
 *  matching a struct { val; addr; we; } aggregated with compact=bit by HLS,
 *  zero-padded to whole bytes. A read request (!we) must be answered by
 *  exactly one response carrying the read value, in request order.
 *
 *  The axilite adapter issues accesses without a ready from the kernel,
 *  which, as a stalling pipeline, may not accept requests for a while. The
 *  requests are therefore queued, and new accesses are only accepted on the
 *  AXI-Lite channels while the queue can hold every access in flight: at
 *  most a write, a read and the issued access in its output register.
 *****************************************************************************/

module axilite_stream #(
	int unsigned  ADDR_WIDTH,	// AXI-Lite byte address
	int unsigned  IP_DATA_WIDTH,	// value width, at most 32 bits
	int unsigned  IP_ADDR_WIDTH = ADDR_WIDTH - 2,	// word address of the kernel

	localparam int unsigned  REQ_WIDTH = (IP_DATA_WIDTH + IP_ADDR_WIDTH + 1 + 7)/8 * 8,
	localparam int unsigned  RSP_WIDTH = (IP_DATA_WIDTH + 7)/8 * 8
)(
	// Global Control
	input	logic  aclk,
	input	logic  aresetn,

	// AXI-Lite Slave
	output	logic                   s_axilite_AWREADY,
	input	logic                   s_axilite_AWVALID,
	input	logic [2:0]             s_axilite_AWPROT,
	input	logic [ADDR_WIDTH-1:0]  s_axilite_AWADDR,

	output	logic         s_axilite_WREADY,
	input	logic         s_axilite_WVALID,
	input	logic [ 3:0]  s_axilite_WSTRB,
	input	logic [31:0]  s_axilite_WDATA,

	input	logic        s_axilite_BREADY,
	output	logic        s_axilite_BVALID,
	output	logic [1:0]  s_axilite_BRESP,

	output	logic                   s_axilite_ARREADY,
	input	logic                   s_axilite_ARVALID,
	input	logic [2:0]             s_axilite_ARPROT,
	input	logic [ADDR_WIDTH-1:0]  s_axilite_ARADDR,

	input	logic         s_axilite_RREADY,
	output	logic         s_axilite_RVALID,
	output	logic [ 1:0]  s_axilite_RRESP,
	output	logic [31:0]  s_axilite_RDATA,

	// Request Stream
	output	logic [REQ_WIDTH-1:0]  m_axis_req_tdata,
	output	logic  m_axis_req_tvalid,
	input	logic  m_axis_req_tready,

	// Response Stream
	input	logic [RSP_WIDTH-1:0]  s_axis_rsp_tdata,
	input	logic  s_axis_rsp_tvalid,
	output	logic  s_axis_rsp_tready
);

`default_nettype none

	initial begin
		if(IP_DATA_WIDTH > 32) begin
			$error("%m: Values of %0d bits exceed the AXI-Lite data width.", IP_DATA_WIDTH);
			$finish;
		end
		if(IP_ADDR_WIDTH + 2 > ADDR_WIDTH) begin
			$error("%m: Word address of %0d bits exceeds the AXI-Lite address.", IP_ADDR_WIDTH);
			$finish;
		end
	end

	// Up to three accesses are in flight towards the queue (see description),
	// which takes new ones only with room for all of them.
	localparam int unsigned  DEPTH = 8;
	localparam int unsigned  IN_FLIGHT = 3;
	uwire  accept;

	//=======================================================================
	// AXI-Lite Adapter
	uwire  ip_en;
	uwire  ip_wen;
	uwire [ADDR_WIDTH-3:0]  ip_addr;
	uwire [IP_DATA_WIDTH-1:0]  ip_wdata;
	if(1) begin : blkAxi
		uwire  awready;
		uwire  wready;
		uwire  arready;
		axilite #(.ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(32), .IP_DATA_WIDTH(IP_DATA_WIDTH)) axi (
			.aclk, .aresetn,

			.awready, .awvalid(accept && s_axilite_AWVALID), .awaddr(s_axilite_AWADDR), .awprot(s_axilite_AWPROT),
			.wready,  .wvalid(accept && s_axilite_WVALID),   .wdata(s_axilite_WDATA),   .wstrb(s_axilite_WSTRB),
			.bready(s_axilite_BREADY), .bvalid(s_axilite_BVALID), .bresp(s_axilite_BRESP),

			.arready, .arvalid(accept && s_axilite_ARVALID), .araddr(s_axilite_ARADDR), .arprot(s_axilite_ARPROT),
			.rready(s_axilite_RREADY), .rvalid(s_axilite_RVALID), .rresp(s_axilite_RRESP), .rdata(s_axilite_RDATA),

			.ip_en, .ip_wen, .ip_addr, .ip_wdata,
			.ip_rack(s_axis_rsp_tvalid), .ip_rdata(s_axis_rsp_tdata[IP_DATA_WIDTH-1:0])
		);
		assign	s_axilite_AWREADY = accept && awready;
		assign	s_axilite_WREADY  = accept && wready;
		assign	s_axilite_ARREADY = accept && arready;
	end : blkAxi
	assign	s_axis_rsp_tready = 1;

	//=======================================================================
	// Request Queue: SRL-style, Ptr = count - 1
	if(1) begin : blkQueue
		typedef logic [IP_DATA_WIDTH + IP_ADDR_WIDTH:0]  req_t;
		req_t  Buf[DEPTH] = '{ default: 'x };
		logic signed [$clog2(DEPTH):0]  Ptr = -1;

		uwire  vld = !Ptr[$left(Ptr)];
		uwire  pop = vld && m_axis_req_tready;
		always_ff @(posedge aclk) begin
			if(ip_en) begin
				for(int unsigned  i = DEPTH-1; i > 0; i--)  Buf[i] <= Buf[i-1];
				Buf[0] <= { ip_wen, IP_ADDR_WIDTH'(ip_addr), ip_wdata };
			end
		end
		always_ff @(posedge aclk) begin
			if(!aresetn)  Ptr <= -1;
			else          Ptr <= Ptr + (ip_en == pop? 0 : ip_en? 1 : -1);
		end

		// count + IN_FLIGHT <= DEPTH after accepting one more access
		assign	accept = (Ptr < $signed(DEPTH - IN_FLIGHT));
		assign	m_axis_req_tvalid = vld;
		assign	m_axis_req_tdata  = REQ_WIDTH'(Buf[vld? Ptr : 0]);
	end : blkQueue

`default_nettype wire

endmodule : axilite_stream

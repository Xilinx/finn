/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *	 this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *	 contributors may be used to endorse or promote products derived from
 *	 this software without specific prior written permission.
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
 *****************************************************************************/

module axilite #(
	int unsigned  ADDR_WIDTH,
	int unsigned  DATA_WIDTH,	// AXI4 spec requires this to be strictly 32 or 64
	int unsigned  IP_DATA_WIDTH,

	localparam int unsigned  BSEL_BITS = $clog2(DATA_WIDTH/8),
	localparam int unsigned  FOLD      = 1 + (IP_DATA_WIDTH-1)/DATA_WIDTH,
	localparam int unsigned  WSEL_BITS = $clog2(FOLD),
	localparam int unsigned  IP_ADDR_WIDTH0 = ADDR_WIDTH - WSEL_BITS - BSEL_BITS,
	localparam int unsigned  IP_ADDR_WIDTH = IP_ADDR_WIDTH0? IP_ADDR_WIDTH0 : 1
)(
	// Global Control
	input	logic  aclk,
	input	logic  aresetn,

	// Write Channels
	output	logic  awready,
	input	logic  awvalid,
	input	logic [2:0]  awprot, /* ignored */
	input	logic [ADDR_WIDTH-1:0]  awaddr,

	output	logic  wready,
	input	logic  wvalid,
	input	logic [DATA_WIDTH/8-1:0]  wstrb, /* ignored */
	input	logic [DATA_WIDTH  -1:0]  wdata,

	input	logic  bready,
	output	logic  bvalid,
	output	logic [1:0]  bresp,	// 00 = OKAY, 10 = SLVERR (write error)

	// Read Channels
	output	logic  arready,
	input	logic  arvalid,
	input	logic [2:0]  arprot, /* ignored */
	input	logic [ADDR_WIDTH-1:0]  araddr,

	input	logic  rready,
	output	logic  rvalid,
	output	logic [1:0]  rresp, // 00 = OKAY, 10 = SLVERR (read error)
	output	logic [DATA_WIDTH-1:0]  rdata,

	// IP-side Interface
	output	logic  ip_en,
	output	logic  ip_wen,
	output	logic [IP_ADDR_WIDTH-1:0]  ip_addr,
	output	logic [IP_DATA_WIDTH-1:0]  ip_wdata,
	input	logic  ip_rack,
	input	logic [IP_DATA_WIDTH-1:0]  ip_rdata
);

	uwire  rst = !aresetn;

	//-----------------------------------------------------------------------
	// AXI-lite Frontend
	localparam bit [1:0]  RESP_OKAY   = 'b00;
	localparam bit [1:0]  RESP_SLVERR = 'b10;

	typedef struct {
		logic  vld;
		logic [IP_ADDR_WIDTH+WSEL_BITS-1:0]  val;
	} addr_t;
	typedef struct {
		logic  vld;
		logic [DATA_WIDTH-1:0]  val;
	} data_t;

	// Address & Data Captures
	addr_t  RAddr = '{ vld: 0, val: 'x };
	logic   RLock = 0;
	addr_t  WAddr = '{ vld: 0, val: 'x };
	data_t  WData = '{ vld: 0, val: 'x };
	uwire  clr_rd;
	uwire  clr_wr;
	always_ff @(posedge aclk) begin
		if(rst || clr_rd) begin
			RAddr <= '{ vld: 0, val: 'x };
			RLock <= 0;
		end
		else begin
			if(!RAddr.vld)  RAddr <= '{ vld: arvalid, val: araddr >> BSEL_BITS };
			if(snk_re)      RLock <= 1;
		end
	end
	always_ff @(posedge aclk) begin
		if(rst || clr_wr) begin
			WAddr <= '{ vld: 0, val: 'x };
			WData <= '{ vld: 0, val: 'x };
		end
		else begin
			if(!WAddr.vld)  WAddr <= '{ vld: awvalid, val: awaddr >> BSEL_BITS };
			if(!WData.vld)  WData <= '{ vld: wvalid,  val: wdata };
		end
	end
	assign	arready = !RAddr.vld;
	assign	awready = !WAddr.vld;
	assign	wready  = !WData.vld;

	// Reply Buffers
	logic  WIssued = 0;
	data_t  RData = '{ vld: 0, val: 'x };

	uwire  snk_we = WAddr.vld && WData.vld && !WIssued;
	uwire  snk_re = !snk_we && RAddr.vld && !RLock && !RData.vld;
	uwire [IP_ADDR_WIDTH+WSEL_BITS-1:0]  snk_addr = snk_we? WAddr.val : RAddr.val;
	uwire [IP_DATA_WIDTH          -1:0]  snk_data = WData.val;

	// Write
	always_ff @(posedge aclk) begin
		if(rst)  WIssued <= 0;
		else     WIssued <= snk_we || (WIssued && !bready);
	end
	assign	clr_wr = WIssued && bready;
	assign	bvalid = WIssued;
	assign	bresp  = RESP_OKAY;

	// Read
	uwire  src_ack;
	uwire [DATA_WIDTH-1:0]  src_data;
	always_ff @(posedge aclk) begin
		if(rst)           RData <= '{ vld: 0, val: 'x };
		else if(src_ack)  RData <= '{ vld: 1, val: src_data };
		else if(rready)   RData <= '{ vld: 0, val: 'x };
	end
	assign	rvalid = RData.vld;
	assign	rdata  = RData.val;
	assign	rresp  = RESP_OKAY;

	//-----------------------------------------------------------------------
	// IP Backend
	if(FOLD == 1) begin : genNoFold

		// Command Issue
		logic  IpEn =  0;
		logic  IpWe = 'x;
		logic [IP_DATA_WIDTH-1:0]  IpData = 'x;
		always_ff @(posedge aclk) begin
			if(rst) begin
				IpEn <=  0;
				IpWe <= 'x;
				IpData <= 'x;
			end
			else begin
				IpEn <= snk_we || snk_re;
				IpWe <= snk_we;
				IpData <= snk_data;
			end
		end
		if(IP_ADDR_WIDTH0 == 0)  assign  ip_addr = 0;
		else begin
			logic [IP_ADDR_WIDTH-1:0]  IpAddr = 'x;
			always_ff @(posedge aclk) begin
				if(rst)  IpAddr <= 'x;
				else     IpAddr <= snk_addr;
			end
			assign	ip_addr = IpAddr;
		end
		assign	ip_en = IpEn;
		assign	ip_wen = IpWe;
		assign	ip_wdata = IpData;

		// Reply Capture
		assign	src_ack = ip_rack;
		assign	src_data = ip_rdata;
		assign	clr_rd = snk_re;

	end : genNoFold
	else begin : genFold
		uwire [WSEL_BITS-1:0]  ofs_fold = snk_addr[WSEL_BITS-1:0];
		uwire  sel_fold[FOLD];
		for(genvar  i = 0; i < FOLD; i++)  assign  sel_fold[i] = (ofs_fold == i);

		// Command Issue
		logic  IpEn =  0;
		logic  IpWe = 'x;
		logic [ADDR_WIDTH-BSEL_BITS-1:0]  IpAddr = 'x;
		logic [FOLD-1:0][DATA_WIDTH-1:0]  IpWData = 'x;
		always_ff @(posedge aclk) begin
			if(rst) begin
				IpEn <=  0;
				IpWe <= 'x;
				IpAddr <= 'x;
				IpWData <= 'x;
			end
			else begin
				IpEn <= (snk_we && sel_fold[FOLD-1]) || (snk_re && sel_fold[0]);
				IpWe <= snk_we;
				if(snk_we || snk_re)  IpAddr <= snk_addr;
				if(snk_we)  foreach(sel_fold[i])  if(sel_fold[i])  IpWData[i] <= snk_data;
			end
		end
		assign	ip_en = IpEn;
		assign	ip_wen = IpWe;
		assign	ip_wdata = IpWData;
		if(IP_ADDR_WIDTH0 == 0)  assign  ip_addr = 0;
		else  assign  ip_addr = IpAddr[$left(IpAddr):WSEL_BITS];

		// Reply Capture
		logic  IpRAck = 0;
		logic [FOLD-1:0][DATA_WIDTH-1:0]  IpRData = 'x;
		always_ff @(posedge aclk) begin
			if(rst) begin
				IpRAck <= 0;
				IpRData <= 'x;
			end
			else begin
				IpRAck <= ip_rack || (snk_re && !sel_fold[0]);
				if(ip_rack)  IpRData <= ip_rdata;
			end
		end
		assign	src_ack = IpRAck;
		assign	src_data = IpAddr[WSEL_BITS-1:0] < FOLD? IpRData[IpAddr[WSEL_BITS-1:0]] : '0;
		assign	clr_rd = src_ack;

	end : genFold

endmodule : axilite

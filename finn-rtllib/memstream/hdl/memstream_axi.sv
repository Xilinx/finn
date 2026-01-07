/**
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
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
	int unsigned  SETS,
	int unsigned  DEPTH,
	int unsigned  WIDTH,

	parameter  INIT_FILE = "",
	parameter  RAM_STYLE = "auto",
	bit  PUMPED_MEMORY = 0,

	localparam int unsigned  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2,
	localparam int unsigned  SET_BITS = SETS > 2? $clog2(SETS) : 1
)(
	// Global Control
	input	logic  clk,
	input	logic  clk2x,
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

	// Set selector stream (ignored for SETS = 1)
	output	logic  s_axis_0_tready,
	input	logic  s_axis_0_tvalid,
	input	logic [SET_BITS-1:0]  s_axis_0_tdata,

	// Continuous output stream
	input	logic  m_axis_0_tready,
	output	logic  m_axis_0_tvalid,
	output	logic [((WIDTH+7)/8)*8-1:0]  m_axis_0_tdata
);

	localparam int unsigned  IP_ADDR_WIDTH0 = $clog2(DEPTH);
	localparam int unsigned  IP_ADDR_WIDTH = IP_ADDR_WIDTH0? IP_ADDR_WIDTH0 : 1;

	//-----------------------------------------------------------------------
	// AXI-lite to ap_memory Adapter
	uwire [IP_ADDR_WIDTH-1:0]  config_address;
	uwire  config_ce;
	uwire  config_we;
	uwire  config_rack;
	uwire [WIDTH-1:0]  config_d0;
	uwire [WIDTH-1:0]  config_q0;
	axilite #(
		.ADDR_WIDTH(AXILITE_ADDR_WIDTH),
		.DATA_WIDTH(32),
		.IP_DATA_WIDTH(WIDTH)
	) cfg (
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
	localparam int unsigned  DEPTH_EFF = PUMPED_MEMORY? 2*DEPTH     : DEPTH;
	localparam int unsigned  WIDTH_EFF = PUMPED_MEMORY? (WIDTH+1)/2 : WIDTH;
	uwire  mem_ce;
	uwire  mem_we;
	uwire [         31:0]  mem_a0;
	uwire [WIDTH_EFF-1:0]  mem_d0;
	uwire  mem_rack;
	uwire [WIDTH_EFF-1:0]  mem_q0;
	uwire  set_rdy;
	uwire  set_vld;
	uwire [WIDTH_EFF-1:0]  set_dat;
	uwire  mem_rdy;
	uwire  mem_vld;
	uwire [WIDTH_EFF-1:0]  mem_dat;
	if(!PUMPED_MEMORY) begin : genUnpumped
		assign	mem_ce = config_ce;
		assign	mem_we = config_we;
		assign	mem_a0 = config_address;
		assign	mem_d0 = config_d0;
		assign	config_rack = mem_rack;
		assign	config_q0   = mem_q0;

		assign	s_axis_0_tready = set_rdy;
		assign	set_vld = s_axis_0_tvalid;
		assign	set_dat = s_axis_0_tdata;

		assign	mem_rdy = m_axis_0_tready;
		assign	m_axis_0_tvalid = mem_vld;
		assign	m_axis_0_tdata  = mem_dat;

		memstream #(
			.SETS(SETS),
			.DEPTH(DEPTH_EFF),
			.WIDTH(WIDTH_EFF),
			.INIT_FILE(INIT_FILE),
			.RAM_STYLE(RAM_STYLE)
		) mem (
			.clk(clk), .rst,

			.config_address(mem_a0),
			.config_ce(mem_ce),
			.config_we(mem_we),
			.config_d0(mem_d0),
			.config_q0(mem_q0),
			.config_rack(mem_rack),

			.srdy(set_rdy), .svld(set_vld), .sidx(set_dat),
			.ordy(mem_rdy), .ovld(mem_vld), .odat(mem_dat)
		);
	end : genUnpumped
	else begin : genPumped

		// Identifier of fast active clock edge coinciding with slow active clock edge
		logic Active;
		always_ff @(posedge clk2x) begin
			if(rst)  Active <= 0;
			else     Active <= !Active;
		end

		// Clock translation for config requests, which are spread across two fast cycles
		logic  Cfg2x_CE =  0;
		logic  Cfg2x_WE = 'x;
		logic [30     :0]  Cfg2x_A0 = 'x;
		logic [WIDTH-1:0]  Cfg2x_D0 = 'x;
		always_ff @(posedge clk2x) begin
			if(rst) begin
				Cfg2x_CE <=  0;
				Cfg2x_WE <= 'x;
				Cfg2x_A0 <= 'x;
				Cfg2x_D0 <= 'x;
			end
			else begin
				if(Active) begin
					Cfg2x_CE <= config_ce;
					Cfg2x_WE <= config_we;
					Cfg2x_A0 <= config_address;
				end
				Cfg2x_D0 <= Active? config_d0 : { {(WIDTH-WIDTH_EFF){1'bx}}, Cfg2x_D0[WIDTH-1:WIDTH_EFF] };
			end
		end
		assign	mem_ce = Cfg2x_CE;
		assign	mem_we = Cfg2x_WE;
		assign	mem_a0 = { Cfg2x_A0, Active };
		assign	mem_d0 = Cfg2x_D0;

		// Assemble two consecutive read replies into one
		logic [1:0]  Cfg2x_Rack =  0;
		logic [2*WIDTH_EFF-1:0]  Cfg2x_Q0 = 'x;
		always_ff @(posedge clk2x) begin
			if(rst) begin
				Cfg2x_Rack <=  0;
				Cfg2x_Q0   <= 'x;
			end
			else begin
				if(mem_rack)  Cfg2x_Q0 <= { mem_q0, Cfg2x_Q0[WIDTH_EFF+:WIDTH_EFF] };
				// Count replies and clear when seen in slow clock domain
				Cfg2x_Rack <= Cfg2x_Rack + mem_rack;
				if(Cfg2x_Rack[1] && Active)  Cfg2x_Rack <= 0;
			end
		end
		assign	config_rack = Cfg2x_Rack[1];
		assign	config_q0   = Cfg2x_Q0[WIDTH-1:0];

		// Thin out a set index into a single fast clock
		if(1) begin : blkSetSelIn
			logic [SET_BITS-1:0]  IBuf = 'x;
			logic  IVld = 0;
			always_ff @(posedge clk2x) begin
				if(IVld)  IVld <= !set_rdy;
				else if(Active) begin
					IBuf <= s_axis_0_tdata;
					IVld <= s_axis_0_tvalid;
				end
			end
			assign	set_dat = IBuf;
			assign	set_vld = IVld;

			assign	s_axis_0_tready = !IVld;
		end : blkSetSelIn

		// Assemble two consecutive stream outputs into
		if(1) begin : blkStreamOut
			logic [3:0][WIDTH_EFF-1:0]  SBuf = 'x;
			logic [2:0]  SCnt = 0;	// 0..4
			logic  SVld = 0;
			always_ff @(posedge clk2x) begin
				if(rst) begin
					SBuf <= 'x;
					SCnt <=  0;
					SVld <=  0;
				end
				else begin
					automatic logic [4:0][WIDTH_EFF-1:0]  sbuf = { {WIDTH_EFF{1'bx}}, SBuf };
					automatic logic [2:0]  scnt = SCnt;

					sbuf[scnt] = mem_dat;
					if(m_axis_0_tvalid && (Active && m_axis_0_tready)) begin
						scnt[2:1] = { 1'b0, scnt[2] };
						sbuf[1:0] = sbuf[3:2];
					end
					scnt += mem_rdy && mem_vld;

					SBuf <= sbuf[3:0];
					SCnt <= scnt;
					if(Active)  SVld <= |scnt[2:1];
				end
			end
			assign	mem_rdy = !SCnt[2];
			assign	m_axis_0_tvalid = SVld;
			assign	m_axis_0_tdata  = { SBuf[1][0+:WIDTH-WIDTH_EFF], SBuf[0] };
		end : blkStreamOut

		memstream #(
			.SETS(SETS),
			.DEPTH(DEPTH_EFF),
			.WIDTH(WIDTH_EFF),
			.INIT_FILE(INIT_FILE),
			.RAM_STYLE(RAM_STYLE)
		) mem (
			.clk(clk2x), .rst,

			.config_address(mem_a0),
			.config_ce(mem_ce),
			.config_we(mem_we),
			.config_d0(mem_d0),
			.config_q0(mem_q0),
			.config_rack(mem_rack),

			.srdy(set_rdy), .svld(set_vld), .sidx(set_dat),
			.ordy(mem_rdy), .ovld(mem_vld), .odat(mem_dat)
		);
	end : genPumped
	if($bits(m_axis_0_tdata) > WIDTH) begin
		assign	m_axis_0_tdata[$left(m_axis_0_tdata):WIDTH] = '0;
	end

endmodule : memstream_axi

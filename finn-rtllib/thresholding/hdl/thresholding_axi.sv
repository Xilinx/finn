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
 * @brief	All-AXI interface adapter for thresholding module.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module thresholding_axi #(
	int unsigned  N,	// output precision
	int unsigned  M,	// input/threshold precision
	int unsigned  C		// Channels
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                    s_axilite_AWVALID,
	output	logic                    s_axilite_AWREADY,
	input	logic [$clog2(C)+N-1:0]  s_axilite_AWADDR,

	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,

	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	// Reading
	input	logic        s_axilite_ARVALID,
	output	logic        s_axilite_ARREADY,
	input	logic [0:0]  s_axilite_ARADDR,

	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [((M+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [((N+7)/8)*8-1:0]  m_axis_tdata
);
	//- Global Control ------------------------------------------------------
	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;

	//- AXI Lite: Threshold Configuration -----------------------------------
	uwire  twe;
	uwire [$clog2(C)+N-1:0]  twa;
	uwire [          M-1:0]  twd;
	if(1) begin : blkAxiLite
		logic  WABusy = 0;
		logic  WDBusy = 0;
		logic [$clog2(C)+N-1:0]  Addr = 'x;
		logic [          M-1:0]  Data = 'x;

		assign	twe = WABusy && WDBusy;
		assign	twa = Addr;
		assign	twd = Data;

		uwire  clr_wr = rst || (twe && s_axilite_BREADY);
		always_ff @(posedge clk) begin : blockName
			if(clr_wr) begin
				WABusy <= 0;
				Addr <= 'x;
				WDBusy <= 0;
				Data <= 'x;
			end
			else begin
				if(!WABusy) begin
					WABusy <= s_axilite_AWVALID;
					Addr   <= s_axilite_AWADDR[$clog2(C)+N-1:0];
				end
				if(!WDBusy) begin
					WDBusy <= s_axilite_WVALID;
					Data   <= s_axilite_WDATA[M-1:0];
				end
			end
		end
		assign	s_axilite_AWREADY = !WABusy;
		assign	s_axilite_WREADY  = !WDBusy;
		assign	s_axilite_BVALID  = WABusy && WDBusy;
		assign	s_axilite_BRESP   = '0; // OK

		// Answer all reads with '1
		logic  RValid =  0;
		uwire  clr_rd = rst || (RValid && s_axilite_RREADY);
		always_ff @(posedge clk) begin
			if(clr_rd)        RValid <=  0;
			else if(!RValid)  RValid <= s_axilite_ARVALID;
		end
		assign	s_axilite_ARREADY = !RValid;
		assign	s_axilite_RVALID  = RValid;
		assign	s_axilite_RDATA   = '1;
		assign	s_axilite_RRESP   = '0; // OK

	end : blkAxiLite

	//- IO-Sandwich with two-stage output buffer for containing a local enable
	uwire  en;
	uwire [N-1:0]  odat;
	uwire  ovld;
	if(1) begin : blkOutputDecouple
		typedef struct {
			logic          vld;
			logic [N-1:0]  dat;
		} buf_t;
		buf_t  Buf[2] = '{ default: '{ vld: 0, dat: 'x } };
		always_ff @(posedge clk) begin
			if(rst)  Buf <= '{ default: '{ vld: 0, dat: 'x } };
			else begin
				if(!Buf[1].vld || m_axis_tready) begin
					Buf[1] <= '{
						vld: Buf[0].vld || ovld,
						dat: Buf[0].vld? Buf[0].dat : odat
					};
				end
				Buf[0].vld <= Buf[1].vld && !m_axis_tready && (Buf[0].vld || ovld);
				if(!Buf[0].vld)  Buf[0].dat <= odat;
			end
		end
		assign	en = !Buf[0].vld;

		assign	m_axis_tvalid = Buf[1].vld;
		assign	m_axis_tdata  = Buf[1].dat;

	end : blkOutputDecouple

	localparam int unsigned  C_BITS = C < 2? 1 : $clog2(C);
	uwire  ivld = s_axis_tvalid;
	uwire [C_BITS-1:0]  icnl;
	uwire [M     -1:0]  idat = s_axis_tdata[M-1:0];
	assign	s_axis_tready = en;
	if(C == 1)  assign  icnl = 'x;
	else begin
		logic [C_BITS-1:0]  Chnl = 0;
		logic               Last = 0;
		uwire  inc = ivld && en;
		uwire  clr = rst || (Last && inc);
		always_ff @(posedge clk) begin
			if(clr) begin
				Chnl <= 0;
				Last <= 0;
			end
			else if(inc) begin
				Chnl <= Chnl + 1;
				Last <= (~Chnl & (C-2)) == 0;
			end
		end
		assign	icnl = Chnl;
	end

	// Core Thresholding Module
	thresholding #(.N(N), .M(M), .C(C)) core (
		.clk, .rst,
		.twe, .twa, .twd,
		.en,
		.ivld, .icnl, .idat,
		.ovld, .ocnl(), .odat
	);

endmodule : thresholding_axi

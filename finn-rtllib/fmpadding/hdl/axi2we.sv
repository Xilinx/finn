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
 * @brief	AXI-Light adapter for trivial write enable interface.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module axi2we #(
	int unsigned  ADDR_BITS
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	                 s_axilite_AWVALID,
	output	                 s_axilite_AWREADY,
	input	[ADDR_BITS-1:0]  s_axilite_AWADDR,

	input	        s_axilite_WVALID,
	output	        s_axilite_WREADY,
	input	[31:0]  s_axilite_WDATA,
	input	[ 3:0]  s_axilite_WSTRB,

	output	       s_axilite_BVALID,
	input	       s_axilite_BREADY,
	output	[1:0]  s_axilite_BRESP,

	// Reading tied to all-ones
	input	       s_axilite_ARVALID,
	output	       s_axilite_ARREADY,
	input	[3:0]  s_axilite_ARADDR,

	output	        s_axilite_RVALID,
	input	        s_axilite_RREADY,
	output	[31:0]  s_axilite_RDATA,
	output	[ 1:0]  s_axilite_RRESP,

	// Write Enable Interface
	output	logic                  we,
	output	logic [ADDR_BITS-1:0]  wa,
	output	logic [         31:0]  wd
);

	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;


	logic  WABusy = 0;
	logic  WDBusy = 0;
	logic [ADDR_BITS-1:0]  Addr = 'x;
	logic [         31:0]  Data = 'x;

	assign	we = WABusy && WDBusy && s_axilite_BREADY;
	assign	wa = Addr;
	assign	wd = Data;

	uwire  clr_wr = rst || we;
	always_ff @(posedge clk) begin
		if(clr_wr) begin
			WABusy <= 0;
			Addr <= 'x;
			WDBusy <= 0;
			Data <= 'x;
		end
		else begin
			if(!WABusy) begin
				WABusy <= s_axilite_AWVALID;
				Addr   <= s_axilite_AWADDR;
			end
			if(!WDBusy) begin
				WDBusy <= s_axilite_WVALID;
				Data   <= s_axilite_WDATA;
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

endmodule : axi2we

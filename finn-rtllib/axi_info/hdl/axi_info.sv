/******************************************************************************
 *  Copyright (c) 2022, Advanced Micro Devices, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Read-only exposure of compiled-in info data on AXI-lite.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 *******************************************************************************/
module axi_info #(
	int unsigned  N,
	int unsigned  S_AXI_DATA_WIDTH = 32,
	bit [S_AXI_DATA_WIDTH-1:0]  DATA[N]
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  s_axi_AWVALID,
	output	logic                  s_axi_AWREADY,
	input	logic [$clog2(N)+1:0]  s_axi_AWADDR,

	input	logic                           s_axi_WVALID,
	output	logic                           s_axi_WREADY,
	input	logic [S_AXI_DATA_WIDTH  -1:0]  s_axi_WDATA,
	input	logic [S_AXI_DATA_WIDTH/8-1:0]  s_axi_WSTRB,

	output	logic        s_axi_BVALID,
	input	logic        s_axi_BREADY,
	output	logic [1:0]  s_axi_BRESP,

	// Reading
	input	logic                  s_axi_ARVALID,
	output	logic                  s_axi_ARREADY,
	input	logic [$clog2(N)+1:0]  s_axi_ARADDR,

	output	logic                         s_axi_RVALID,
	input	logic                         s_axi_RREADY,
	output	logic [S_AXI_DATA_WIDTH-1:0]  s_axi_RDATA,
	output	logic [                 1:0]  s_axi_RRESP
);

	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;

	//-----------------------------------------------------------------------
	// Error out all Writes
	if(1) begin : blkKillWrites
		logic  WABusy = 0;
		logic  WDBusy = 0;
		uwire  clr = rst || (WABusy && WDBusy && s_axi_BREADY);
		always_ff @(posedge clk) begin : blockName
			if(clr) begin
				WABusy <= 0;
				WDBusy <= 0;
			end
			else begin
				WABusy <= WABusy || s_axi_AWVALID;
				WDBusy <= WDBusy || s_axi_WVALID;
			end
		end
		assign	s_axi_AWREADY = !WABusy;
		assign	s_axi_WREADY  = !WDBusy;
		assign	s_axi_BVALID  = WABusy && WDBusy;
		assign	s_axi_BRESP   = '1; // DECERR

	end : blkKillWrites

	//-----------------------------------------------------------------------
	// Answer Reads
	if(1) begin : blkRead
		logic                         RValid =  0;
		logic [S_AXI_DATA_WIDTH-1:0]  RData;//  = 'x;
		logic [                 1:0]  RResp;//  = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				RValid <=  0;
				RData  <= 'x;
				RResp  <= 'x;
			end
			else if(s_axi_ARREADY) begin
				RValid <= s_axi_ARVALID;
				if(s_axi_ARADDR < N) begin
					RData  <= DATA[s_axi_ARADDR[$left(s_axi_ARADDR):2]];
					RResp  <= '0; // OKAY
				end
				else begin
					RData  <= 'x;
					RResp  <= '1; // DECERR
				end
			end
		end
		assign	s_axi_ARREADY = !RValid || s_axi_RREADY;
		assign	s_axi_RVALID  = RValid;
		assign	s_axi_RDATA   = RData;
		assign	s_axi_RRESP   = RResp;

	end : blkRead

endmodule : axi_info

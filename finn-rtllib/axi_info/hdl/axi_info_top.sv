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
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 *******************************************************************************/
module axi_info_top #(
	bit [31:0]  SIG_CUSTOMER,
	bit [31:0]  SIG_APPLICATION,
	bit [31:0]  VERSION,
	bit [31:0]  CHECKSUM_COUNT
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic        s_axi_AWVALID,
	output	logic        s_axi_AWREADY,
	input	logic [4:0]  s_axi_AWADDR,

	input	logic         s_axi_WVALID,
	output	logic         s_axi_WREADY,
	input	logic [31:0]  s_axi_WDATA,
	input	logic [ 3:0]  s_axi_WSTRB,

	output	logic        s_axi_BVALID,
	input	logic        s_axi_BREADY,
	output	logic [1:0]  s_axi_BRESP,

	// Reading
	input	logic        s_axi_ARVALID,
	output	logic        s_axi_ARREADY,
	input	logic [4:0]  s_axi_ARADDR,

	output	logic         s_axi_RVALID,
	input	logic         s_axi_RREADY,
	output	logic [31:0]  s_axi_RDATA,
	output	logic [ 1:0]  s_axi_RRESP
);

	axi_info #(
		.N(6),
		.S_AXI_DATA_WIDTH(32),
		.DATA('{
			32'h4649_4E4E,
			SIG_CUSTOMER,
			SIG_CUSTOMER,
			VERSION,
			32'h0,
			CHECKSUM_COUNT
		})
	)(
		//- Global Control ------------------
		.ap_clk, .ap_rst_n,

		//- AXI Lite ------------------------
		// Writing
		.s_axi_AWVALID,	.s_axi_AWREADY,	.s_axi_AWADDR,
		.s_axi_WVALID,	.s_axi_WREADY,	.s_axi_WDATA,	.s_axi_WSTRB,
		.s_axi_BVALID,	.s_axi_BREADY,	.s_axi_BRESP,
		// Reading
		.s_axi_ARVALID,	.s_axi_ARREADY,	.s_axi_ARADDR,
		.s_axi_RVALID,	.s_axi_RREADY,	.s_axi_RDATA,	.s_axi_RRESP
	);

endmodule : axi_info_top

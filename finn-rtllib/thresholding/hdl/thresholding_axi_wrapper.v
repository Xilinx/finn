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
 * @brief	Verilog wrapper for IP packaging.
 */

module thresholding_axi_wrapper #(
	parameter  N =  4,	// output precision
	parameter  K = 16,	// input/threshold precision
	parameter  C =  1,	// Channels
	parameter  PE = 1,	// Processing Parallelism, requires C = k*PE

	parameter  SIGNED = 1,	// signed inputs
	parameter  FPARG  = 0,	// floating-point inputs: [sign] | exponent | mantissa
	parameter  BIAS   = 0,	// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	parameter  CF = C/PE,	// Channel Fold
	parameter  ADDR_BITS = $clog2(CF) + $clog2(PE) + N + 2,
	parameter  O_BITS = $clog2(2**N+BIAS)
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:s_axis:m_axis, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input                  s_axilite_AWVALID,
	output                 s_axilite_AWREADY,
	input [ADDR_BITS-1:0]  s_axilite_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input         s_axilite_WVALID,
	output        s_axilite_WREADY,
	input [31:0]  s_axilite_WDATA,
	input [ 3:0]  s_axilite_WSTRB,

	output        s_axilite_BVALID,
	input         s_axilite_BREADY,
	output [1:0]  s_axilite_BRESP,

	// Reading
	input                  s_axilite_ARVALID,
	output                 s_axilite_ARREADY,
	input [ADDR_BITS-1:0]  s_axilite_ARADDR,

	output         s_axilite_RVALID,
	input          s_axilite_RREADY,
	output [31:0]  s_axilite_RDATA,
	output [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output  s_axis_tready,
	input   s_axis_tvalid,
	input [((PE*K+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input   m_axis_tready,
	output  m_axis_tvalid,
	output [((PE*O_BITS+7)/8)*8-1:0]  m_axis_tdata
);

	thresholding_axi #(
		.N(N), .K(K), .C(C), .PE(PE),
		.SIGNED(SIGNED),
		.FPARG(FPARG),
		.BIAS(BIAS)
	) core (
		.ap_clk, .ap_rst_n,

		.s_axilite_AWVALID, .s_axilite_AWREADY, .s_axilite_AWADDR,
		.s_axilite_WVALID, .s_axilite_WREADY, .s_axilite_WDATA, .s_axilite_WSTRB,
		.s_axilite_BVALID, .s_axilite_BREADY, .s_axilite_BRESP,

		.s_axilite_ARVALID, .s_axilite_ARREADY, .s_axilite_ARADDR,
		.s_axilite_RVALID, .s_axilite_RREADY, .s_axilite_RDATA, .s_axilite_RRESP,

		.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
		.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
	);

endmodule : thresholding_axi_wrapper

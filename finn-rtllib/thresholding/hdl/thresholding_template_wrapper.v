/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
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
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Verilog wrapper for IP packaging.
 */

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter  WI = $WI$,	// input precision
	parameter  WT = $WT$,	// threshold precision
	parameter  N = $N$,		// number of thresholds
	parameter  C = $C$,	// Channels
	parameter  PE = $PE$,	// Processing Parallelism, requires C = k*PE

	parameter  SIGNED = $SIGNED$,	// signed inputs
	parameter  FPARG  = $FPARG$,	// floating-point inputs: [sign] | exponent | mantissa
	parameter  BIAS   = $BIAS$,		// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	parameter  SETS = 1,  // Number of independent threshold sets

	parameter  THRESHOLDS_PATH = $THRESHOLDS_PATH$,	// Directory with initial threshold data
	parameter  USE_AXILITE = $USE_AXILITE$,	// Implement AXI-Lite for threshold read/write

	// Force Use of On-Chip Memory Blocks
	parameter  DEPTH_TRIGGER_URAM = $DEPTH_TRIGGER_URAM$,	// if non-zero, local mems of this depth or more go into URAM (prio)
	parameter  DEPTH_TRIGGER_BRAM = $DEPTH_TRIGGER_BRAM$,	// if non-zero, local mems of this depth or more go into BRAM
	parameter  DEEP_PIPELINE = $DEEP_PIPELINE$,	// [bit] extra pipeline stages for easier timing closure

	parameter  O_BITS = $O_BITS$
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input   s_axilite_AWVALID,
	output  s_axilite_AWREADY,
	input [$clog2(SETS) + $clog2(C/PE) + $clog2(PE) + $clog2(N) + 1:0]  s_axilite_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input         s_axilite_WVALID,
	output        s_axilite_WREADY,
	input [31:0]  s_axilite_WDATA,
	input [ 3:0]  s_axilite_WSTRB,

	output        s_axilite_BVALID,
	input         s_axilite_BREADY,
	output [1:0]  s_axilite_BRESP,

	// Reading
	input   s_axilite_ARVALID,
	output  s_axilite_ARREADY,
	input [$clog2(C/PE) + $clog2(PE) + $clog2(N) + 1:0]  s_axilite_ARADDR,

	output         s_axilite_RVALID,
	input          s_axilite_RREADY,
	output [31:0]  s_axilite_RDATA,
	output [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Set Selection ------
	// (ignored for SETS < 2)
	output  in1_V_TREADY,
	input   in1_V_TVALID,
	input [(((SETS > 2? $clog2(SETS) : 1)+7)/8)*8-1:0]  in1_V_TDATA,

	//- AXI Stream - Input --------------
	output  in0_V_TREADY,
	input   in0_V_TVALID,
	input [((PE*WI+7)/8)*8-1:0]  in0_V_TDATA,

	//- AXI Stream - Output -------------
	input   out0_V_TREADY,
	output  out0_V_TVALID,
	output [((PE*O_BITS+7)/8)*8-1:0]  out0_V_TDATA
);

	thresholding_axi #(
		.N(N), .WI(WI), .WT(WT), .C(C), .PE(PE),
		.SIGNED(SIGNED),
		.FPARG(FPARG),
		.BIAS(BIAS),
		.SETS(SETS),
		.THRESHOLDS_PATH(THRESHOLDS_PATH),
		.USE_AXILITE(USE_AXILITE),
		.DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM),
		.DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
		.DEEP_PIPELINE(DEEP_PIPELINE)
	) core (
		.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),

		.s_axilite_AWVALID(s_axilite_AWVALID), .s_axilite_AWREADY(s_axilite_AWREADY), .s_axilite_AWADDR(s_axilite_AWADDR),
		.s_axilite_WVALID(s_axilite_WVALID), .s_axilite_WREADY(s_axilite_WREADY), .s_axilite_WDATA(s_axilite_WDATA), .s_axilite_WSTRB(s_axilite_WSTRB),
		.s_axilite_BVALID(s_axilite_BVALID), .s_axilite_BREADY(s_axilite_BREADY), .s_axilite_BRESP(s_axilite_BRESP),

		.s_axilite_ARVALID(s_axilite_ARVALID), .s_axilite_ARREADY(s_axilite_ARREADY), .s_axilite_ARADDR(s_axilite_ARADDR),
		.s_axilite_RVALID(s_axilite_RVALID), .s_axilite_RREADY(s_axilite_RREADY), .s_axilite_RDATA(s_axilite_RDATA), .s_axilite_RRESP(s_axilite_RRESP),

		.s_axis_set_tready(in1_V_TREADY), .s_axis_set_tvalid(in1_V_TVALID), .s_axis_set_tdata(in1_V_TDATA),
		.s_axis_tready(in0_V_TREADY), .s_axis_tvalid(in0_V_TVALID), .s_axis_tdata(in0_V_TDATA),
		.m_axis_tready(out0_V_TREADY), .m_axis_tvalid(out0_V_TVALID), .m_axis_tdata(out0_V_TDATA)
	);

endmodule // $MODULE_NAME_AXI_WRAPPER$

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
 * @brief	Verilog AXI-lite wrapper for dynamic MVU.
 *****************************************************************************/

module $MODULE_NAME$_dynamic_load_wrapper #(
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
	parameter	MW = $MW$,
	parameter	MH = $MH$,

	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
	parameter   N_REPS = $N_REPS$,

	// Safely deducible parameters
	parameter	INPUT_STREAM_WIDTH_BA = (PE*WEIGHT_WIDTH+7)/8 * 8,
	parameter 	OUTPUT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	//(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	//input	ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Input stream
	input	[INPUT_STREAM_WIDTH_BA-1:0]  s_axis_0_TDATA,
	input	s_axis_0_TVALID,
	output	s_axis_0_TREADY,
	// Output Stream
	output	[OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_0_TDATA,
	output	m_axis_0_TVALID,
	input	m_axis_0_TREADY
);

dynamic_load #(
	.PE(PE),
	.SIMD(SIMD),
	.MW(MW),
	.MH(MH),
	.WEIGHT_WIDTH(WEIGHT_WIDTH),
	.N_REPS(N_REPS)
) inst (
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),
	.ivld(s_axis_0_TVALID),
	.irdy(s_axis_0_TREADY),
	.idat(s_axis_0_TDATA),
	.ovld(m_axis_0_TVALID),
	.ordy(m_axis_0_TREADY),
	.odat(m_axis_0_TDATA)
);

endmodule // $MODULE_NAME$_dynamic_load_wrapper

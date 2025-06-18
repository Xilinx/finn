/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
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
 * @brief	Verilog wrapper for stream tap component.
 *****************************************************************************/

module $MODULE_NAME$_stream_tap_wrapper #(
	parameter  DATA_WIDTH = $DATA_WIDTH$,
	parameter  TAP_REP = $TAP_REP$
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis_0:m_axis_0,m_axis_1 ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Input stream
	input	[DATA_WIDTH-1:0]  s_axis_0_TDATA,
	input	s_axis_0_TVALID,
	output	s_axis_0_TREADY,
	// Output Stream
	output	[DATA_WIDTH-1:0]  m_axis_0_TDATA,
	output	m_axis_0_TVALID,
	input	m_axis_0_TREADY

	// Tap Stream
	output	[DATA_WIDTH-1:0]  m_axis_1_TDATA,
	output	m_axis_1_TVALID,
	input	m_axis_1_TREADY
);

	stream_tap #(
		.DATA_WIDTH(DATA_WIDTH),
		.TAP_REP(TAP_REP),
	) inst (
		.clk(ap_clk), .rst(!ap_rst_n),
		.idat(s_axis_0_TDATA), .ivld(s_axis_0_TVALID), .irdy(s_axis_0_TREADY),
		.odat(m_axis_0_TDATA), .ovld(m_axis_0_TVALID), .ordy(m_axis_0_TREADY),
		.tdat(m_axis_1_TDATA), .tvld(m_axis_1_TVALID), .trdy(m_axis_1_TREADY)
	);

endmodule // $MODULE_NAME$_stream_tap_wrapper

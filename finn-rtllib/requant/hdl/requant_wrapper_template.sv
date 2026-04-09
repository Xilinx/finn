/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Template wrapper for AXI stream integer requantization.
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
	parameter int unsigned  VERSION = $VERSION$,  // DSP version
	parameter int unsigned  K = $K$,              // input precision
	parameter int unsigned  N = $N$,              // output precision
	parameter int unsigned  C = $C$,              // channels
	parameter int unsigned  PE = $PE$,            // parallel lanes, requires C = k*PE

	localparam int unsigned  CF = C/PE,
	parameter shortreal      SCALES[PE][CF] = $SCALES$,
	parameter shortreal      BIASES[PE][CF] = $BIASES$
)(
	//- Global Control ------------------
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input  ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input  ap_rst_n,

	//- AXI Stream - Input --------------
	output  in0_V_TREADY,
	input   in0_V_TVALID,
	input  [((PE*K+7)/8)*8-1:0]  in0_V_TDATA,

	//- AXI Stream - Output -------------
	input   out0_V_TREADY,
	output  out0_V_TVALID,
	output [((PE*N+7)/8)*8-1:0]  out0_V_TDATA
);

	requant_axi #(
		.VERSION(VERSION),
		.K(K), .N(N), .C(C), .PE(PE),
		.SCALES(SCALES), .BIASES(BIASES)
	) impl (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),
		.s_axis_tready(in0_V_TREADY),
		.s_axis_tvalid(in0_V_TVALID),
		.s_axis_tdata(in0_V_TDATA),
		.m_axis_tready(out0_V_TREADY),
		.m_axis_tvalid(out0_V_TVALID),
		.m_axis_tdata(out0_V_TDATA)
	);

endmodule // $TOP_MODULE_NAME$

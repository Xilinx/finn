/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief   SystemVerilog implementation module for requantization.
 *
 * This module contains the actual parameters and instantiates requant_axi.
 * It is instantiated by the Verilog stub wrapper for IP packaging.
 ***************************************************************************/

module $TOP_MODULE_NAME$_impl (
	//- Global Control ------------------
	input  ap_clk,
	input  ap_rst_n,

	//- AXI Stream - Input --------------
	output  in0_V_TREADY,
	input   in0_V_TVALID,
	input  [$IN_STREAM_WIDTH$-1:0]  in0_V_TDATA,

	//- AXI Stream - Output -------------
	input   out0_V_TREADY,
	output  out0_V_TVALID,
	output [$OUT_STREAM_WIDTH$-1:0]  out0_V_TDATA
);

	// Parameters
	localparam int unsigned  VERSION = $VERSION$;
	localparam int unsigned  K = $K$;
	localparam int unsigned  N = $N$;
	localparam int unsigned  C = $C$;
	localparam int unsigned  PE = $PE$;
	localparam int unsigned  CF = C/PE;

	// Scale and bias arrays
	localparam shortreal  SCALES[PE][CF] = $SCALES$;
	localparam shortreal  BIASES[PE][CF] = $BIASES$;

	requant_axi #(
		.VERSION(VERSION),
		.K(K), .N(N), .C(C), .PE(PE),
		.SCALES(SCALES), .BIASES(BIASES)
	) core (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),
		.s_axis_tready(in0_V_TREADY),
		.s_axis_tvalid(in0_V_TVALID),
		.s_axis_tdata(in0_V_TDATA),
		.m_axis_tready(out0_V_TREADY),
		.m_axis_tvalid(out0_V_TVALID),
		.m_axis_tdata(out0_V_TDATA)
	);

endmodule

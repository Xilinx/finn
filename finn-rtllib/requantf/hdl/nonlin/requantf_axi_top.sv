/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Synthesis top-level for requantf_axi.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module requantf_axi_top #(
	int unsigned  N = 8,
	int unsigned  C = 4,
	int unsigned  PE = 2,
	bit  SIGNED_OUT = 0,

	localparam int unsigned  INPUT_STREAM_WIDTH  = PE*32,
	localparam int unsigned  OUTPUT_STREAM_WIDTH = ((PE*N+7)/8)*8
)(
	input	logic  ap_clk,
	input	logic  ap_rst_n,
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [INPUT_STREAM_WIDTH-1:0]  s_axis_tdata,
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OUTPUT_STREAM_WIDTH-1:0]  m_axis_tdata
);
`default_nettype none

	requantf_axi #(
		.N(N), .C(C), .PE(PE),
		.SCALES('{ '{ 1.5, 0.75 }, '{ 2.0, 0.5 } }),
		.BIASES('{ '{ 0.1, -0.1 }, '{ 0.2, -0.2 } }),
		.SIGNED_OUT(SIGNED_OUT)
	) dut (
		.ap_clk, .ap_rst_n,
		.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
		.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
	);

`default_nettype wire
endmodule : requantf_axi_top

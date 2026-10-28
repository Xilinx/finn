/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Synthesis top-level for requantf_axi_decoupled.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module requantf_axi_decoupled_top #(
	int unsigned  N = 8,
	int unsigned  C = 6,
	int unsigned  PE = 2,
	bit  SIGNED_OUT = 0,

	localparam int unsigned  INPUT_STREAM_WIDTH  = PE*32,
	localparam int unsigned  OUTPUT_STREAM_WIDTH = ((PE*N+7)/8)*8,
	localparam int unsigned  PARAMS_STREAM_WIDTH = PE*64
)(
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [INPUT_STREAM_WIDTH-1:0]  s_axis_tdata,

	output	logic  s_params_tready,
	input	logic  s_params_tvalid,
	input	logic [PARAMS_STREAM_WIDTH-1:0]  s_params_tdata,

	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OUTPUT_STREAM_WIDTH-1:0]  m_axis_tdata
);
`default_nettype none

	requantf_axi_decoupled #(
		.N(N), .C(C), .PE(PE),
		.SIGNED_OUT(SIGNED_OUT)
	) dut (
		.ap_clk, .ap_rst_n,
		.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
		.s_params_tready, .s_params_tvalid, .s_params_tdata,
		.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
	);

`default_nettype wire
endmodule : requantf_axi_decoupled_top

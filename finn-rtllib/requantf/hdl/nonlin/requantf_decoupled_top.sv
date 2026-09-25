/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Synthesis top-level for requantf_decoupled.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module requantf_decoupled_top #(
	int unsigned  N = 8,
	int unsigned  C = 6,
	int unsigned  PE = 2,
	bit  SIGNED_OUT = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [PE-1:0][31:0]  idat,
	input	logic  ivld,
	input	logic [PE-1:0][1:0][31:0]  pdat,

	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);
`default_nettype none

	requantf_decoupled #(
		.N(N), .C(C), .PE(PE),
		.SIGNED_OUT(SIGNED_OUT)
	) dut (
		.clk, .rst,
		.idat, .ivld,
		.pdat,
		.odat, .ovld
	);

`default_nettype wire
endmodule : requantf_decoupled_top

/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Synthesis top-level for fmaf.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module fmaf_top #(
	parameter  OP = "ADD"
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  a,
	input	logic [31:0]  b,
	input	logic [31:0]  c,
	input	logic  ivld,

	output	logic [31:0]  r,
	output	logic  ovld
);
`default_nettype none

	fmaf #(.OP(OP)) dut (
		.clk, .rst,
		.a, .b, .c, .ivld,
		.r, .ovld
	);

`default_nettype wire
endmodule : fmaf_top

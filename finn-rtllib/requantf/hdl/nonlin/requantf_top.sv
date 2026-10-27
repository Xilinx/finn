/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Synthesis top-level for requantf.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module requantf_top #(
	int unsigned  N = 8,
	int unsigned  C = 6,
	int unsigned  PE = 2,

	localparam int unsigned  CF = C/PE,
	shortreal  SCALES[PE][CF] = '{
		'{ 0.0625, 0.1250, 0.2500 },
		'{ 0.5000, 0.7500, 1.0000 }
	},
	shortreal  BIASES[PE][CF] = '{
		'{ 12.0, 8.0, 4.0 },
		'{ 0.0, -2.0, -4.0 }
	},

	bit  SIGNED_OUT = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [PE-1:0][31:0]  idat,
	input	logic  ivld,

	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);
`default_nettype none

	requantf #(
		.N(N), .C(C), .PE(PE),
		.SCALES(SCALES), .BIASES(BIASES),
		.SIGNED_OUT(SIGNED_OUT)
	) dut (
		.clk, .rst,
		.idat, .ivld,
		.odat, .ovld
	);

`default_nettype wire
endmodule : requantf_top

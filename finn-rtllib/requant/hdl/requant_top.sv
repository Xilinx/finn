/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module requant_top #(
	int unsigned  VERSION = 1,
	int unsigned  K = 12,
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
	}
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	input	logic signed [PE-1:0][K-1:0]  idat,
	input	logic  ivld,

	// Output Stream
	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);

	requant #(
		.VERSION(VERSION),
		.K(K), .N(N), .C(C), .PE(PE),
		.SCALES(SCALES), .BIASES(BIASES)
	) dut (
		.clk, .rst,
		.idat, .ivld,
		.odat, .ovld
	);

endmodule : requant_top

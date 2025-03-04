/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Synthesis validator for fifo.sv.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module fifo_top #(
	int unsigned  DEPTH = 520,  // Minimum FIFO Depth
	int unsigned  WIDTH = 18,  // Element Bitwidth
	parameter     RAM_STYLE = "auto"
)(
	input	logic  clk,
	input	logic  rst,

	input	logic  ivld,
	output	logic  irdy,
	input	T  idat,

	output	logic  ovld,
	input	logic  ordy,
	output	T  odat
);

	fifo #(.DEPTH(DEPTH), .WIDTH(WIDTH), .RAM_STYLE(RAM_STYLE)) impl (
		.clk, .rst,
		.ivld, .irdy, .idat,
		.ovld, .ordy, .odat
	);

endmodule : fifo_top

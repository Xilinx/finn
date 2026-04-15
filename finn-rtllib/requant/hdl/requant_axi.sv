/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	AXI stream wrapper for integer requantization.
 *****************************************************************************/

module requant_axi #(
	int unsigned  VERSION = 1,  // DSP Version
	int unsigned  K,  // Input Precision
	int unsigned  N,  // Output Precision

	int unsigned  C,  // Channel count
	int unsigned  PE = 1,  // parallel processing elements, requires C = k*PE

	shortreal     SCALES[PE][C/PE],
	shortreal     BIASES[PE][C/PE],

	localparam int unsigned  INPUT_STREAM_WIDTH = ((PE*K+7)/8)*8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH = ((PE*N+7)/8)*8
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [INPUT_STREAM_WIDTH-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OUTPUT_STREAM_WIDTH-1:0]  m_axis_tdata
);
	localparam int unsigned  CF = C/PE;  // Channel fold

	uwire  rst = !ap_rst_n;

	// Parameter Constraints Checking
	initial begin
		if(CF*PE != C) begin
			$error("%m: Parallelism PE=%0d does not divide channel count C=%0d.", PE, C);
			$finish;
		end
	end

	//-----------------------------------------------------------------------
	// Credit-based Input Admission
	localparam int unsigned  CREDIT = 7;
	logic signed [$clog2(CREDIT):0]  Credit = CREDIT-1; // CREDIT-1, ..., 1, 0, -1
	uwire  have_cap = !Credit[$left(Credit)];
	uwire  issue  = s_axis_tvalid && s_axis_tready;
	uwire  settle = m_axis_tvalid && m_axis_tready;
	always @(posedge ap_clk) begin
		if(rst)  Credit <= CREDIT-1;
		else     Credit <= Credit + (issue == settle? 0 : settle? 1 : -1);
	end
	assign	s_axis_tready = have_cap;

	//-----------------------------------------------------------------------
	// Free-running requant compute core
	uwire signed [PE-1:0][K-1:0]  core_idat = s_axis_tdata[0+:PE*K];
	uwire [PE-1:0][N-1:0]  core_odat;
	uwire  core_ovld;
	requant #(
		.VERSION(VERSION),
		.K(K), .N(N), .C(C), .PE(PE),
		.SCALES(SCALES), .BIASES(BIASES)
	) impl (
		.clk(ap_clk), .rst,
		.idat(core_idat), .ivld(issue),
		.odat(core_odat), .ovld(core_ovld)
	);

	//-----------------------------------------------------------------------
	// Output AXI stream queue
	uwire [PE-1:0][N-1:0]  q_odat;
	uwire  q_ovld;
	uwire  q_irdy;
	always_ff @(posedge ap_clk) begin
		assert(!core_ovld || q_irdy) else begin
			$error("%m: Overrrun of output queue.");
			$stop;
		end
	end

	queue #(
		.DATA_WIDTH(PE*N),
		.ELASTICITY(CREDIT)
	) outq (
		.clk(ap_clk), .rst,
		.idat(core_odat), .ivld(core_ovld), .irdy(q_irdy),
		.odat(q_odat), .ovld(q_ovld), .ordy(m_axis_tready)
	);

	assign	m_axis_tvalid = q_ovld;
	assign	m_axis_tdata = { {(OUTPUT_STREAM_WIDTH-PE*N){1'b0}}, q_odat };

endmodule : requant_axi

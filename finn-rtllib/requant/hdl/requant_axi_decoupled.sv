// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
/******************************************************************************
 * @brief	AXI stream wrapper for integer requantization with decoupled
 *		(streamed) scale/bias parameters.
 *
 * @description
 *	Like requant_axi.sv but the scale/bias are not embedded as module
 *	parameters. Instead two additional AXI-Stream slave ports deliver the
 *	fixed-point parameter words (one word per lane-parallel compute beat),
 *	produced by two memstreams. A compute beat is only issued when the input
 *	data and both parameter words are simultaneously available.
 *****************************************************************************/

module requant_axi_decoupled #(
	int unsigned  VERSION = 1,  // DSP Version
	int unsigned  K,  // Input Precision
	int unsigned  N,  // Output Precision

	int unsigned  C,  // Channel count
	int unsigned  PE = 1,  // parallel processing elements, requires C = k*PE

	int unsigned  TAP_MIN,  // Worst-case minimum tap across all channels
	int unsigned  TAP_MAX,  // Worst-case maximum tap across all channels

	// Derived multiplier operand widths (must match derive_MUL_WIDTHS)
	localparam int unsigned  S_WIDTH = (K <= (VERSION==3? 24 : 18))? 25 :
	                                    (VERSION==3? 24 : 18),
	localparam int unsigned  X_WIDTH = (K <= (VERSION==3? 24 : 18))? K :
	                                    ((VERSION==1? 25 : 27) < K? (VERSION==1? 25 : 27) : K),
	localparam int unsigned  BIAS_W  = S_WIDTH + X_WIDTH,
	localparam int unsigned  TAP_RANGE = TAP_MAX - TAP_MIN + 1,
	localparam int unsigned  TAP_BITS  = (TAP_RANGE > 1)? $clog2(TAP_RANGE) : 1,
	localparam int unsigned  SCALE_LANE_W = S_WIDTH + TAP_BITS,

	localparam int unsigned  INPUT_STREAM_WIDTH  = ((PE*K+7)/8)*8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH = ((PE*N+7)/8)*8,
	localparam int unsigned  SCALE_STREAM_WIDTH  = ((PE*SCALE_LANE_W+7)/8)*8,
	localparam int unsigned  BIAS_STREAM_WIDTH   = ((PE*BIAS_W+7)/8)*8
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Data Input ---------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [INPUT_STREAM_WIDTH-1:0]  s_axis_tdata,

	//- AXI Stream - Scale Param Input --
	output	logic  s_scale_tready,
	input	logic  s_scale_tvalid,
	input	logic [SCALE_STREAM_WIDTH-1:0]  s_scale_tdata,

	//- AXI Stream - Bias Param Input ---
	output	logic  s_bias_tready,
	input	logic  s_bias_tvalid,
	input	logic [BIAS_STREAM_WIDTH-1:0]  s_bias_tdata,

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

	// Synchronized join: fire only when data and both param words are present
	uwire  params_vld = s_scale_tvalid && s_bias_tvalid;
	uwire  issue  = have_cap && s_axis_tvalid && params_vld;
	uwire  settle = m_axis_tvalid && m_axis_tready;
	always @(posedge ap_clk) begin
		if(rst)  Credit <= CREDIT-1;
		else     Credit <= Credit + (issue == settle? 0 : settle? 1 : -1);
	end
	assign	s_axis_tready  = issue;
	assign	s_scale_tready = issue;
	assign	s_bias_tready  = issue;

	//-----------------------------------------------------------------------
	// Free-running decoupled requant compute core
	uwire signed [PE-1:0][K-1:0]  core_idat = s_axis_tdata[0+:PE*K];
	uwire [PE-1:0][SCALE_LANE_W-1:0]  core_sdat = s_scale_tdata[0+:PE*SCALE_LANE_W];
	uwire [PE-1:0][BIAS_W-1:0]  core_bdat = s_bias_tdata[0+:PE*BIAS_W];
	uwire [PE-1:0][N-1:0]  core_odat;
	uwire  core_ovld;
	requant_decoupled #(
		.VERSION(VERSION),
		.K(K), .N(N), .C(C), .PE(PE),
		.TAP_MIN(TAP_MIN), .TAP_MAX(TAP_MAX)
	) impl (
		.clk(ap_clk), .rst,
		.idat(core_idat), .ivld(issue),
		.sdat(core_sdat), .bdat(core_bdat),
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

endmodule : requant_axi_decoupled

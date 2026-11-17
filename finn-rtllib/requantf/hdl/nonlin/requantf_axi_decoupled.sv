/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	AXI stream wrapper for FP32 requantization with decoupled
 *		(streamed) parameters.
 *
 * @description
 *	Like requantf_axi.sv but the scale/bias are not embedded as module
 *	parameters. Instead a single additional AXI-Stream slave port delivers
 *	the FP32 parameter pairs (one pair per lane-parallel compute beat),
 *	produced by a memstream. A compute beat is only issued when the input
 *	data and the parameter word are simultaneously available.
 ***************************************************************************/

module requantf_axi_decoupled #(
	int unsigned  N,  // Output Precision
	int unsigned  C,  // Channel count
	int unsigned  PE = 1,  // parallel processing elements, requires C = k*PE

	bit  SIGNED_OUT = 0,
	bit  FORCE_BEHAVIORAL = 0,

	localparam int unsigned  INPUT_STREAM_WIDTH  = PE*32,
	localparam int unsigned  OUTPUT_STREAM_WIDTH = ((PE*N+7)/8)*8,
	localparam int unsigned  PARAMS_STREAM_WIDTH = PE*64
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Data Input ---------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [INPUT_STREAM_WIDTH-1:0]  s_axis_tdata,

	//- AXI Stream - Params Input -------
	output	logic  s_params_tready,
	input	logic  s_params_tvalid,
	input	logic [PARAMS_STREAM_WIDTH-1:0]  s_params_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OUTPUT_STREAM_WIDTH-1:0]  m_axis_tdata
);
`default_nettype none
	localparam int unsigned  CF = C/PE;

	uwire  rst = !ap_rst_n;

	initial begin
		if(CF*PE != C) begin
			$error("%m: Parallelism PE=%0d does not divide channel count C=%0d.", PE, C);
			$finish;
		end
	end

	//-----------------------------------------------------------------------
	// Credit-based Input Admission
	localparam int unsigned  CREDIT = 7;
	logic signed [$clog2(CREDIT):0]  Credit = CREDIT-1;
	uwire  have_cap = !Credit[$left(Credit)];

	// Synchronized join: fire only when data and params are present
	uwire  issue  = have_cap && s_axis_tvalid && s_params_tvalid;
	uwire  settle = m_axis_tvalid && m_axis_tready;
	always_ff @(posedge ap_clk) begin
		if(rst)  Credit <= CREDIT-1;
		else     Credit <= Credit + (issue == settle? 0 : settle? 1 : -1);
	end
	assign	s_axis_tready   = issue;
	assign	s_params_tready = issue;

	//-----------------------------------------------------------------------
	// Free-running decoupled requantf compute core
	uwire [PE-1:0][31:0]  core_idat = s_axis_tdata[0+:PE*32];
	uwire [PE-1:0][1:0][31:0]  core_pdat = s_params_tdata[0+:PE*64];
	uwire [PE-1:0][N-1:0]  core_odat;
	uwire  core_ovld;
	requantf_decoupled #(
		.N(N), .C(C), .PE(PE),
		.SIGNED_OUT(SIGNED_OUT),
		.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
	) impl (
		.clk(ap_clk), .rst,
		.idat(core_idat), .ivld(issue),
		.pdat(core_pdat),
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

	fifo #(.DATA_WIDTH(PE*N), .DEPTH(CREDIT)) outq (
		.clk(ap_clk), .rst,
		.idat(core_odat), .ivld(core_ovld), .irdy(q_irdy),
		.odat(q_odat), .ovld(q_ovld), .ordy(m_axis_tready)
	);

	assign	m_axis_tvalid = q_ovld;
	assign	m_axis_tdata = { {(OUTPUT_STREAM_WIDTH-PE*N){1'b0}}, q_odat };

`default_nettype wire
endmodule : requantf_axi_decoupled

// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
module mmau #(
	bit IS_MVU,
	parameter COMPUTE_CORE,
	int unsigned MW,
	int unsigned MH,
	int unsigned PE,
	int unsigned SIMD,

	int unsigned ACTIVATION_WIDTH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,
	bit NARROW_WEIGHTS     = 1,

	int unsigned SEGMENTLEN = 0,
	bit SIGNED_ACTIVATIONS = 1,
	bit PUMPED_COMPUTE = 0,
	bit FORCE_BEHAVIORAL = 0,
	bit M_REG_LUT = 1,

	// MMAU specific
	int unsigned TH,
	int unsigned IO_REORDER = 1,

	// Safely deducible parameters
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE*ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN = SIMD % 2
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	// synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_b_tdata,
	input	logic  s_axis_b_tvalid,
	output	logic  s_axis_b_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_a_tdata,
	input	logic  s_axis_a_tvalid,
	output	logic  s_axis_a_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_c_tdata,
	output	logic  m_axis_c_tvalid,
	input	logic  m_axis_c_tready
);

	//-------------------- Parameter sanity checks --------------------\\
	initial begin
		if (MW % SIMD != 0) begin
			$error("Matrix width (%0d) is not a multiple of SIMD (%0d).", MW, SIMD);
			$finish;
		end
		if (MH % PE != 0) begin
			$error("Matrix height (%0d) is not a multiple of PE (%0d).", MH, PE);
			$finish;
		end
		if (WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if (ACTIVATION_WIDTH > 8) begin
			if (!(SIGNED_ACTIVATIONS == 1 && ACTIVATION_WIDTH == 9 && COMPUTE_CORE == "mvu_vvu_8sx9_dsp58")) begin
				$error("Activation width of %0d-bits exceeds maximum of 9-bits for signed numbers on DSP48", ACTIVATION_WIDTH);
				$finish;
			end
		end
		if (COMPUTE_CORE == "mvu_vvu_8sx9_dsp58") begin
			if (SEGMENTLEN == 0) begin
				$warning("Segment length of %0d defaults to chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
			end
			if (SEGMENTLEN > (SIMD+2)/3) begin
				$error("Segment length of %0d exceeds chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
				$finish;
			end
		end
		if (!IS_MVU) begin
			if (COMPUTE_CORE != "mvu_vvu_8sx9_dsp58" && COMPUTE_CORE != "mvu_vvu_lut") begin
				$error("VVU only supported on DSP58 or LUT-based implementation");
				$finish;
			end
		end
	end

	uwire  clk = ap_clk;
	uwire  clk2x = ap_clk2x;
	uwire  rst = !ap_rst_n;

	//- Replay to Accommodate Neuron Fold -----------------------------------
	//-----------------------------------------------------------------------
	typedef logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_a_t;
	uwire mvu_a_t amvau;
	uwire alast;
	uwire avld;
	uwire ardy;

	if(IO_REORDER == 1) begin
		replay_buff_tile #(.SF(TH), .NF(MH/SIMD), .SIMD(SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .N_RPLYS(MW/PE)) activation_replay (
			.clk, .rst,
			.ivld(s_axis_a_tvalid), .irdy(s_axis_a_tready), .idat(mvu_a_t'(s_axis_a_tdata)),
			.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast)
		);
	end
	else begin
		replay_buff_vector #(.LEN(TH*(MH/SIMD)), .REP(MW/PE), .W($bits(mvu_a_t))) activation_replay (
			.clk, .rst,
			.ivld(s_axis_a_tvalid), .irdy(s_axis_a_tready), .idat(mvu_a_t'(s_axis_a_tdata)),
			.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast), .ofin(afin)
		);
	end

	//- Weights widen  ------------------------------------------------------
	//-----------------------------------------------------------------------
	typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  mvu_w_t;
	uwire mvu_w_t mvu_w;
	uwire wgt_val;
	uwire wgt_rdy;

	assign wgt_val = s_axis_b_tvalid;
	assign s_axis_b_tready = wgt_rdy;
	assign mvu_w = mvu_w_t'(s_axis_b_tdata);

	//- Flow Control Bracket around Compute Core ----------------------------
	//-----------------------------------------------------------------------
	uwire en;
	uwire istb = avld && wgt_val;
	assign ardy = en && wgt_val;
	assign wgt_rdy = en && avld;

	//- MMAU ----------------------------------------------------------------
	//-----------------------------------------------------------------------
	typedef logic [PE-1:0][SIMD-1:0][2*ACTIVATION_WIDTH-1:0] mul_t;
	typedef logic [PE-1:0][ACCU_WIDTH-1:0] add_t;
	typedef logic [PE-1:0][ACCU_WIDTH-1:0] acc_t;
	uwire mul_t p_mul;
	logic p_mul_val;
	logic p_mul_last;
	uwire add_t p_add;
	logic p_add_val;
	logic p_add_last;
	logic inc_acc;
	uwire acc_t acc_out;
	uwire acc_t odat;
	logic ovld;

	// MUL-ADD stage
`define MULADD
`ifdef MULADD
	mul_add_stage #(
		.SIMD(SIMD),
		.PE(PE),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH),
		.COMPUTE_CORE(COMPUTE_CORE)
	) inst_mul_add_stage (
		.clk(clk),
		.clk2x(clk2x),
		.rst(rst),
		.en(en),

		.amvau_i(amvau),
		.mvu_w(mvu_w),
		.ival(istb),
		.ilast(alast),

		.i_acc(acc_out),
		.inc_acc(inc_acc),

		.odat(p_add),
		.oval(p_add_val),
		.olast(p_add_last)
	);
`else
    // MUL stage
	mul_stage #(
		.SIMD(SIMD),
		.PE(PE),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.WEIGHT_WIDTH(WEIGHT_WIDTH)
	) inst_mul_stage (
		.clk(clk),
		.rst(rst),
		.en(en),
		
		.a(amvau),
		.w(mvu_w),
		.ival(istb),
		.ilast(alast),
		.odat(p_mul),
		.oval(p_mul_val),
		.olast(p_mul_last)
	);

	// ADD stage
	add_stage #(
		.SIMD(SIMD),
		.PE(PE),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH)	
	) inst_add_stage (
		.clk(clk),
		.rst(rst),
		.en(en),
		
		.idat_mul(p_mul),
		.ival(p_mul_val),
		.ilast(p_mul_last),

		.i_acc(acc_out),
		.inc_acc(inc_acc),

		.odat(p_add),
		.oval(p_add_val),
		.olast(p_add_last)
	);
`endif
    
	// ACC stage
	acc_stage #(
		.PE(PE),
		.ACCU_WIDTH(ACCU_WIDTH),
		.TH(TH)	
	) inst_acc_stage (
		.clk(clk),
		.rst(rst),
		.en(en),
		
		.idat(p_add),
		.ival(p_add_val),
		.ilast(p_add_last),

		.o_acc(acc_out),
		.inc_acc(inc_acc),

		.odat(odat),
		.oval(ovld)
	);

	//-------------------- Output register slice --------------------\\
	// Make `en`computation independent from external inputs.
	// Drive all outputs from registers.
	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  axis_c_tdata_int;
	logic  axis_c_tvalid_int;
	logic  axis_c_tready_int;

	struct packed {
		logic rdy;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  A = '{ rdy: 1, default: 'x };	// side-step register used when encountering backpressure
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  B = '{ vld: 0, default: 'x };	// ultimate output register

	assign	en = A.rdy;
	uwire  b_load = !B.vld || axis_c_tready_int;

	always_ff @(posedge clk) begin
		if(rst) begin
			A <= '{ rdy: 1, default: 'x };
			B <= '{ vld: 0, default: 'x };
		end
		else begin
			if(A.rdy)  A.dat <= odat;
			A.rdy <= (A.rdy && !ovld) || b_load;

			if(b_load) begin
				B <= '{
					vld: ovld || !A.rdy,
					dat: A.rdy? odat : A.dat
				};
			end
		end
	end
	assign	axis_c_tvalid_int = B.vld;
	// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
	// These extra bits should never be used. Why not 'x them out?
	assign	axis_c_tdata_int  = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){B.dat[PE-1][ACCU_WIDTH-1]}}, B.dat};

	// ---------------------------------------------------------------------------
	// Reorder stage
	// ---------------------------------------------------------------------------
	if(IO_REORDER == 1) begin
		shuffle_out #(
			.SF(TH),
			.NF(MW/PE),
			.PE(PE),
			.ACTIVATION_WIDTH(ACCU_WIDTH)
		) inst_shuffle_out (
			.clk(ap_clk),
			.rst(~ap_rst_n),
			.ivld(axis_c_tvalid_int),
			.irdy(axis_c_tready_int),
			.idat(axis_c_tdata_int),
			.ovld(m_axis_c_tvalid),
			.ordy(m_axis_c_tready),
			.odat(m_axis_c_tdata)
		);
	end
	else begin
		assign m_axis_c_tvalid = axis_c_tvalid_int;
		assign m_axis_c_tdata  = axis_c_tdata_int;
		assign axis_c_tready_int = m_axis_c_tready;
	end
	

endmodule : mmau

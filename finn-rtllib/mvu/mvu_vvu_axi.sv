/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Matrix Vector Unit (MVU) & Vector Vector Unit (VVU) AXI-lite interface wrapper.
 * @details
 *	 The following compute cores are supported:
 *   - 4-bit MVU on DSP48 & DSP58 achieving 4 MACs/DSP, 
 *     (4,8]-bit MVU on DSP48 achieving 2 MACs/DSP,
 *     [4,9]-bit MVU and VVU on DSP58 achieving 3 MACs/DSP,
 *     'unconstrained' LUT-based MVU and VVU.
 *  Folding hints:
 *	 - PE scaling should divide MH.
 *   - SIMD scaling should divide MW.
 *	 - Otherwise, keep SIMD and PE somewhat balanced. SIMD scaling tends to
 *	   impact critical paths more than PE scaling. PE scaling implies a
 *	   bigger fanout on the input activations.
 *	 - Full unfolding along MH (PE=MH) results in no replay buffer instantiated
 *****************************************************************************/

module mvu_vvu_axi #(
	bit IS_MVU,
	parameter COMPUTE_CORE,
	int unsigned MW,
	int unsigned MH,
	int unsigned PE,
	int unsigned SIMD,
	int unsigned ACTIVATION_WIDTH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,
	bit SIGNED_ACTIVATIONS = 0,
	int unsigned SEGMENTLEN = 0,
	bit FORCE_BEHAVIORAL = 0,
	bit M_REG_LUT = 1,

	// Safely deducible parameters
	localparam int unsigned  WEIGHT_STREAM_WIDTH	= PE * SIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA	= (WEIGHT_STREAM_WIDTH + 7) / 8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH	= SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA	= (INPUT_STREAM_WIDTH + 7) / 8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH	= PE * ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA	= (OUTPUT_STREAM_WIDTH + 7) / 8 * 8,
	localparam int unsigned  SF = MW / SIMD,
	localparam int unsigned  NF = MH / PE
)
(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_weights_tdata,
	input	logic  s_axis_weights_tvalid,
	output	logic  s_axis_weights_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata,
	input	logic  s_axis_input_tvalid,
	output	logic  s_axis_input_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready
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
	end

	uwire clk = ap_clk;
	uwire rst = !ap_rst_n;

	//- Replay to Accommodate Neuron Fold -----------------------------------
	typedef logic [PE*SIMD-1:0][ACTIVATION_WIDTH-1:0] mvu_flatin_t;
	uwire mvu_flatin_t amvau;
	uwire alast;
	uwire afin;
	uwire avld;
	uwire ardy;

	replay_buffer #(.LEN(SF), .REP(NF), .W($bits(mvu_flatin_t))) activation_replay (
	.clk, .rst,
	.ivld(s_axis_input_tvalid), .irdy(s_axis_input_tready), .idat(mvu_flatin_t'(s_axis_input_tdata)),
	.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast), .ofin(afin)
	);

	//- Unflatten inputs into structured matrices ---------------------------
	typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH    -1:0]  mvu_w_t;
	typedef logic         [SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_a_t;

	uwire  mvu_w_t  mvu_w = s_axis_weights_tdata;
	uwire  mvu_a_t  mvu_a = amvau;

	//- Flow Control Bracket around Compute Core ----------------------------
	uwire en;
	uwire istb = avld && s_axis_weights_tvalid;
	assign ardy = en && s_axis_weights_tvalid;
	assign s_axis_weights_tready = en && avld;

	//- Instantiate compute core ----------------------------
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
	uwire dsp_vld;
	uwire dsp_p_t  dsp_p;

	uwire dsp_clk = ap_clk;
	uwire dsp_en = en;
	uwire dsp_last = alast && avld;
	uwire dsp_zero = !istb;
	uwire mvu_w_t dsp_w = mvu_w;
	uwire mvu_a_t dsp_a = mvu_a;
	uwire ovld = dsp_vld;
	uwire dsp_p_t  odat = dsp_p;

	case(COMPUTE_CORE)
	"mvu_vvu_8sx9_dsp58":
		mvu_vvu_8sx9_dsp58 #(.IS_MVU(IS_MVU), .PE(PE), .SIMD(SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN),
		.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	"mvu_4sx4u":
		mvu_4sx4u #(.PE(PE), .SIMD(SIMD), .ACCU_WIDTH(ACCU_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	"mvu_8sx8u_dsp48":
		mvu_8sx8u_dsp48 #(.PE(PE), .SIMD(SIMD), .ACCU_WIDTH(ACCU_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
		.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	default: initial begin
		$error("Unrecognized COMPUTE_CORE '%s'", COMPUTE_CORE);
		$finish;
	end
	endcase

//-------------------- Output register slice --------------------\\
	// Make `en`computation independent from external inputs.
	// Drive all outputs from registers.
	struct packed {
		logic rdy;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  A = '{ rdy: 1, default: 'x };	// side-step register used when encountering backpressure
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  B = '{ vld: 0, default: 'x };	// ultimate output register

	assign	en = A.rdy;
	uwire  b_load = !B.vld || m_axis_output_tready;

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
	assign	m_axis_output_tvalid = B.vld;
	// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
	// These extra bits should never be used. Why not 'x them out?
	assign	m_axis_output_tdata  = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){B.dat[PE-1][ACCU_WIDTH-1]}}, B.dat};


endmodule : mvu_vvu_axi

/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
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
 * @brief	Matrix Vector Unit with Tiling (MVU-Tiled) AXI-lite interface wrapper.
 * @details
 *	 The following compute cores are supported:
 *   - [4,9]-bit MVU on DSP58 achieving 3 MACs/DSP,
 *  Folding hints:
 *	 - PE scaling should divide MH.
 *   - SIMD scaling should divide MW.
 *   - TH scaling should divide MH_OUTER
 *   - WSIMD * TH <= PE * SIMD
 *	 - Otherwise, keep SIMD and PE somewhat balanced. SIMD scaling tends to
 *	   impact critical paths more than PE scaling. PE scaling implies a
 *	   bigger fanout on the input activations.
 *	 - Full unfolding along MH (PE=MH) results in no replay buffer instantiated
 *****************************************************************************/

module mvu_tiled_axi #(
	int unsigned PE,
	int unsigned SIMD,

	int unsigned WEIGHT_WIDTH,
	int unsigned ACTIVATION_WIDTH,
	int unsigned ACCU_WIDTH,

	int unsigned MW,
	int unsigned MH,
	int unsigned TH,

	int unsigned IN_TILED = 0,
	int unsigned OUT_TILED = 0,

	bit NARROW_WEIGHTS   = 0,	// Weights in (-W:W) rather than [-W:W) with W = 2**(WEIGHT_WIDTH-1)
	bit SIGNED_ACTIVATIONS = 0,
	bit PUMPED_COMPUTE = 0, // Not meaningful for SIMD < 2, which will error out.
						// Best utilization for even values.
	bit FORCE_BEHAVIORAL = 0,
	bit M_REG_LUT = 1,

	parameter COMPUTE_CORE = "mvu_vvu_8sx9_dsp58",
	int unsigned N_DCPL_STAGES = 2,

	// Safely deducible parameters
	localparam int unsigned  WSIMD = (PE * SIMD) / TH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = WSIMD * WEIGHT_WIDTH,
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
			$error("%m: Matrix width (%0d) is not a multiple of SIMD (%0d).", MW, SIMD);
			$finish;
		end
		if (MH % PE != 0) begin
			$error("%m: Matrix height (%0d) is not a multiple of PE (%0d).", MH, PE);
			$finish;
		end
		if ((PE * SIMD) % TH != 0) begin
			$error("%m: Tile (%0d) is not a multiple of TH (%0d).", (PE*SIMD), TH);
			$finish;
		end

		if (PUMPED_COMPUTE && (SIMD == 1)) begin
			$error("Clock pumping an input of SIMD=1 is not meaningful.");
			$finish;
		end
		if (WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if (ACTIVATION_WIDTH > 8) begin
			$error("Activation width of %0d-bits exceeds maximum of 8-bits", ACTIVATION_WIDTH);
			$finish;
		end
	end

	uwire  rst = !ap_rst_n;

	//- Replay to Accommodate Neuron Fold -----------------------------------
	typedef logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_flatin_t;
	uwire mvu_flatin_t amvau;
	uwire alast;
	uwire avld;
	uwire ardy;

	replay_buff_tile #(.XC(MW/SIMD), .YC(TH), .W($bits(mvu_flatin_t)), .N_REPS(MH/PE), .IO_TILED(IN_TILED)) activation_replay (
		.clk(ap_clk), .rst(rst),
		.ivld(s_axis_input_tvalid), .irdy(s_axis_input_tready), .idat(mvu_flatin_t'(s_axis_input_tdata)),
		.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast)
	);

	//- Unflatten weights ---------------------------------------------------
	typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  mvu_w_t;
	uwire  mvu_w_t wdat;
	uwire wvld;
	uwire wrdy;

	weights_buff_tile #(
		.WEIGHT_WIDTH(WEIGHT_WIDTH),
		.SIMD(SIMD), .PE(PE),
		.TH(TH), .WSIMD(WSIMD),
		.N_DCPL_STAGES(N_DCPL_STAGES)
	) inst_weights_buff_tile (
		.clk(ap_clk), .rst(rst),
		.ivld(s_axis_weights_tvalid), .irdy(s_axis_weights_tready), .idat(s_axis_weights_tdata),
		.ovld(wvld), .ordy(wrdy), .odat(wdat)
	);

	//- Flow Control Bracket around Compute Core ----------------------------
	uwire en;
	uwire istb = avld && wvld;
	assign ardy = en && wvld;
	assign wrdy = en && avld;

	//- Conditionally Pumped DSP Compute ------------------------------------
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
	uwire  ovld;
	uwire dsp_p_t  odat;
	if(1) begin : blkDsp
		localparam int unsigned  EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE ? SIMD+1 : SIMD;
		localparam int unsigned  DSP_SIMD = EFFECTIVE_SIMD/(PUMPED_COMPUTE+1);
		typedef logic [PE    -1:0][DSP_SIMD-1:0][WEIGHT_WIDTH    -1:0]  dsp_w_t;
		typedef logic [DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  dsp_a_t;

		uwire  dsp_last;
		uwire  dsp_zero;
		uwire dsp_w_t  dsp_w;
		uwire dsp_a_t  dsp_a;

		uwire  dsp_vld;
		uwire dsp_p_t  dsp_p;

		// TODO: No double-pumping in the initial implementation
		assign	dsp_en  = en;

		assign	dsp_last = alast && istb;
		assign	dsp_zero = !istb;
		assign	dsp_w = wdat;
		assign	dsp_a = amvau;

		assign	ovld = dsp_vld;
		assign	odat = dsp_p;

        //
        // Compute Unit
        //

        case(COMPUTE_CORE)
        "mvu_vvu_8sx9_dsp58": begin : core
            cu_mvau_tiled #(
                .PE(PE), .SIMD(SIMD),
                .TH(TH),
                .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
                .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS)
            ) inst_cu_mvau_tiled (
                .clk(ap_clk), .rst(rst), .en(dsp_en),
				.ivld(istb), .ilast(dsp_last), .w(dsp_w), .a(dsp_a),
				.ovld(dsp_vld), .p(dsp_p)
            );
        end
		default: initial begin
			$error("Unrecognized COMPUTE_CORE '%s'", COMPUTE_CORE);
			$finish;
		end
		endcase

	end : blkDsp

	//-------------------- Output register slice --------------------\\
	// Make `en`computation independent from external inputs.
	// Drive all outputs from registers.

	logic m_axis_int_tvalid;
	logic m_axis_int_tready;
	logic [OUTPUT_STREAM_WIDTH_BA-1:0] m_axis_int_tdata;

	struct packed {
		logic rdy;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  A = '{ rdy: 1, default: 'x };	// side-step register used when encountering backpressure
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  B = '{ vld: 0, default: 'x };	// ultimate output register

	assign	en = A.rdy;
	uwire  b_load = !B.vld || m_axis_int_tready;

	always_ff @(posedge ap_clk) begin
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
	assign	m_axis_int_tvalid = B.vld;
	// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
	// These extra bits should never be used. Why not 'x them out?
	assign	m_axis_int_tdata  = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){B.dat[PE-1][ACCU_WIDTH-1]}}, B.dat};

	//-------------------- Output reordering --------------------\\

	if(OUT_TILED == 0) begin
		reorder_out #(.W(OUTPUT_STREAM_WIDTH_BA), .XC(MH/PE), .YC(TH)) inst_reorder_out (
			.clk(ap_clk), .rst(rst),
			.ivld(m_axis_int_tvalid), .irdy(m_axis_int_tready), .idat(m_axis_int_tdata),
			.ovld(m_axis_output_tvalid), .ordy(m_axis_output_tready), .odat(m_axis_output_tdata)
		);
	end else begin
		assign m_axis_output_tvalid = m_axis_int_tvalid;
		assign m_axis_int_tready = m_axis_output_tready;
		assign m_axis_output_tdata = m_axis_int_tdata;
	end

endmodule : mvu_tiled_axi

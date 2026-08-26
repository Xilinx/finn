/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Matrix Vector Unit with Tiling (MVU-Tiled) AXI-Stream wrapper.
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
	int unsigned  PE,
	int unsigned  SIMD,

	int unsigned  WEIGHT_WIDTH,
	int unsigned  ACTIVATION_WIDTH,
	int unsigned  ACCU_WIDTH,

	int unsigned  MW,
	int unsigned  MH,
	int unsigned  TH,

	int unsigned  IN_TILED  = 0,
	int unsigned  OUT_TILED = 0,

	bit  NARROW_WEIGHTS    = 0,  // unused — reserved for future narrow-weight support
	bit  SIGNED_ACTIVATIONS = 0,
	bit  PUMPED_COMPUTE    = 0,  // Not meaningful for SIMD < 2, which will error out.
	bit  FORCE_BEHAVIORAL  = 0,  // unused — reserved for future behavioral fallback
	bit  M_REG_LUT         = 1,  // unused — reserved for future LUT-based M register

	parameter  COMPUTE_CORE = "mvu_vvu_8sx9_dsp58",
	int unsigned  N_DCPL_STAGES = 2,

	// Safely deducible parameters
	localparam int unsigned  WSIMD = (PE * SIMD) / TH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = WSIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8,
	localparam bit           SIMD_UNEVEN = SIMD % 2
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,  // synchronous, double-speed clock; only used for PUMPED_COMPUTE
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

	//=== Parameter Validation ==============================================
	initial begin
		if(MW % SIMD != 0) begin
			$error("%m: Matrix width (%0d) is not a multiple of SIMD (%0d).", MW, SIMD);
			$finish;
		end
		if(MH % PE != 0) begin
			$error("%m: Matrix height (%0d) is not a multiple of PE (%0d).", MH, PE);
			$finish;
		end
		if((PE * SIMD) % TH != 0) begin
			$error("%m: Tile (%0d) is not a multiple of TH (%0d).", (PE*SIMD), TH);
			$finish;
		end
		if(PUMPED_COMPUTE && (SIMD == 1)) begin
			$error("Clock pumping an input of SIMD=1 is not meaningful.");
			$finish;
		end
		if(WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if(ACTIVATION_WIDTH > 8) begin
			$error("Activation width of %0d-bits exceeds maximum of 8-bits", ACTIVATION_WIDTH);
			$finish;
		end
	end

	uwire  rst = !ap_rst_n;

	//=== Activation Replay =================================================
	typedef logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_flatin_t;
	uwire  mvu_flatin_t  amvau;
	uwire  alast;
	uwire  avld;
	uwire  ardy;

	localparam int unsigned  SF = MW / SIMD;
	localparam int unsigned  NF = MH / PE;

	uwire [2:0]  act_done;
	input_gen #(
		.DATA_WIDTH($bits(mvu_flatin_t)),
		.FM_SIZE(SF * TH),
		.D(3),
		.DIMS('{NF, SF, TH}),
		.COEFS('{0, 1, SF})
	) activation_replay (
		.clk(ap_clk), .rst(rst),
		.idat(mvu_flatin_t'(s_axis_input_tdata)),
		.ivld(s_axis_input_tvalid), .irdy(s_axis_input_tready),
		.odat(amvau), .ovld(avld), .olst(), .odone(act_done), .ordy(ardy)
	);
	assign	alast = act_done[1];

	//=== Weight Buffering ==================================================
	typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  mvu_w_t;
	uwire  mvu_w_t  wdat;
	uwire  wvld;
	uwire  wrdy;

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

	//=== Flow Control ======================================================
	uwire  en;
	uwire  istb = avld && wvld;
	assign  ardy = en && wvld;
	assign  wrdy = en && avld;

	//=== DSP Compute =======================================================
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
	uwire  ovld;
	uwire  dsp_p_t  odat;
	if(1) begin : blkDsp
		localparam int unsigned  EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE? SIMD+1 : SIMD;
		localparam int unsigned  DSP_SIMD = EFFECTIVE_SIMD / (PUMPED_COMPUTE+1);
		typedef logic [PE    -1:0][DSP_SIMD-1:0][WEIGHT_WIDTH    -1:0]  dsp_w_t;
		typedef logic [DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  dsp_a_t;

		uwire  dsp_last;
		uwire  dsp_w_t  dsp_w;
		uwire  dsp_a_t  dsp_a;

		uwire  dsp_vld;
		uwire  dsp_p_t  dsp_p;

		// TODO: No double-pumping in the initial implementation
		uwire  dsp_en = en;

		assign	dsp_last = alast && istb;
		assign	dsp_w = wdat;
		assign	dsp_a = amvau;

		assign	ovld = dsp_vld;
		assign	odat = dsp_p;

		case(COMPUTE_CORE)
		"mvu_vvu_8sx9_dsp58": begin : core
			cu_mvau_tiled #(
				.PE(PE), .SIMD(SIMD),
				.TH(TH),
				.WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
				.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
				.TARGET(1'b1)   // this core is DSP58-only, i.e. Versal
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

	//=== Output Register Slice =============================================
	// Make `en` computation independent from external inputs.
	// Drive all outputs from registers.

	logic  MIntVld;
	uwire  m_int_rdy;
	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  MIntDat;

	struct packed {
		logic  rdy;
		logic [PE-1:0][ACCU_WIDTH-1:0]  dat;
	}  A = '{ rdy: 1, default: 'x };  // side-step register used when encountering backpressure
	struct packed {
		logic  vld;
		logic [PE-1:0][ACCU_WIDTH-1:0]  dat;
	}  B = '{ vld: 0, default: 'x };  // ultimate output register

	assign	en = A.rdy;
	uwire  b_load = !B.vld || m_int_rdy;

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
	assign	MIntVld = B.vld;
	assign	MIntDat = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){B.dat[PE-1][ACCU_WIDTH-1]}}, B.dat };

	//=== Output Reordering =================================================

	if(OUT_TILED == 0) begin : genReorder
		input_gen #(
			.DATA_WIDTH(OUTPUT_STREAM_WIDTH_BA),
			.FM_SIZE(NF * TH),
			.D(2),
			.DIMS('{TH, NF}),
			.COEFS('{1, TH})
		) inst_reorder_out (
			.clk(ap_clk), .rst(rst),
			.idat(MIntDat),
			.ivld(MIntVld), .irdy(m_int_rdy),
			.odat(m_axis_output_tdata), .ovld(m_axis_output_tvalid),
			.olst(), .odone(), .ordy(m_axis_output_tready)
		);
	end : genReorder
	else begin : genPassthru
		assign  m_axis_output_tvalid = MIntVld;
		assign  m_int_rdy = m_axis_output_tready;
		assign  m_axis_output_tdata = MIntDat;
	end : genPassthru

endmodule : mvu_tiled_axi

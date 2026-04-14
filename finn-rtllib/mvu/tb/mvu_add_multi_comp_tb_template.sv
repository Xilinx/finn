/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for MVU with add_multi compressor-replaced adder trees.
 *		Exercises the full mvu_vvu_axi pipeline through the DSP lane path
 *		(genSoftVec in mvu.sv) where add_multi instances are replaced by
 *		generated LUT compressors via the CATCH_COMP mechanism.
 *
 * Template placeholders expanded by run_mvu_add_multi_comp_tests.sh:
 *   {mh}         - Matrix Height
 *   {mw}         - Matrix Width
 *   {pe}         - Processing Elements
 *   {simd}       - SIMD lanes
 *   {ww}         - Weight Width
 *   {aw}         - Activation Width
 *   {accu_width} - Accumulator Width
 *   {signed_act} - Signed Activations (0 or 1)
 *   {narrow}     - Narrow Weights (0 or 1)
 *   {label}      - Configuration label
 *****************************************************************************/

module mvu_add_multi_comp_{label}_tb;

	localparam int unsigned  ROUNDS = 17;

	localparam int unsigned  MH   = {mh};
	localparam int unsigned  MW   = {mw};
	localparam int unsigned  PE   = {pe};
	localparam int unsigned  SIMD = {simd};
	localparam int unsigned  WEIGHT_WIDTH     = {ww};
	localparam int unsigned  ACTIVATION_WIDTH = {aw};
	localparam int unsigned  ACCU_WIDTH       = {accu_width};
	localparam bit  SIGNED_ACTIVATIONS = {signed_act};
	localparam bit  NARROW_WEIGHTS     = {narrow};

	localparam int unsigned  SF = MW / SIMD;	// SIMD folds
	localparam int unsigned  NF = MH / PE;		// Neuron folds

	typedef logic signed [WEIGHT_WIDTH    -1:0]  weight_t;
	typedef logic        [ACTIVATION_WIDTH-1:0]  activation_t;
	typedef logic signed [ACCU_WIDTH      -1:0]  accu_t;

	//-----------------------------------------------------------------------
	// AXI Stream widths (byte-aligned)
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH;
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7) / 8 * 8;
	localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH;
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH + 7) / 8 * 8;
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH;
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7) / 8 * 8;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  ap_rst_n = 0;
	initial begin
		repeat(16) @(posedge clk);
		ap_rst_n <= 1;
	end

	//-----------------------------------------------------------------------
	// DUT — full MVU with AXI interfaces (DSP lane path with compressor trees)
	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_weights_tdata;
	logic  s_axis_weights_tvalid;
	uwire  s_axis_weights_tready;

	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata;
	logic  s_axis_input_tvalid;
	uwire  s_axis_input_tready;

	uwire [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata;
	uwire  m_axis_output_tvalid;
	logic  m_axis_output_tready;

	mvu_vvu_axi #(
		.IS_MVU(1),
		.VERSION(3),
		.MW(MW), .MH(MH),
		.PE(PE), .SIMD(SIMD),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.WEIGHT_WIDTH(WEIGHT_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH),
		.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
		.NARROW_WEIGHTS(NARROW_WEIGHTS)
	) dut (
		.ap_clk(clk),
		.ap_clk2x(clk),
		.ap_rst_n,
		.s_axis_weights_tdata,
		.s_axis_weights_tvalid,
		.s_axis_weights_tready,
		.s_axis_input_tdata,
		.s_axis_input_tvalid,
		.s_axis_input_tready,
		.m_axis_output_tdata,
		.m_axis_output_tvalid,
		.m_axis_output_tready
	);

	//-----------------------------------------------------------------------
	// Reference Model
	accu_t  ExpQ[$];

	task automatic feed_mvau(
		input weight_t     [MH-1:0][MW-1:0]  W,
		input activation_t [MW-1:0]           A
	);
		automatic accu_t [MH-1:0]  expected = '{ default: '0 };

		// Compute expected output
		for(int unsigned  h = 0; h < MH; h++) begin
			for(int unsigned  w = 0; w < MW; w++) begin
				expected[h] += $signed(W[h][w])
					* $signed({SIGNED_ACTIVATIONS && A[w][ACTIVATION_WIDTH-1], A[w]});
			end
		end

		// Push expected outputs (PE at a time, NF groups)
		for(int unsigned  nf = 0; nf < NF; nf++) begin
			for(int unsigned  pe = 0; pe < PE; pe++) begin
				ExpQ.push_back(expected[nf * PE + pe]);
			end
		end

		// Drive weight stream: NF * SF beats
		// Drive input stream:  SF beats (replayed NF times by hardware)
		fork
			// Weight feeder
			begin
				for(int unsigned  nf = 0; nf < NF; nf++) begin
					for(int unsigned  sf = 0; sf < SF; sf++) begin
						automatic logic [WEIGHT_STREAM_WIDTH-1:0]  wdata = '0;
						for(int unsigned  pe = 0; pe < PE; pe++) begin
							for(int unsigned  simd = 0; simd < SIMD; simd++) begin
								automatic int unsigned  h = nf * PE + pe;
								automatic int unsigned  w = sf * SIMD + simd;
								wdata[(pe * SIMD + simd) * WEIGHT_WIDTH +: WEIGHT_WIDTH] = W[h][w];
							end
						end
						s_axis_weights_tdata  <= WEIGHT_STREAM_WIDTH_BA'(wdata);
						s_axis_weights_tvalid <= 1;
						@(posedge clk iff s_axis_weights_tready);
					end
				end
				s_axis_weights_tvalid <= 0;
				s_axis_weights_tdata  <= 'x;
			end

			// Activation feeder
			begin
				for(int unsigned  sf = 0; sf < SF; sf++) begin
					automatic logic [INPUT_STREAM_WIDTH-1:0]  adata = '0;
					for(int unsigned  simd = 0; simd < SIMD; simd++) begin
						adata[simd * ACTIVATION_WIDTH +: ACTIVATION_WIDTH] = A[sf * SIMD + simd];
					end
					s_axis_input_tdata  <= INPUT_STREAM_WIDTH_BA'(adata);
					s_axis_input_tvalid <= 1;
					@(posedge clk iff s_axis_input_tready);
				end
				s_axis_input_tvalid <= 0;
				s_axis_input_tdata  <= 'x;
			end
		join
	endtask : feed_mvau

	//-----------------------------------------------------------------------
	// Stimulus
	initial begin
		s_axis_weights_tdata  = 'x;
		s_axis_weights_tvalid = 0;
		s_axis_input_tdata    = 'x;
		s_axis_input_tvalid   = 0;
		@(posedge clk iff ap_rst_n);
		repeat(4) @(posedge clk);

		// All zeros
		begin
			automatic weight_t     [MH-1:0][MW-1:0]  W = '0;
			automatic activation_t [MW-1:0]           A = '0;
			feed_mvau(W, A);
		end

		// All ones
		begin
			automatic weight_t     [MH-1:0][MW-1:0]  W = '1;
			automatic activation_t [MW-1:0]           A = '1;
			feed_mvau(W, A);
		end

		// Random rounds
		repeat(ROUNDS) begin
			automatic weight_t     [MH-1:0][MW-1:0]  W;
			automatic activation_t [MW-1:0]           A;
			void'(std::randomize(W, A));
			if(NARROW_WEIGHTS) begin
				foreach(W[h,w]) begin
					if(W[h][w] === weight_t'(-2**(WEIGHT_WIDTH-1)))
						W[h][w] += 1;
				end
			end
			feed_mvau(W, A);
		end

		// Wait for all outputs to drain
		repeat(100) @(posedge clk);

		assert(ExpQ.size() == 0) else begin
			$error("Missing %0d outputs.", ExpQ.size());
			$stop;
		end

		$display("Test completed successfully.");
		$finish;
	end

	//-----------------------------------------------------------------------
	// Output Checker
	int unsigned  Checks  = 0;
	int unsigned  OutBeat = 0;
	always begin
		m_axis_output_tready <= $urandom_range(0, 3) != 0;
		@(posedge clk);
	end

	always_ff @(posedge clk iff ap_rst_n) begin
		if(m_axis_output_tvalid && m_axis_output_tready) begin
			for(int unsigned  pe = 0; pe < PE; pe++) begin
				automatic accu_t  got = m_axis_output_tdata[pe * ACCU_WIDTH +: ACCU_WIDTH];
				automatic accu_t  exp;

				assert(ExpQ.size() > 0) else begin
					$error("Spurious output at beat %0d, PE %0d: %0d", OutBeat, pe, got);
					$stop;
				end

				exp = ExpQ.pop_front();
				assert(got === exp) else begin
					$error("Mismatch at beat %0d, PE %0d: got %0d, expected %0d", OutBeat, pe, got, exp);
					$stop;
				end
				Checks++;
			end
			OutBeat++;
		end
	end

	final begin
		automatic int unsigned  expected_checks = ROUNDS + 2;  // +2 directed tests
		expected_checks *= MH;
		assert(Checks == expected_checks)
			$display("Successfully performed %0d checks.", Checks);
		else
			$error("Unexpected number of checks: %0d instead of %0d.", Checks, expected_checks);
	end

endmodule : mvu_add_multi_comp_{label}_tb

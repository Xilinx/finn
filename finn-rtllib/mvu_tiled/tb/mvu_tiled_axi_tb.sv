/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for MVU-Tiled AXI wrapper module.
 * @details
 *  Adapted from mvu/tb/mvu_axi_tb.sv for the tiled architecture.
 *  Exercises mvu_tiled_axi with multiple parameter configurations in parallel.
 *
 *  Data flow under test:
 *    activations  -> input_gen (replays MH/PE times)
 *    weights      -> weights_buff_tile (collects NW=TH words, replays TH times)
 *    compute      -> cu_mvau_tiled (DSP58 INT8, 3 MACs/DSP)
 *    accumulate   -> acc_stage (pipelined add_multi tree + circular FIFO)
 *    reorder      -> input_gen (transpose tiled -> sequential NF order)
 *
 *  Weight feed order:
 *    For each neuron fold (h), for each SIMD fold (w):
 *      send TH chunks of WSIMD weights (PE*SIMD total per tile).
 *
 *  Activation feed order:
 *    For each TH tile (y), for each SIMD fold (x):
 *      send one SIMD-wide activation word.
 *    Replay buffer handles repetition across neuron folds.
 *
 *  Output order (after reorder_out):
 *    Sequential neuron folds: nf=0..MH/PE-1, one PE-wide word per fold,
 *    repeated for each input vector.
 *****************************************************************************/

module mvu_tiled_axi_tb;

	// Test Configurations
	localparam int unsigned  ROUNDS = 7;

	typedef struct {
		int unsigned  mh;
		int unsigned  mw;
		int unsigned  pe;
		int unsigned  simd;
		int unsigned  th;
		int unsigned  weight_width;
		int unsigned  activation_width;
		int unsigned  accu_width;
		bit  signed_activations;
		bit  narrow_weights;
	} cfg_t;

	// Constraints enforced by mvu_tiled_axi:
	//  - MW % SIMD == 0
	//  - MH % PE == 0
	//  - (PE * SIMD) % TH == 0
	//  - WEIGHT_WIDTH <= 8
	//  - ACTIVATION_WIDTH <= 8 (9 for signed -- uses full 9-bit A port)
	//  - TH >= 2 (TH=1 uses the non-tiled path)
	//
	// Test selection rationale:
	//  0: Baseline -- balanced PE/SIMD, TH=2, signed activations, narrow weights
	//  1: Larger TH (=3), odd SIMD (=3) -> CHAINLEN=1 (3 lanes in one DSP)
	//  2: TH = PE*SIMD (maximum tiling, WSIMD=1) -- edge case: 1 weight/cycle
	//  3: High PE (=6), low SIMD (=2) -> wide PE fanout, CHAINLEN=1
	//  4: PE=MH (no replay, single neuron fold) -- tests replay bypass
	//  5: Large matrix, moderate tiling -- closer to real workload
	//  6: SIMD=6 (CHAINLEN=2), TH=2 -- multi-DSP chain with tiling
	//  7: Unsigned activations, small bitwidths -- corner case for sign extension
	//  8: TH=6 (high tiling), PE=2, SIMD=3 -- stress accumulator depth
	localparam int unsigned  TEST_COUNT = 9;
	//       mh  mw  pe simd th  ww  aw  accw  sa  nw
	localparam cfg_t  TESTS[TEST_COUNT] = '{
		'{ 12, 12,  6,  3,  2,  8,  8, 24, 1, 1 },
		'{ 12, 12,  6,  3,  3,  4,  4, 16, 1, 0 },
		'{ 12,  8,  2,  4,  8,  8,  8, 24, 1, 0 },
		'{ 12, 10,  6,  2,  3,  4,  8, 20, 0, 0 },
		'{  4, 12,  4,  3,  2,  8,  4, 20, 1, 1 },
		'{ 24, 18,  6,  6,  3,  4,  4, 18, 1, 0 },
		'{ 16, 12,  4,  6,  2,  8,  8, 24, 0, 1 },
		'{  8, 12,  4,  3,  2,  2,  2, 12, 0, 1 },
		'{  6,  9,  2,  3,  6,  4,  4, 16, 1, 0 }
	};

	//=== Global Control ====================================================
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  clk2x = 0;
	always #2.5ns clk2x = !clk2x;

	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
		// Allow 100ns DSP startup recovery before any input
		#100ns;
	end

	bit [TEST_COUNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	//=== Parallel Test Instantiation =======================================
	for(genvar  t = 0; t < TEST_COUNT; t++) begin : genTests
		localparam cfg_t  CFG = TESTS[t];
		localparam int unsigned  MH   = CFG.mh;
		localparam int unsigned  MW   = CFG.mw;
		localparam int unsigned  PE   = CFG.pe;
		localparam int unsigned  SIMD = CFG.simd;
		localparam int unsigned  TH   = CFG.th;
		localparam int unsigned  WEIGHT_WIDTH     = CFG.weight_width;
		localparam int unsigned  ACTIVATION_WIDTH = CFG.activation_width;
		localparam int unsigned  ACCU_WIDTH       = CFG.accu_width;

		// Derived
		localparam int unsigned  SF   = MW / SIMD;   // SIMD folds
		localparam int unsigned  NF   = MH / PE;     // neuron folds
		localparam int unsigned  WSIMD = (PE * SIMD) / TH;

		typedef logic signed [WEIGHT_WIDTH    -1:0]  weight_t;
		typedef logic        [ACTIVATION_WIDTH-1:0]  activation_t;
		typedef logic signed [ACCU_WIDTH      -1:0]  accu_t;

		// Stream widths (matching mvu_tiled_axi localparams)
		localparam int unsigned  WEIGHT_STREAM_WIDTH    = WSIMD * WEIGHT_WIDTH;
		localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8;
		localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH;
		localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8;
		localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH;
		localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8;

		// DUT signals
		logic [WEIGHT_STREAM_WIDTH_BA-1:0]  wdat;
		logic  wvld;
		uwire  wrdy;
		logic [INPUT_STREAM_WIDTH_BA-1:0]  idat;
		logic  ivld;
		uwire  irdy;
		uwire [OUTPUT_STREAM_WIDTH_BA-1:0]  odat;
		uwire  ovld;
		logic  ordy;

		mvu_tiled_axi #(
			.PE(PE), .SIMD(SIMD),
			.WEIGHT_WIDTH(WEIGHT_WIDTH),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
			.ACCU_WIDTH(ACCU_WIDTH),
			.MW(MW), .MH(MH), .TH(TH),
			.SIGNED_ACTIVATIONS(CFG.signed_activations),
			.NARROW_WEIGHTS(CFG.narrow_weights),
			.PUMPED_COMPUTE(0),
			.FORCE_BEHAVIORAL(0)
		) dut (
			.ap_clk(clk),
			.ap_clk2x(clk2x),
			.ap_rst_n(!rst),
			.s_axis_weights_tdata(wdat),
			.s_axis_weights_tvalid(wvld),
			.s_axis_weights_tready(wrdy),
			.s_axis_input_tdata(idat),
			.s_axis_input_tvalid(ivld),
			.s_axis_input_tready(irdy),
			.m_axis_output_tdata(odat),
			.m_axis_output_tvalid(ovld),
			.m_axis_output_tready(ordy)
		);

		//=== Input Feed & Reference Generation =============================
		// TH input vectors are batched per round. The replay buffer
		// stores TH*SF activation words (TH vectors, each SF folds)
		// and the weight buffer replays each tile TH times internally.
		//
		// Output reorder order: for each TH slot, all NF neuron
		// folds in sequence.
		accu_t [PE-1:0]  Q[$];
		initial begin
			wdat = 'x;  wvld = 0;
			idat = 'x;  ivld = 0;
			@(posedge clk iff !rst);

			// Wait for DSP startup recovery
			repeat(20) @(posedge clk);

			repeat(ROUNDS) begin
				// TH activation vectors per batch
				automatic activation_t [TH-1:0][MW-1:0]          ivecs;
				automatic weight_t     [MH-1:0][MW-1:0]          iwgt;
				automatic accu_t       [TH-1:0][MH-1:0]          ovecs;

				// Randomize all inputs
				void'(std::randomize(ivecs, iwgt));

				// Sanitize weights (narrow + overflow) using first vector
				for(int unsigned  h = 0; h < MH; h++) begin
					automatic accu_t  p = 0;
					for(int unsigned  w = 0; w < MW; w++) begin
						automatic weight_t  w0 = iwgt[h][w];
						automatic accu_t  m0, p0;

						if(CFG.narrow_weights && (w0 == weight_t'(1 << (WEIGHT_WIDTH-1))))  w0++;
						m0 = w0 * $signed({CFG.signed_activations && ivecs[0][w][ACTIVATION_WIDTH-1], ivecs[0][w]});
						p0 = p + m0;
						if(((m0 < 0) == (p < 0)) && ((m0 < 0) != (p0 < 0)))  w0 = 0;
						else  p = p0;

						iwgt[h][w] = w0;
					end
				end

				// Compute golden reference for each of TH vectors
				for(int unsigned  y = 0; y < TH; y++) begin
					for(int unsigned  h = 0; h < MH; h++) begin
						automatic accu_t  p = 0;
						for(int unsigned  w = 0; w < MW; w++) begin
							p += $signed(iwgt[h][w]) * $signed({CFG.signed_activations && ivecs[y][w][ACTIVATION_WIDTH-1], ivecs[y][w]});
						end
						ovecs[y][h] = p;
					end
				end

				// Enqueue expected outputs in reorder_out order:
				// for each TH slot, all NF neuron folds
				for(int unsigned  y = 0; y < TH; y++) begin
					for(int unsigned  h = 0; h < MH; h += PE) begin
						Q.push_back(ovecs[y][h+:PE]);
					end
				end

				// Feed activations and weights concurrently
				fork
					//-- Activation feed --
					// Replay buffer write order: X (SF) inner, Y (TH) outer.
					// Feed TH vectors, each SF SIMD-wide words.
					begin : blkActFeed
						for(int unsigned  y = 0; y < TH; y++) begin
							for(int unsigned  x = 0; x < SF; x++) begin
								while($urandom()%19 == 0) @(posedge clk);
								idat <= ivecs[y][x*SIMD +: SIMD];
								ivld <= 1;
								@(posedge clk iff irdy);
								idat <= 'x;
								ivld <= 0;
							end
						end
					end : blkActFeed

					//-- Weight feed --
					// One weight matrix, chunked: for each NF, for each SF,
					// send TH chunks of WSIMD weights.
					begin : blkWgtFeed
						for(int unsigned  h = 0; h < MH; h += PE) begin
							for(int unsigned  w = 0; w < MW; w += SIMD) begin
								// Build full PE*SIMD weight tile
								automatic weight_t [PE-1:0][SIMD-1:0]  wtile;
								for(int unsigned  pe = 0; pe < PE; pe++) begin
									for(int unsigned  simd = 0; simd < SIMD; simd++) begin
										wtile[pe][simd] = iwgt[h+pe][w+simd];
									end
								end

								// Slice into TH chunks of WSIMD weights
								for(int unsigned  chunk = 0; chunk < TH; chunk++) begin
									automatic logic [WEIGHT_STREAM_WIDTH_BA-1:0]  wword = '0;
									for(int unsigned  k = 0; k < WSIMD; k++) begin
										automatic int unsigned  flat_idx = chunk * WSIMD + k;
										automatic int unsigned  pe_idx   = flat_idx / SIMD;
										automatic int unsigned  simd_idx = flat_idx % SIMD;
										wword[k*WEIGHT_WIDTH +: WEIGHT_WIDTH] = wtile[pe_idx][simd_idx];
									end

									while($urandom()%23 == 0) @(posedge clk);
									wdat <= wword;
									wvld <= 1;
									@(posedge clk iff wrdy);
									wdat <= 'x;
									wvld <= 0;
								end
							end
						end
					end : blkWgtFeed
				join
			end

			repeat(256) @(posedge clk);
			assert(Q.size == 0) else begin
				$error("Test #%0d: Missing %0d outputs.", t, Q.size);
				$stop;
			end
			done[t] = 1;
		end

		//=== Output Checker ================================================
		int unsigned  Checks = 0;
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			forever begin
				automatic accu_t [PE-1:0]  exp;
				automatic accu_t [PE-1:0]  p;

				while(($urandom() % 59) == 0) @(posedge clk);

				// Drain one output
				ordy <= 1;
				@(posedge clk iff ovld);
				ordy <= 0;

				p = odat;
				assert(Q.size > 0) else begin
					$error("Test #%0d: Spurious output: %0p.", t, p);
					$stop;
				end

				exp = Q.pop_front();
				assert(p === exp) else begin
					$error("Test #%0d: Output mismatch %0p instead of %0p.", t, p, exp);
					$stop;
				end

				Checks <= Checks + 1;
			end
		end

		final begin
			assert(Checks == ROUNDS * NF * TH)
				$display("Test #%0d: OK -- %0d checks (MH=%0d MW=%0d PE=%0d SIMD=%0d TH=%0d).",
					t, Checks, MH, MW, PE, SIMD, TH);
			else
				$error("Test #%0d: Unexpected check count: %0d instead of %0d.", t, Checks, ROUNDS * NF * TH);
		end

	end : genTests

endmodule : mvu_tiled_axi_tb

/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for MVU AXI wrapper module.
 *****************************************************************************/

module mvu_axi_tb;

	// Test Configurations
	localparam bit  FORCE_BEHAVIORAL = 0;
	localparam int unsigned  ROUNDS = 5;

	typedef struct {
		int unsigned  version;
		int unsigned  mh;
		int unsigned  mw;
		int unsigned  pe;
		int unsigned  simd;
		int unsigned  weight_width;
		int unsigned  activation_width;
		int unsigned  accu_width;
		bit  signed_activations;
		bit  narrow_weights;
		int unsigned  th;     // Tile height (>=1)
		int unsigned  wsimd;  // Weight stream parallelism (0 => PE*SIMD)
	} cfg_t;
	localparam int unsigned  TEST_COUNT = 16;
	localparam cfg_t  TESTS[TEST_COUNT] = '{
		//     ver  mh  mw  pe simd ww  aw  accw sa nw  th  wsimd
		// --- Lane-sliced × TH=1 ---
		// T0: DSP48E2, PE=2, SIMD=2
		'{      2,   4,  4,  2,  2,  4,  4,  16,  0, 0,  1,  0 },
		// T1: DSP48E2, PE=1, SIMD=1 — single lane
		'{      2,   4,  4,  1,  1,  4,  4,  16,  0, 0,  1,  0 },
		// --- Lane-sliced × TH≥2 ---
		// T2: DSP48E2, TH=2, PE=2, SIMD=1 — cross-lane overflow
		'{      2,   4,  2,  2,  1,  4,  4,  16,  0, 0,  2,  0 },
		// T3: DSP48E2, TH=2, PE=1, SIMD=1 — single lane
		'{      2,   4,  2,  1,  1,  4,  4,  16,  0, 0,  2,  0 },
		// T4: DSP48E2, TH=2, PE=2, SIMD=2 — multi-SIMD reduction tree
		'{      2,   4,  4,  2,  2,  4,  4,  16,  0, 0,  2,  0 },
		// T5: DSP48E2, TH=3, PE=2, SIMD=1 — Fb ring, ClrLine, HiClrDly
		'{      2,   6,  3,  2,  1,  4,  4,  16,  0, 0,  3,  0 },
		// T6: DSP48E2, TH=2, signed activations
		'{      2,   4,  2,  2,  1,  4,  4,  16,  1, 0,  2,  0 },
		// T7: DSP48E2, TH=2, narrow weights
		'{      2,   4,  2,  2,  1,  4,  4,  16,  0, 1,  2,  0 },
		// T8: DSP48E1, TH=2
		'{      1,   4,  2,  2,  1,  4,  4,  16,  0, 0,  2,  0 },
		// T9: DSP48E1, TH=1
		'{      1,   4,  4,  2,  2,  4,  4,  16,  0, 0,  1,  0 },
		// --- INT8 (mvu_vvu_8sx9_dsp58) × TH=1 ---
		// T10: DSP58, 8b×8b, PE=4, SIMD=6 — 3-MAC baseline
		'{      3,  16, 48,  4,  6,  8,  8,  24,  0, 0,  1,  0 },
		// --- INT8 (mvu_vvu_8sx9_dsp58) × TH≥2 ---
		// T11: DSP58, 8b×8b, TH=2, PE=2, SIMD=1 — minimal tiled INT8
		'{      3,   4,  2,  2,  1,  8,  8,  24,  0, 0,  2,  0 },
		// T12: DSP58, 8b×8b, TH=2, PE=4, SIMD=6 — full 3-MAC chain tiled
		'{      3,  16, 48,  4,  6,  8,  8,  24,  0, 0,  2,  0 },
		// T13: DSP58, 4b×4b, TH=3, PE=2, SIMD=1 — sub-INT8 widths, deeper tile
		'{      3,   6,  3,  2,  1,  4,  4,  16,  0, 0,  3,  0 },
		// T14: DSP58, 8b×8b, TH=2, signed activations
		'{      3,   4,  2,  2,  1,  8,  8,  24,  1, 0,  2,  0 },
		// T15: DSP58, 8b×8b, TH=3, PE=2, SIMD=1 — true INT8 × TH>2
		'{      3,   6,  3,  2,  1,  8,  8,  24,  0, 0,  3,  0 }
	};

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	bit [TEST_COUNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	//-----------------------------------------------------------------------
	// Parallel Test Instantiation
	for(genvar  t = 0; t < TEST_COUNT; t++) begin : genTests
		localparam cfg_t  CFG = TESTS[t];
		localparam int unsigned  MH   = CFG.mh;
		localparam int unsigned  MW   = CFG.mw;
		localparam int unsigned  PE   = CFG.pe;
		localparam int unsigned  SIMD = CFG.simd;
		localparam int unsigned  WEIGHT_WIDTH     = CFG.weight_width;
		localparam int unsigned  ACTIVATION_WIDTH = CFG.activation_width;
		localparam int unsigned  ACCU_WIDTH       = CFG.accu_width;
		localparam int unsigned  TH   = CFG.th;
		localparam int unsigned  WSIMD = CFG.wsimd > 0 ? CFG.wsimd : PE*SIMD;
		typedef logic signed [WEIGHT_WIDTH    -1:0]  weight_t;
		typedef logic        [ACTIVATION_WIDTH-1:0]  activation_t;
		typedef logic signed [ACCU_WIDTH      -1:0]  accu_t;

		// DUT stream widths
		localparam int unsigned  WEIGHT_STREAM_WIDTH    = (TH > 1 ? WSIMD : PE * SIMD) * WEIGHT_WIDTH;
		localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8;
		localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH;
		localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8;
		localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE*ACCU_WIDTH;
		localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8;

		localparam int unsigned  SF = MW / SIMD;
		localparam int unsigned  NF = MH / PE;
		localparam int unsigned  NW = (PE*SIMD) / WSIMD;

		logic [WEIGHT_STREAM_WIDTH_BA-1:0]  wdat;
		logic  wvld;
		uwire  wrdy;
		logic [INPUT_STREAM_WIDTH_BA-1:0]  idat;
		logic  ivld;
		uwire  irdy;
		uwire [OUTPUT_STREAM_WIDTH_BA-1:0]  odat;
		uwire  ovld;
		logic  ordy;
		mvu_vvu_axi #(
			.IS_MVU(1),
			.VERSION(CFG.version),

			.MH(MH), .MW(MW),
			.PE(PE), .SIMD(SIMD),
			.WEIGHT_WIDTH    (WEIGHT_WIDTH),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
			.ACCU_WIDTH      (ACCU_WIDTH),

			.SIGNED_ACTIVATIONS(CFG.signed_activations),
			.NARROW_WEIGHTS    (CFG.narrow_weights),
			.TH(TH),
			.WSIMD(WSIMD),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) dut (
			.ap_clk(clk), .ap_rst_n(!rst),
			.s_axis_weights_tdata(wdat), .s_axis_weights_tvalid(wvld), .s_axis_weights_tready(wrdy),
			.s_axis_input_tdata  (idat), .s_axis_input_tvalid  (ivld), .s_axis_input_tready  (irdy),
			.m_axis_output_tdata (odat), .m_axis_output_tvalid (ovld), .m_axis_output_tready (ordy)
		);

		// Input Feed
		accu_t [PE-1:0]  Q[$]; // Reference Queue
		initial begin
			wdat = 'x; wvld = 0;
			idat = 'x; ivld = 0;
			@(posedge clk iff !rst);

			if(TH <= 1) begin
				// ===================== TH=1 Path =====================
				// Produce Results for ROUNDS Vectors
				repeat(ROUNDS) begin
					automatic activation_t [MW-1:0]          ivec;
					automatic weight_t     [MH-1:0][MW-1:0]  iwgt;
					automatic accu_t       [MH-1:0]          ovec;

					// Valid randomized input and reference result
					void'(std::randomize(ivec, iwgt));
					for(int unsigned  h = 0; h < MH; h++) begin
						automatic accu_t  p = 0;
						for(int unsigned  w = 0; w < MW; w++) begin
							automatic weight_t  w0 = iwgt[h][w];
							automatic accu_t  m0;
							automatic accu_t  p0;

							if(CFG.narrow_weights && (w0 == weight_t'(1<<(WEIGHT_WIDTH-1))))  w0++;
							m0 = w0 * $signed({CFG.signed_activations && ivec[w][ACTIVATION_WIDTH-1], ivec[w]});
							p0 = p + m0;
							if(((m0 < 0) == (p < 0)) && ((m0 < 0) != (p0 < 0)))  w0 = 0;
							else  p = p0;

							iwgt[h][w] = w0;
						end
						ovec[h] = p;
					end

					// Input Feed
					fork
						// Scan through Activation Vector
						for(int unsigned  w = 0; w < MW; w += SIMD) begin : blkActFeed
							while($urandom()%19 == 0) @(posedge clk);
							idat <= ivec[w+:SIMD];
							ivld <= 1;
							@(posedge clk iff irdy);
							idat <= 'x;
							ivld <= 0;
						end : blkActFeed

						// Scan through Weight Matrix
						for(int unsigned  h = 0; h < MH; h += PE) begin : blkWgtFeed
							for(int unsigned  w = 0; w < MW; w += SIMD) begin
								automatic weight_t [PE-1:0][SIMD-1:0]  wtile;
								for(int unsigned  pe = 0; pe < PE; pe++) begin
									for(int unsigned  simd = 0; simd < SIMD; simd++) begin
										wtile[pe][simd] = iwgt[h+pe][w+simd];
									end
								end

								while($urandom()%23 == 0) @(posedge clk);
								wdat <= wtile;
								wvld <= 1;
								@(posedge clk iff wrdy);
								wdat <= 'x;
								wvld <= 0;
							end

							// Enqueue Reference Output for this Slice
							Q.push_back(ovec[h+:PE]);

						end : blkWgtFeed
					join
				end
			end
			else begin
				// ===================== TH>1 Path =====================
				// Each round: TH independent activation vectors, shared weight matrix
				repeat(ROUNDS) begin
					automatic activation_t [TH-1:0][MW-1:0]  ivecs;
					automatic weight_t     [MH-1:0][MW-1:0]  iwgt;
					automatic accu_t       [TH-1:0][MH-1:0]  ovecs;

					// Fully randomized inputs
					void'(std::randomize(ivecs, iwgt));
					// th=0: correct weights and compute reference
					for(int unsigned  h = 0; h < MH; h++) begin
						automatic accu_t  p = 0;
						for(int unsigned  w = 0; w < MW; w++) begin
							automatic weight_t  w0 = iwgt[h][w];
							automatic accu_t  m0;
							automatic accu_t  p0;

							if(CFG.narrow_weights && (w0 == weight_t'(1<<(WEIGHT_WIDTH-1))))  w0++;
							m0 = w0 * $signed({CFG.signed_activations && ivecs[0][w][ACTIVATION_WIDTH-1], ivecs[0][w]});
							p0 = p + m0;
							if(((m0 < 0) == (p < 0)) && ((m0 < 0) != (p0 < 0)))  w0 = 0;
							else  p = p0;

							iwgt[h][w] = w0;
						end
						ovecs[0][h] = p;
					end
					// th>0: use corrected weights, no further modification
					for(int unsigned  th = 1; th < TH; th++) begin
						for(int unsigned  h = 0; h < MH; h++) begin
							automatic accu_t  p = 0;
							for(int unsigned  w = 0; w < MW; w++) begin
								automatic weight_t  w0 = iwgt[h][w];
								automatic accu_t  m0;

								m0 = w0 * $signed({CFG.signed_activations && ivecs[th][w][ACTIVATION_WIDTH-1], ivecs[th][w]});
								p = p + m0;
							end
							ovecs[th][h] = p;
						end
					end

					// Enqueue reference outputs in transposed order matching output reorder.
					// Output reorder (DIMS={TH, NF}, COEFS={1, TH}) produces:
					//   for th: for nf: emit(buf[1*th + TH*nf])
					// which yields order: th0_nf0, th0_nf1, ..., th0_nfN, th1_nf0, ...
					for(int unsigned  th = 0; th < TH; th++)
						for(int unsigned  h = 0; h < MH; h += PE)
							Q.push_back(ovecs[th][h+:PE]);

					// Input Feed
					fork
						// Feed activations: SF*TH elements
						// input_gen schedule: for nf: for sf: for th: emit(buf[sf + SF*th])
						// So we need to provide SF*TH elements in memory order:
						//   buf[0], buf[1], ..., buf[SF-1],        <- vector 0
						//   buf[SF], buf[SF+1], ..., buf[2*SF-1],  <- vector 1
						//   ...
						//   buf[(TH-1)*SF], ..., buf[TH*SF-1]      <- vector TH-1
						begin : blkActFeed
							for(int unsigned  th = 0; th < TH; th++) begin
								for(int unsigned  w = 0; w < MW; w += SIMD) begin
									while($urandom()%19 == 0) @(posedge clk);
									idat <= ivecs[th][w+:SIMD];
									ivld <= 1;
									@(posedge clk iff irdy);
									idat <= 'x;
									ivld <= 0;
								end
							end
						end : blkActFeed

						// Feed weights: NW bursts of WSIMD elements per NF neuron fold
						begin : blkWgtFeed
							for(int unsigned  h = 0; h < MH; h += PE) begin
								for(int unsigned  w = 0; w < MW; w += SIMD) begin
									// Pack [PE][SIMD] tile into NW bursts of WSIMD
									automatic weight_t [PE-1:0][SIMD-1:0]  wtile;
									for(int unsigned  pe = 0; pe < PE; pe++) begin
										for(int unsigned  simd = 0; simd < SIMD; simd++) begin
											wtile[pe][simd] = iwgt[h+pe][w+simd];
										end
									end

									// Send NW bursts
									for(int unsigned  nw = 0; nw < NW; nw++) begin
										automatic logic [WSIMD-1:0][WEIGHT_WIDTH-1:0]  wburst;
										for(int unsigned  ws = 0; ws < WSIMD; ws++) begin
											automatic int unsigned  flat_idx = nw * WSIMD + ws;
											automatic int unsigned  pe_idx = flat_idx / SIMD;
											automatic int unsigned  simd_idx = flat_idx % SIMD;
											wburst[ws] = wtile[pe_idx][simd_idx];
										end

										while($urandom()%23 == 0) @(posedge clk);
										wdat <= wburst;
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
			end

			repeat(16 + 4*TH*NF) @(posedge clk);
			assert(Q.size == 0) else begin
				$error("Test #%0d: Missing %0d outputs.", t, Q.size);
				$stop;
			end
			done[t] = 1;
		end

		// Output Checker
		int unsigned  Checks = 0;
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			forever begin
				automatic accu_t [PE-1:0]  exp;
				automatic accu_t [PE-1:0]  p;

				while(($urandom() % 59) == 0) @(posedge clk);

				// Drain
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

		localparam int unsigned  EXPECTED_CHECKS = TH <= 1 ? ROUNDS*NF : ROUNDS*NF*TH;
		final begin
			assert(Checks == EXPECTED_CHECKS)  $display("Test #%0d: Successfully performed %0d checks.", t, Checks);
			else  $error("Test #%0d: Unexpected number of checks: %0d instead of %0d.", t, Checks, EXPECTED_CHECKS);
		end

	end : genTests

endmodule : mvu_axi_tb

/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for DOTP AXI wrapper module.
 *****************************************************************************/

module mvu_axi_tb;

	// Test Configurations
	localparam bit  FORCE_BEHAVIORAL = 0;
	localparam int unsigned  ROUNDS = 17;

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
	} cfg_t;
	localparam int unsigned  TEST_COUNT = 9;
	localparam cfg_t  TESTS[TEST_COUNT] = '{
		'{ 1, 68, 48,  4,  3,  4,  4, 16, 0, 1 },
		'{ 1, 56, 45,  7,  5,  4,  3, 15, 1, 0 },
		'{ 1, 42, 52,  6,  4,  3,  5,  8, 0, 0 },
		'{ 2, 62, 22,  2,  2, 15, 10, 40, 1, 0 },
		'{ 2, 48, 28,  8,  4,  4,  4, 18, 0, 0 },
		'{ 3, 57, 34,  3,  2,  2,  4, 17, 1, 1 },
		'{ 3, 70, 40, 10,  5,  2,  2, 17, 1, 1 },
		'{ 3, 36, 37,  3,  1,  2, 20, 25, 0, 1 },
		'{ 3, 30, 40,  2, 10,  7,  8, 23, 0, 0 }
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
		typedef logic signed [WEIGHT_WIDTH    -1:0]  weight_t;
		typedef logic        [ACTIVATION_WIDTH-1:0]  activation_t;
		typedef logic signed [ACCU_WIDTH      -1:0]  accu_t;

		// DUT
		localparam int unsigned  WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH;
		localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8;
		localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH;
		localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8;
		localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE*ACCU_WIDTH;
		localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8;
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

			repeat(16) @(posedge clk);
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

		final begin
			assert(Checks == ROUNDS*MH/PE)  $display("Test #%0d: Successfully performed %0d checks.", t, Checks);
			else  $error("Test #%0d: Unexpected number of checks: %0d instead of %0d.", t, Checks, ROUNDS);
		end

	end : genTests

endmodule : mvu_axi_tb

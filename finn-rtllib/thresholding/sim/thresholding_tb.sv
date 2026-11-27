/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *	 this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *	 contributors may be used to endorse or promote products derived from
 *	 this software without specific prior written permission.
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
 * @brief	Testbench for thresholding_axi.
 * @author	Monica Chiosa <monica.chiosa@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module thresholding_tb;

	localparam bit  VERBOSE = 0;
	localparam int unsigned  ROUNDS = 503;
	typedef struct {
		int unsigned  k;	// input precision
		int unsigned  kt;	// threshold precision, at most k
		int unsigned  n;	// number of thresholds per channel
		int unsigned  c;	// number of channels
		int unsigned  pe;	// parallel PEs
		bit  sign;	// signed inputs
		bit  fparg;	// floating-point inputs
		bit  deep_pipeline;
		bit  throttled;	// throttle input and output interfaces occasionally
	} testcfg_t;
	localparam int unsigned  TEST_CNT = 6;
	localparam testcfg_t  TESTS[TEST_CNT] = '{
		testcfg_t'{ k:  5, kt:  5, n:  1, c:  1, pe: 1, sign: 0, fparg: 0, deep_pipeline: 0, throttled: 0 },
		testcfg_t'{ k: 10, kt: 10, n:  8, c: 12, pe: 3, sign: 1, fparg: 0, deep_pipeline: 1, throttled: 1 },
		testcfg_t'{ k:  8, kt:  8, n:  3, c:  8, pe: 4, sign: 0, fparg: 0, deep_pipeline: 0, throttled: 1 },
		testcfg_t'{ k: 17, kt: 17, n:  9, c: 10, pe: 5, sign: 1, fparg: 1, deep_pipeline: 1, throttled: 0 },
		testcfg_t'{ k: 12, kt:  5, n: 15, c:  6, pe: 2, sign: 1, fparg: 0, deep_pipeline: 0, throttled: 1 },	// narrow thresholds
		testcfg_t'{ k:  9, kt:  4, n:  5, c:  4, pe: 4, sign: 0, fparg: 0, deep_pipeline: 1, throttled: 1 }	// narrow thresholds
	};

	//-----------------------------------------------------------------------
	// Clock and Reset Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		#10ns;
		@(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// Parallel Instances Running Individual Test
	bit [TEST_CNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end
	for(genvar  i = 0; i < TEST_CNT; i++) begin : genDUTs

		//- Extract Test Config ---------
		localparam testcfg_t  CFG = TESTS[i];
		localparam int unsigned  K = CFG.k;
		localparam int unsigned  KT = CFG.kt;
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  C = CFG.c;
		localparam int unsigned  PE = CFG.pe;
		localparam bit  SIGNED = CFG.sign;
		localparam bit  FPARG  = CFG.fparg;
		localparam bit  DEEP_PIPELINE = CFG.deep_pipeline;
		localparam bit  THROTTLED     = CFG.throttled;

		// Derived Parameters and Types -
		localparam int unsigned  CF = C/PE;	// Channel Fold
		localparam int unsigned  O_BITS = $clog2(N+1);
		typedef logic [K     -1:0]  val_t;
		typedef logic [KT    -1:0]  thr_t;
		typedef logic [O_BITS-1:0]  res_t;
		typedef logic [$clog2(CF)+$clog2(PE)+$clog2(N)-1:0]  addr_t;

		// Configuration Address: ((c/PE)*2^$clog2(PE) + c%PE)*2^$clog2(N) + t
		//	Composed arithmetically as fields may be empty (CF, PE or N = 1).
		function automatic addr_t cfg_addr(input int unsigned  cf, input int unsigned  pe, input int unsigned  t);
			return  addr_t'(((cf * 2**$clog2(PE)) + pe) * 2**$clog2(N) + t);
		endfunction : cfg_addr
		function automatic int unsigned  addr_cnl(input addr_t  addr);	// channel of an address
			automatic int unsigned  a = int'(addr) / 2**$clog2(N);
			return  (a / 2**$clog2(PE)) * PE + a % 2**$clog2(PE);
		endfunction : addr_cnl
		function automatic int unsigned  addr_idx(input addr_t  addr);	// threshold index of an address
			return  int'(addr) % 2**$clog2(N);
		endfunction : addr_idx

		//- DUT -------------------------
		typedef val_t [PE-1:0]  input_t;
		typedef res_t [PE-1:0]  output_t;

		logic  cfg_en;
		logic  cfg_we;
		addr_t  cfg_a;
		thr_t  cfg_d;
		uwire  cfg_rack;
		uwire thr_t  cfg_q;

		uwire  irdy;
		logic  ivld;
		input_t  idat;

		logic  ordy = 0;
		uwire  ovld;
		uwire output_t  odat;

		// Comparison policy bound per data type as by thresholding_axi
		if(FPARG) begin : genFloat
			thresholding #(
				.CMP(thresholding_pkg::cmp_float_t#(K)),
				.N(N), .K(K), .KT(KT), .C(C), .PE(PE), .USE_CONFIG(1), .DEEP_PIPELINE(DEEP_PIPELINE)
			) dut (.iset('x), .*);
		end : genFloat
		else if(SIGNED) begin : genSigned
			thresholding #(
				.CMP(thresholding_pkg::cmp_sfixed_t#(K, KT)),
				.N(N), .K(K), .KT(KT), .C(C), .PE(PE), .USE_CONFIG(1), .DEEP_PIPELINE(DEEP_PIPELINE)
			) dut (.iset('x), .*);
		end : genSigned
		else begin : genUnsigned
			thresholding #(
				.CMP(thresholding_pkg::cmp_ufixed_t#(K, KT)),
				.N(N), .K(K), .KT(KT), .C(C), .PE(PE), .USE_CONFIG(1), .DEEP_PIPELINE(DEEP_PIPELINE)
			) dut (.iset('x), .*);
		end : genUnsigned

		// Expected Ordering
		function val_t reord(input val_t  x);
			automatic val_t  res = x;
			if(SIGNED) begin
				if(FPARG && x[K-1])  res[K-2:0] = ~x[K-2:0];
				res[K-1] = !x[K-1];
			end
			return  res;
		endfunction : reord

		//- Threshold Definition --------
		typedef val_t  threshs_t[C][N];
		threshs_t  THRESHS;
		initial begin
			static val_t  row[N];

			// Generate thresholds
			foreach(THRESHS[c]) begin
				static thr_t [N-1:0]  r;
				void'(std::randomize(r));
				// thresholds of KT bits, extended to the input precision
				foreach(row[i])  row[i] = SIGNED? val_t'($signed(r[i])) : val_t'(r[i]);
				// Key inlined: xelab 2024.2 crashes on reord(item) here
				row.sort with (item ^ {SIGNED, {(K-1){SIGNED && FPARG && item[K-1]}}});
				THRESHS[c] = row;
			end

			// Report test case details
			$display("[%0d] Thresholding %s%s%0d -> uint%0d", i, SIGNED? "s" : "u", FPARG? "fp" : "int", K, N);
			for(int unsigned  c = 0; c < C; c++) begin
				$write("[%0d] Channel #%0d: Thresholds = {", i, c);
				for(int unsigned  i = 0; i < N; i++)  $write(" %0X", THRESHS[c][i]);
				$display(" }");
			end
		end

		//- Stimulus Driver -------------
		input_t  QW[$];  // Input tracing
		addr_t   QC[$];  // Readback tracking
		int unsigned  error_cnt = 0;
		initial begin
			automatic bit  term = 0;

			// Config
			cfg_en = 0;
			cfg_we = 'x;
			cfg_a  = 'x;
			cfg_d  = 'x;

			// Stream Input
			ivld = 0;
			idat = 'x;

			@(posedge clk iff !rst);

			// Threshold Configuratin
			cfg_en <= 1;
			cfg_we <= 1;
			for(int unsigned  c = 0; c < C; c+=PE) begin
				for(int unsigned  pe = 0; pe < PE; pe++) begin
					for(int unsigned  t = 0; t < N; t++) begin
						cfg_a <= cfg_addr(c/PE, pe, t);
						cfg_d <= thr_t'(THRESHS[c+pe][t]);
						@(posedge clk);
					end
				end
			end
			cfg_en <= 0;
			cfg_we <= 'x;
			cfg_d  <= 'x;

			// Operation
			fork
				// Intermittent configuration readback
				while(!term) begin
					@(posedge clk);
					if(($urandom()%41) == 0) begin
						automatic addr_t  addr = cfg_addr($urandom()%CF, $urandom()%PE, $urandom()%N);

						cfg_en <= 1;
						cfg_we <= 0;
						cfg_a  <= addr;
						@(posedge clk);
						QC.push_back(addr);
						cfg_en <= 0;
						cfg_we <= 'x;
						cfg_a  <= 'x;
					end
				end

				// AXI-Stream Input
				repeat(ROUNDS) begin
					automatic input_t  dat;

					while(THROTTLED && ($urandom()%7 == 0)) @(posedge clk);

					void'(std::randomize(dat));
					ivld <= 1;
					idat <= dat;
					@(posedge clk iff irdy);
					ivld <=  0;
					idat <= 'x;
					QW.push_back(dat);
				end
			join_any
			term = 1;

			// Termination Checks: drain all outputs and readbacks (throttled
			// outputs may take longer than the pipeline depth), then idle a while
			// for the output checker to catch spurious outputs
			for(int unsigned  t = 0; (QW.size() || QC.size()) && (t < 1000); t++)  @(posedge clk);
			repeat((DEEP_PIPELINE+1)*$clog2(N+1)+8)  @(posedge clk);

			assert(QW.size() == 0) else begin
				$error("[%0d] Missing %0d outputs.", i, QW.size());
				$stop;
			end
			assert(QC.size() == 0) else begin
				$error("[%0d] Missing %0d readback replies.", i, QC.size());
				$stop;
			end

			$display("[%0d] Test completed: %0d errors in %0d tests.", i, error_cnt, ROUNDS);
			$display("=============================================");
			done[i] <= 1;
		end

		//- Readback Checker --------------
		always_ff @(posedge clk iff cfg_rack) begin
			assert(QC.size()) begin
				automatic addr_t  addr = QC.pop_front();
				automatic int unsigned  cnl = addr_cnl(addr);
				automatic int unsigned  idx = addr_idx(addr);
				automatic thr_t  exp = thr_t'(THRESHS[cnl][idx]);
				assert(cfg_q == exp) else begin
					$error("[%0d] Readback mismatch on #%0d.%0d: %0d instead of %0d", i, cnl, idx, cfg_q, exp);
					$stop;
				end
			end
			else begin
				$error("[%0d] Spurious readback output.", i);
				$stop;
			end
		end

		// Output Checker
		int unsigned  OCnl = 0;
		always @(posedge clk) begin
			if(rst) begin
				OCnl <= 0;
				ordy <= 1'b0;
			end
			else begin
				if(!ordy || ovld)  ordy <= ($urandom()%5 != 0) || !THROTTLED;

				if(ordy && ovld) begin
					assert(QW.size()) begin
						automatic input_t  x = QW.pop_front();

						for(int unsigned  pe = 0; pe < PE; pe++) begin
							automatic int unsigned  cnl = OCnl + pe;

							if(VERBOSE) $display("[%0d] Mapped CNL=%0d DAT=%3x -> #%2d", i, cnl, x[pe], odat[pe]);
							assert(
								((odat[pe] == 0) || (reord(THRESHS[cnl][odat[pe]-1]) <= reord(x[pe]))) &&
								((odat[pe] == N) || (reord(x[pe]) < reord(THRESHS[cnl][odat[pe]])))
							) else begin
								$error("[%0d] Output error on presumed input CNL=%0d DAT=0x%0x -> #%0d", i, cnl, x[pe], odat[pe]);
								error_cnt++;
								$stop;
							end
						end
					end
					else begin
						$error("[%0d] Spurious output.", i);
						$stop;
					end

					OCnl <= (OCnl + PE)%C;
				end
			end
		end

	end : genDUTs

endmodule: thresholding_tb

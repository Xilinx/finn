/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Skeleton testbench for requant.
 *****************************************************************************/

module requant_tb;

	localparam int unsigned  ROUNDS = 257;
	localparam int unsigned  PIPELINE_LATENCY = 4;
	localparam int unsigned  C = 6;

	typedef struct {
		int unsigned  version;
		int unsigned  pe;
		int unsigned  k;
		int unsigned  n;
		shortreal  scales[C];
		shortreal  biases[C];
		bit  throttled;
	} cfg_t;

	localparam int unsigned  TEST_CNT = 7;
	localparam cfg_t  TESTS[TEST_CNT] = '{
		'{ version: 1, pe: 1, k: 12, n: 8,
		   scales: '{ 0.0625, 0.1250, 0.2500, 0.5000, 0.7500, 1.0000 },
		   biases: '{ 12.0, 8.0, 4.0, 0.0, -2.0, -4.0 },
		   throttled: 1 },
		'{ version: 2, pe: 2, k: 16, n: 6,
		   scales: '{ 0.03125, 0.0625, 0.1250, 0.2500, 0.5000, 0.7500 },
		   biases: '{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 },
		   throttled: 1 },
		'{ version: 3, pe: 3, k: 10, n: 5,
		   scales: '{ 1.0000, 0.7500, 0.5000, 0.2500, 0.1250, 0.0625 },
		   biases: '{ -4.0, -2.0, 0.0, 2.0, 4.0, 6.0 },
		   throttled: 0 },
		'{ version: 1, pe: 6, k: 8, n: 4,
		   scales: '{ 0.5000, 0.3750, 0.2500, 0.1875, 0.1250, 0.0625 },
		   biases: '{ 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 },
		   throttled: 0 },
		'{ version: 2, pe: 2, k: 14, n: 7,
		   scales: '{ 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 1.2000 },
		   biases: '{ -3.0, -1.5, 0.0, 1.5, 3.0, 4.5 },
		   throttled: 1 },
		'{ version: 3, pe: 3, k: 11, n: 6,
		   scales: '{ 0.015625, 0.03125, 0.0625, 0.1250, 0.2500, 0.5000 },
		   biases: '{ 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 },
		   throttled: 0 },
		'{ version: 1, pe: 1, k: 9, n: 5,
		   scales: '{ 1.2500, 1.0000, 0.7500, 0.5000, 0.2500, 0.1250 },
		   biases: '{ -1.0, -0.5, 0.0, 0.5, 1.0, 1.5 },
		   throttled: 1 }
	};

	//-----------------------------------------------------------------------
	// Clock and Reset Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// Parallel Instances Running Individual Tests
	bit [TEST_CNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	for(genvar  i = 0; i < TEST_CNT; i++) begin : genDUTs
		localparam cfg_t  CFG = TESTS[i];
		localparam int unsigned  VERSION = CFG.version;
		localparam int unsigned  PE = CFG.pe;
		localparam int unsigned  K = CFG.k;
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  CF = C/PE;
		typedef shortreal  flat_t[C];
		typedef shortreal  mat_t[PE][CF];
		function automatic mat_t to_mat(input flat_t  flat);
			mat_t  mat;
			for(int unsigned  pe = 0; pe < PE; pe++) begin
				for(int unsigned  cf = 0; cf < CF; cf++) begin
					mat[pe][cf] = flat[pe*CF + cf];
				end
			end
			return  mat;
		endfunction : to_mat
		localparam mat_t  SCALES = to_mat(CFG.scales);
		localparam mat_t  BIASES = to_mat(CFG.biases);
		localparam bit  THROTTLED = CFG.throttled;

		typedef logic signed [PE-1:0][K-1:0]  idat_t;
		typedef logic        [PE-1:0][N-1:0]  odat_t;

		logic   ivld;
		idat_t  idat;
		uwire   ovld;
		uwire odat_t  odat;

		requant #(
			.VERSION(VERSION),
			.K(K), .N(N), .C(C), .PE(PE),
			.SCALES(SCALES), .BIASES(BIASES)
		) dut (
			.clk, .rst,
			.idat, .ivld,
			.odat, .ovld
		);

		int unsigned  TxCnt = 0;
		int unsigned  RxCnt = 0;
		typedef struct {
			idat_t  x;
			odat_t  exp;
		} ref_t;
		ref_t  RefQ[$];
		initial begin
			automatic int unsigned  cf = 0;
			ivld = 0;
			idat = 'x;

			$display(
				"[%0d] VERSION=%0d K=%0d N=%0d C=%0d PE=%0d throttled=%0d",
				i, VERSION, K, N, C, PE, THROTTLED
			);
			@(posedge clk iff !rst);

			repeat(ROUNDS) begin
				automatic idat_t  x;
				automatic odat_t  exp;

				while(THROTTLED && ($urandom()%7 == 0)) @(posedge clk);

				void'(std::randomize(x));
				foreach(x[pe]) begin
					automatic int  y = int'(SCALES[pe][cf] * $signed(x[pe]) + BIASES[pe][cf]);
					if(y <        0)  y = 0;
					if(y > 2**N - 1)  y = 2**N - 1;
					exp[pe] = y;
				end
				RefQ.push_back('{ x: x, exp: exp });

				ivld <= 1;
				idat <= x;
				@(posedge clk);
				ivld <=  0;
				idat <= 'x;

				TxCnt++;
				cf = (cf + 1) % CF;
			end

			repeat(PIPELINE_LATENCY + 6) @(posedge clk);

			assert(RxCnt == TxCnt) else begin
				$error("[%0d] Output count mismatch: %0d instead of %0d.", i, RxCnt, TxCnt);
				$stop;
			end
			assert(RefQ.size() == 0) else begin
				$error("[%0d] Missing %0d queued references.", i, RefQ.size());
				$stop;
			end

			$display("[%0d] Test completed: %0d samples.", i, RxCnt);
			done[i] <= 1;
		end

		logic  VldPipe[PIPELINE_LATENCY] = '{ default: 0 };
		always_ff @(posedge clk) begin
			if(rst) begin
				VldPipe <= '{ default: 0};
				RxCnt <= 0;
			end
			else begin
				VldPipe <= { ivld, VldPipe[0:PIPELINE_LATENCY-2] };
				assert(ovld === VldPipe[PIPELINE_LATENCY-1]) else begin
					$error("[%0d] ovld mismatch: got %0b exp %0b.", i, ovld, VldPipe[PIPELINE_LATENCY-1]);
					$stop;
				end

				if(ovld) begin
					automatic ref_t  exp;

					assert(RefQ.size() > 0) else begin
						$error("[%0d] Spurious output without queued reference.", i);
						$stop;
					end

					exp = RefQ.pop_front();
					assert(odat === exp.exp) else begin
						$error("[%0d] Output mismatch: got %p instead of %p.", i, odat, exp.exp);
						$stop;
					end
					RxCnt <= RxCnt + 1;
				end
			end
		end

	end : genDUTs

endmodule : requant_tb

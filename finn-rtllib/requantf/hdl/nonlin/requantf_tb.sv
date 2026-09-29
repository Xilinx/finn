/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Self-checking testbench for requantf.
 ***************************************************************************/

module requantf_tb;
`default_nettype none

	localparam int unsigned  ROUNDS = 257;
	localparam int unsigned  PIPELINE_LATENCY = 5;
	localparam int unsigned  C = 6;
	localparam bit  FORCE_BEHAVIORAL = 1;

	typedef struct {
		int unsigned  pe;
		int unsigned  n;
		shortreal  scales[C];
		shortreal  biases[C];
		bit  throttled;
		bit  signed_out;
	} cfg_t;

	localparam int unsigned  TEST_CNT = 9;
	localparam cfg_t  TESTS[TEST_CNT] = '{
		// Unsigned output tests
		'{ pe: 1, n: 8,
		   scales: '{ 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0 },
		   biases: '{ 12.0, 8.0, 4.0, 0.0, -2.0, -4.0 },
		   throttled: 1, signed_out: 0 },
		'{ pe: 2, n: 4,
		   scales: '{ 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0 },
		   biases: '{ 7.5, 4.0, 2.0, 1.0, 0.0, -0.5 },
		   throttled: 0, signed_out: 0 },
		'{ pe: 3, n: 6,
		   scales: '{ 1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875 },
		   biases: '{ -10.0, -4.0, 0.0, 2.0, 4.0, 6.0 },
		   throttled: 1, signed_out: 0 },
		'{ pe: 6, n: 3,
		   scales: '{ 0.125, 0.25, 0.5, 1.0, 0.0625, 0.03125 },
		   biases: '{ 3.0, 2.0, 1.0, 0.0, 3.5, 3.75 },
		   throttled: 0, signed_out: 0 },
		// Power-of-two scales (exact, no FP rounding deviation)
		'{ pe: 1, n: 8,
		   scales: '{ 0.25, 0.5, 1.0, 2.0, 4.0, 0.125 },
		   biases: '{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
		   throttled: 1, signed_out: 0 },
		// Signed output tests
		'{ pe: 1, n: 8,
		   scales: '{ 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0 },
		   biases: '{ -14.0, -7.0, 0.0, 3.5, 7.0, -10.5 },
		   throttled: 1, signed_out: 1 },
		'{ pe: 2, n: 5,
		   scales: '{ 0.375, 0.1875, 0.09375, 0.046875, 0.5, 0.25 },
		   biases: '{ -4.0, -2.0, 0.0, 1.5, 3.0, -3.0 },
		   throttled: 0, signed_out: 1 },
		'{ pe: 3, n: 4,
		   scales: '{ 0.5, 0.25, 0.125, 0.0625, 1.0, 0.03125 },
		   biases: '{ -4.0, -2.0, 0.0, 1.0, 3.0, 3.5 },
		   throttled: 1, signed_out: 1 },
		// Signed, full parallelism
		'{ pe: 6, n: 6,
		   scales: '{ 0.125, 0.25, 0.5, 1.0, 0.0625, 0.03125 },
		   biases: '{ -15.0, -7.0, 0.0, 3.0, 7.0, 15.0 },
		   throttled: 0, signed_out: 1 }
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
		localparam int unsigned  PE = CFG.pe;
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  CF = C/PE;
		localparam bit  SIGNED_OUT = CFG.signed_out;
		typedef shortreal  flat_t[C];
		typedef shortreal  mat_t[PE][CF];
		function automatic mat_t to_mat(input flat_t  flat);
			mat_t  mat;
			for(int unsigned  pe = 0; pe < PE; pe++)
				for(int unsigned  cf = 0; cf < CF; cf++)
					mat[pe][cf] = flat[pe*CF + cf];
			return  mat;
		endfunction : to_mat
		localparam mat_t  SCALES = to_mat(CFG.scales);
		localparam mat_t  BIASES = to_mat(CFG.biases);
		localparam bit  THROTTLED = CFG.throttled;

		typedef logic [PE-1:0][31:0]  idat_t;
		typedef logic [PE-1:0][N-1:0] odat_t;

		logic   ivld;
		idat_t  idat;
		uwire   ovld;
		uwire odat_t  odat;

		requantf #(
			.N(N), .C(C), .PE(PE),
			.SCALES(SCALES), .BIASES(BIASES),
			.SIGNED_OUT(SIGNED_OUT),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) dut (
			.clk, .rst,
			.idat, .ivld,
			.odat, .ovld
		);

		int unsigned  TxCnt = 0;
		int unsigned  RxCnt = 0;
		typedef struct {
			odat_t  exp;
		} ref_t;
		ref_t  RefQ[$];
		initial begin
			automatic int unsigned  cf = 0;
			ivld = 0;
			idat = 'x;

			$display(
				"[%0d] N=%0d C=%0d PE=%0d throttled=%0d signed_out=%0d",
				i, N, C, PE, THROTTLED, SIGNED_OUT
			);
			@(posedge clk iff !rst);

			repeat(ROUNDS) begin
				automatic idat_t  x;
				automatic odat_t  exp;

				while(THROTTLED && ($urandom()%7 == 0)) @(posedge clk);

				// Generate random FP32 inputs with moderate magnitude
				foreach(x[pe]) begin
					automatic shortreal  v;
					// Random value in approximately [-500, 500]
					v = ($urandom() % 100000) / 100.0 - 500.0;
					x[pe] = $shortrealtobits(v);
				end

				// Compute reference: round(x*scale + bias) with clipping
				foreach(x[pe]) begin
					automatic shortreal  xf = $bitstoshortreal(x[pe]);
					automatic shortreal  v  = xf * SCALES[pe][cf] + BIASES[pe][cf];
					automatic int  y = $rtoi($floor(v + 0.5));
					if(SIGNED_OUT) begin
						if(y < -(2**(N-1)))  y = -(2**(N-1));
						if(y > 2**(N-1)-1)   y = 2**(N-1)-1;
					end
					else begin
						if(y <        0)  y = 0;
						if(y > 2**N - 1)  y = 2**N - 1;
					end
					exp[pe] = y;
				end
				RefQ.push_back('{ exp: exp });

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
				VldPipe <= '{ default: 0 };
				RxCnt <= 0;
			end
			else begin
				VldPipe <= { ivld, VldPipe[0:PIPELINE_LATENCY-2] };
				assert(ovld === VldPipe[PIPELINE_LATENCY-1]) else begin
					$error("[%0d] ovld mismatch: got %0b exp %0b.", i, ovld, VldPipe[PIPELINE_LATENCY-1]);
					$stop;
				end

				if(ovld) begin
					automatic ref_t  r;

					assert(RefQ.size() > 0) else begin
						$error("[%0d] Spurious output without queued reference.", i);
						$stop;
					end

					r = RefQ.pop_front();

					// Allow ±1 tolerance: FP32 FMA rounding may differ from
					// the shortreal reference at boundaries.
					foreach(odat[pe]) begin
						automatic int  got, exp;
						if(SIGNED_OUT) begin
							got = $signed(odat[pe]);
							exp = $signed(r.exp[pe]);
						end
						else begin
							got = odat[pe];
							exp = r.exp[pe];
						end
						assert(got == exp || got == exp+1 || got == exp-1) else begin
							$error("[%0d] PE%0d output mismatch: got %0d instead of %0d.", i, pe, got, exp);
							$stop;
						end
					end
					RxCnt <= RxCnt + 1;
				end
			end
		end

	end : genDUTs

`default_nettype wire
endmodule : requantf_tb

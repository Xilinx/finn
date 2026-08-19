/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Testbench for consolidated FIFO.
 *****************************************************************************/

module fifo_tb;
`default_nettype none
	localparam int unsigned  TXNS = 15317;
	localparam int unsigned  DATA_WIDTH = 13;
	typedef logic [DATA_WIDTH-1:0]  dat_t;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// Test Configurations
	localparam int unsigned  TEST_COUNT = 14;
	localparam int unsigned  DEPTHS[TEST_COUNT] = '{
		2, 5, 17, 18, 33,  // shift: 2 min-SRL, 5 mid, 17 SRL16, 18 SRL32, 33 upper boundary
		34, 50, 65, 257,   // distributed: 34 lower, 50 mid (MEM_SIZE>MEM_DEPTH),
		                   //   65 tight (MEM_SIZE==MEM_DEPTH), 257 upper (256+oreg)
		258, 750,          // block: 258 lower (single), 750 split
		2100, 4113, 4200   // ultra: 2100/4200 split, 4113 single
	};

	bit [TEST_COUNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	//-----------------------------------------------------------------------
	// Parallel Test Instantiation
	for(genvar  t = 0; t < TEST_COUNT; t++) begin : genTests
		localparam int unsigned  DEPTH = DEPTHS[t];

		//- DUT -------------------------
		dat_t  idat;
		logic  ivld;
		uwire  irdy;
		uwire dat_t  odat;
		uwire  ovld;
		logic  ordy;
		uwire [$clog2(DEPTH+1):0]  count;
		uwire [$clog2(DEPTH+1):0]  maxcount;
		fifo #(
			.DATA_WIDTH(DATA_WIDTH),
			.DEPTH(DEPTH)
		) dut (
			.clk, .rst,
			.idat, .ivld, .irdy,
			.odat, .ovld, .ordy,
			.count, .maxcount
		);

		//- Stimulus Feed ---------------
		dat_t         Q[$];
		int unsigned  BackCycles = 0;
		initial begin
			idat = 'x;
			ivld = 0;
			@(posedge clk iff !rst);

			repeat(TXNS) begin
				static dat_t  dat = 0;

				if($urandom()%237 == 0) begin
					repeat(2*DEPTH + 4) begin
						@(posedge clk);
						if(!irdy) begin
							if(BackCycles > 0)  BackCycles--;
							else begin
								$error("Test #%0d (depth=%0d): Unwarranted backpressure.", t, DEPTH);
								$stop;
							end
						end
					end
				end
				while($urandom()%53 == 0)  @(posedge clk);

				idat <= dat;
				ivld <= 1;
				Q.push_back(dat++);
				forever @(posedge clk) begin
					if(irdy)  break;
					if(BackCycles > 0)  BackCycles--;
					else begin
						$error("Test #%0d (depth=%0d): Unwarranted backpressure.", t, DEPTH);
						$stop;
					end
				end
				idat <= 'x;
				ivld <= 0;
			end
		end

		//- Output Checker --------------
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			repeat(TXNS) begin
				automatic dat_t  exp;

				if($urandom()%173 == 0) begin
					repeat(2*DEPTH + 5) begin
						@(posedge clk);
						BackCycles++;
					end
				end
				while($urandom()%19 == 0) begin
					@(posedge clk);
					BackCycles++;
				end
				ordy <= 1;
				@(posedge clk iff ovld);
				assert(Q.size > 0) else begin
					$error("Test #%0d (depth=%0d): Spurious output.", t, DEPTH);
					$stop;
				end
				exp = Q.pop_front();
				assert(odat === exp) else begin
					$error("Test #%0d (depth=%0d): Output mismatch: %0x instead of %0x.", t, DEPTH, odat, exp);
					$stop;
				end
				ordy <= 0;
			end

			// All TXNS items pushed and popped: the FIFO must have drained.
			repeat(4) @(posedge clk);
			assert(count == 0) else begin
				$error("Test #%0d (depth=%0d): FIFO not drained, count=%0d.", t, DEPTH, count);
				$stop;
			end
			assert(maxcount >= DEPTH) else begin
				$error(
					"Test #%0d (depth=%0d): Advertised capacity never reached, maxcount=%0d.",
					t, DEPTH, maxcount
				);
				$stop;
			end

			$display("Test #%0d (depth=%0d) completed, maxcount=%0d.", t, DEPTH, maxcount);
			done[t] <= 1;
		end
	end : genTests

`default_nettype wire
endmodule : fifo_tb

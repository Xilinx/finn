/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module skid_tb;
	localparam int unsigned  ROUNDS = 15317;

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
	// Tests
	localparam int unsigned  FEED_MAX = 7;
	localparam int unsigned  DATA_WIDTH = 13;
	typedef logic [DATA_WIDTH-1:0]  dat_t;

	bit [FEED_MAX:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	for(genvar  test = 0; test <= FEED_MAX; test++) begin : genTests
		localparam int unsigned  FEED_STAGES = test;

		//- DUT -------------------------
		dat_t  idat;
		logic  ivld;
		uwire  irdy;
		uwire dat_t  odat;
		uwire  ovld;
		logic  ordy;
		skid #(.DATA_WIDTH(DATA_WIDTH), .FEED_STAGES(FEED_STAGES)) dut (
			.clk, .rst,
			.idat, .ivld, .irdy,
			.odat, .ovld, .ordy
		);

		//- Stimulus Feed ---------------
		dat_t         Q[$];            // Refernce Output
		int unsigned  BackCycles = 0;  // Track induced Backpressure
		initial begin
			idat = 'x;
			ivld = 0;
			@(posedge clk iff !rst);

			repeat(ROUNDS) begin
				automatic dat_t  dat;

				if($urandom()%237 == 0) begin
					repeat(2*FEED_STAGES + 4) begin
						@(posedge clk);
						if(!irdy) begin
							if(BackCycles > 0)  BackCycles--;
							else begin
								$error("Test #%0d: Encountered unwarranted backpressure.", test);
								$stop;
							end
						end
					end
				end
				while($urandom()%53 == 0) begin
					@(posedge clk);
					if(!irdy) begin
						if(BackCycles > 0)  BackCycles--;
						else begin
							$error("Test #%0d: Encountered unwarranted backpressure.", test);
							$stop;
						end
					end
				end

				void'(std::randomize(dat));
				idat <= dat;
				ivld <= 1;
				Q.push_back(dat);
				forever @(posedge clk) begin
					if(irdy)  break;
					if(BackCycles > 0)  BackCycles--;
					else begin
						$error("Test #%0d: Encountered unwarranted backpressure.", test);
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

			repeat(ROUNDS) begin
				automatic dat_t  exp;

				if($urandom()%173 == 0) begin
					repeat(2 * FEED_STAGES + 5) begin
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
					$error("Test #%0d: Spurious output.", test);
					$stop;
				end
				exp = Q.pop_front();
				assert(odat === exp) else begin
					$error("Test #%0d: Output mismatch: %0x instead of %0x.", test, odat, exp);
					$stop;
				end
				ordy <= 0;
			end

			$display("Test #%0d completed.", test);
			done[test] <= 1;
		end
	end : genTests

endmodule : skid_tb

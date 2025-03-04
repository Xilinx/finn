/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for fifo.sv.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module fifo_tb;

	localparam int unsigned  TEST_COUNT = 14;
	localparam int unsigned  TESTS[TEST_COUNT] = '{ 2, 3, 7, 8, 9, 15, 16, 17, 18, 126, 127, 128, 129, 130 };
	localparam int unsigned  WIDTH = 18;
	typedef logic [WIDTH-1:0]  T;

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

	// Instantiate parallel Tests
	for(genvar  i = 0; i < TEST_COUNT; i++) begin : genTests
		localparam int unsigned  DEPTH = TESTS[i];

		// DUT
		logic  ivld;
		uwire  irdy;
		T      idat;
		uwire  ovld;
		logic  ordy;
		T      odat;
		fifo #(.DEPTH(DEPTH), .WIDTH(WIDTH)) dut (
			.clk, .rst,
			.ivld, .irdy, .idat,
			.ovld, .ordy, .odat
		);


		typedef enum { STOP, HALF, FULL, QUIT } ctrl_e;
		ctrl_e  feed;
		ctrl_e  drain;

		T  Q[$];

		// Phase Driver
		initial begin
			feed  <= STOP;
			drain <= FULL;
			@(posedge clk iff !rst);
			repeat(8) @(posedge clk);

			feed  <= FULL;
			drain <= STOP;
			forever @(posedge clk) begin
				if(!irdy) begin
					int unsigned  size = Q.size;
					$display("Test #%0d [DEPTH=%0d]: Were able to push %0d elements into FIFO.", i, DEPTH, size);
					assert(size >= DEPTH) else begin
						$error("Test #%0d: Unable to queue in %0d elements.", i, DEPTH);
						$stop;
					end
					break;
				end
			end

			feed  <= HALF;
			drain <= FULL;
			forever @(posedge clk) begin
				if(Q.size == 0) begin
					$display("Test #%0d completed.", i);
					done[i] <= 1;
					break;
				end
			end
		end

		// Input Feed
		initial begin
			automatic T  nxt = 0;

			ivld =  0;
			idat = 'x;
			@(posedge clk iff !rst);

			forever @(posedge clk iff !ivld || irdy) begin
				if(ivld && irdy)  Q.push_back(nxt++);

				ivld <=  0;
				idat <= 'x;

				unique case(feed)
				QUIT: break;     // all done
				STOP: continue;  // no feed
				HALF: if($urandom() % 3 != 0)  continue;
				FULL: begin end
				endcase

				// Feed next Value
				ivld <= 1;
				idat <= nxt;
			end
		end

		// Output Checker
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			forever @(posedge clk iff ovld || !ordy) begin

				// Check emitted Output
				if(ovld && ordy) begin
					automatic T  exp;
					assert(Q.size) else begin
						$error("Test #%0d: Spurious output.", i);
						$stop;
					end
					exp = Q.pop_front();
					assert(odat == exp) else begin
						$error("Test #%0d: Received %0p instead of %0p.", i, odat, exp);
						$stop;
					end
				end

				ordy <= 0;
				unique case(drain)
				QUIT: break;     // all done
				STOP: continue;  // no feed
				HALF: if($urandom() % 3 != 0)  continue;
				FULL: begin end
				endcase
				ordy <= 1;

			end
		end

	end : genTests

endmodule : fifo_tb

/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for the fifo_gauge maxcount overflow guard.
 *
 * The gauge FIFO is unbounded (irdy is tied high), so its fill is NOT limited
 * by the nominal depth that COUNT_WIDTH is derived from. A fill beyond
 * 2**COUNT_WIDTH-1 would silently wrap the maxcount port and make the FIFO
 * sizing flow under-report the occupancy it actually observed. The guard in
 * fifo_gauge must turn that into a hard failure instead.
 *
 * This testbench is parameterized by COUNT_WIDTH and OVERFLOW so that a single
 * source covers both directions:
 *  - OVERFLOW=1: fill past the port's capacity, the guard MUST fire ($fatal).
 *    If it does not, we print NO_OVERFLOW_DETECTED and exit normally, which
 *    the calling test treats as a failure.
 *  - OVERFLOW=0: stay within capacity at the production COUNT_WIDTH, the guard
 *    MUST NOT fire. Guards computing the limit as 2**COUNT_WIDTH overflow a
 *    32-bit int context at COUNT_WIDTH=32 and trip on the very first push.
 *****************************************************************************/
`timescale 1ns/1ps
module fifo_gauge_overflow_tb #(
	int unsigned  COUNT_WIDTH = 4,
	bit           OVERFLOW    = 1
);

	localparam int unsigned  W = 8;
	// Overshoot the representable range when testing the guard, and stay
	// comfortably below it (but well above the nominal depth) otherwise.
	localparam int unsigned  N = OVERFLOW? (1<<COUNT_WIDTH) + 4 : 20000;

	// Global Control
	logic  clk = 0;
	always #5ns  clk = !clk;
	logic  rst = 1;

	//-----------------------------------------------------------------------
	// DUT
	logic [W-1:0]  idat = 'x;
	logic  ivld = 0;
	uwire  irdy;

	uwire [W-1:0]  odat;
	uwire  ovld;
	logic  ordy = 0;

	uwire [COUNT_WIDTH-1:0]  count;
	uwire [COUNT_WIDTH-1:0]  maxcount;

	fifo_gauge #(.WIDTH(W), .COUNT_WIDTH(COUNT_WIDTH)) dut (
		.clk, .rst,
		.idat, .ivld, .irdy,
		.odat, .ovld, .ordy,
		.count, .maxcount
	);

	//-----------------------------------------------------------------------
	// Stimulus: push N items and never drain, so the fill is exactly N.
	initial begin
		repeat(2) @(posedge clk);
		rst <= 0;
		@(posedge clk);

		for(int unsigned i = 0; i < N; i++) begin
			idat <= i[W-1:0];
			ivld <= 1;
			@(posedge clk);
		end
		idat <= 'x;
		ivld <=  0;
		repeat(4) @(posedge clk);

		// Only reached if the guard did not fire.
		if(OVERFLOW)  $display("NO_OVERFLOW_DETECTED maxcount=%0d", maxcount);
		else          $display("NO_FALSE_POSITIVE maxcount=%0d", maxcount);
		$finish;
	end

endmodule : fifo_gauge_overflow_tb

// Concrete tops, so the two directions can be elaborated by name without
// relying on simulator-specific parameter-override switches.
module fifo_gauge_overflow_fires_tb;
	fifo_gauge_overflow_tb #(.COUNT_WIDTH(4), .OVERFLOW(1)) tb ();
endmodule : fifo_gauge_overflow_fires_tb

module fifo_gauge_overflow_quiet_tb;
	fifo_gauge_overflow_tb #(.COUNT_WIDTH(32), .OVERFLOW(0)) tb ();
endmodule : fifo_gauge_overflow_quiet_tb

/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Clock domain of a stitched IP with an internal clock generator.
 * @details
 *	A stitched IP with double-pumped nodes derives the clocks of its logic
 *	from its only clock input ap_clk by a Clocking Wizard (MMCM), whose 1x
 *	and 2x outputs are phase-aligned. This module sits between the wizard
 *	and the design logic:
 *	 - In synthesis, it forwards the wizard's outputs as clk and clk2x.
 *	 - In simulation (SYNTHESIS undefined), it holds the wizard in reset,
 *	   so that its unisim MMCM model stays idle, and generates ideal clocks
 *	   instead: clk is ap_clk itself, clk2x rises with every rising edge of
 *	   ap_clk and once more half a (measured) period later. The MMCM model
 *	   would let its outputs lead ap_clk by a picosecond, so that a harness
 *	   driving ap_clk and sampling the outputs right before its rising edge
 *	   would observe values already updated by that edge. The ideal clocks
 *	   lock on the second rising edge of ap_clk after reset.
 *	 - It derives the design reset rst_n. It is asserted while ap_rst_n is
 *	   or while the generator has not locked, and released synchronously to
 *	   clk after RST_DEPTH cycles of a locked clock.
 *	Deviation from the synchronous reset rule: the assertion of rst_n is
 *	asynchronous. ap_rst_n also resets the MMCM, which stops its clocks, so
 *	that a synchronous reset would not take effect. The external interface
 *	gates (clk2x_gen.tcl) rely on rst_n to block handshakes immediately.
 ***************************************************************************/
`timescale 1ps/1ps
module clk2x_domain #(
	parameter  RST_DEPTH = 16
)(
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	input	ap_clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Clocking Wizard
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 gen_resetn RST" *)
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	output	gen_resetn,
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 gen_clk CLK" *)
	input	gen_clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 gen_clk2x CLK" *)
	input	gen_clk2x,
	input	gen_locked,

	// Clocks and reset of the design logic
	// (FREQ_HZ is set in the block design: here, it would be read-only)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET rst_n" *)
	output	clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk2x CLK" *)
	output	clk2x,
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	output	rst_n
);

	//=======================================================================
	// Clocks
	wire  locked;
`ifdef SYNTHESIS
	assign	gen_resetn = ap_rst_n;
	assign	clk    = gen_clk;
	assign	clk2x  = gen_clk2x;
	assign	locked = gen_locked;
`else
	// Simulation only: ideal clocks, see above.
	assign	gen_resetn = 0;
	assign	clk = ap_clk;

	reg  Clk2x = 0;
	reg  Locked = 0;
	time  t_last = 0;
	time  period = 0;
	always @(posedge ap_clk) begin
		if(t_last != 0)  period = $time - t_last;
		t_last = $time;
		Clk2x = 1;
		if(period != 0) begin
			#(period/4)  Clk2x = 0;
			#(period/4)  Clk2x = 1;
			#(period/4)  Clk2x = 0;
		end
	end
	always @(posedge ap_clk or negedge ap_rst_n) begin
		if(!ap_rst_n)  Locked <= 0;
		else if(period != 0)  Locked <= 1;
	end
	assign	clk2x  = Clk2x;
	assign	locked = Locked;
`endif

	//=======================================================================
	// Design reset: asynchronous assertion, synchronous release
	wire  arst = !ap_rst_n || !locked;
	(* ASYNC_REG = "TRUE" *)
	reg [RST_DEPTH-1:0]  Sync = 0;
	always @(posedge clk or posedge arst) begin
		if(arst)  Sync <= 0;
		else      Sync <= { Sync[RST_DEPTH-2:0], 1'b1 };
	end
	assign	rst_n = Sync[RST_DEPTH-1];

endmodule

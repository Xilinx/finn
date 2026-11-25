/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Clock domain of a stitched IP with an internal clock generator.
 * @details
 *	A stitched IP with double-pumped nodes derives the clocks of its logic
 *	from its only clock input ap_clk by an MMCM, whose 1x and 2x outputs are
 *	phase-aligned with ap_clk:
 *	 - In synthesis, it instantiates the MMCM primitive of the device family
 *	   (FAMILY: MMCME2_ADV, MMCME3_ADV, MMCME4_ADV or MMCME5) with the
 *	   settings clk2x_gen.tcl computes (M, D, O1, O2). The feedback runs
 *	   through a BUFG, so that the outputs are phase-aligned with ap_clk, and
 *	   both outputs leave through BUFGs as clk and clk2x.
 *	 - In simulation (SYNTHESIS undefined), there is no MMCM. The module
 *	   generates ideal clocks instead: clk is ap_clk itself, clk2x rises with every rising edge of
 *	   ap_clk and once more half a (measured) period later. The MMCM model
 *	   would let its outputs lead ap_clk by a picosecond, so that a harness
 *	   driving ap_clk and sampling the outputs right before its rising edge
 *	   would observe values already updated by that edge. The ideal clocks
 *	   lock on the second rising edge of ap_clk after reset.
 *	 - It derives the design reset rst_n. It is asserted while ap_rst_n is
 *	   or while the MMCM has not locked, and released synchronously to
 *	   clk after RST_DEPTH cycles of a locked clock.
 *	Deviation from the synchronous reset rule: the assertion of rst_n is
 *	asynchronous. ap_rst_n also resets the MMCM, which stops its clocks, so
 *	that a synchronous reset would not take effect. The external interface
 *	gates (clk2x_gen.tcl) rely on rst_n to block handshakes immediately.
 ***************************************************************************/
`timescale 1ps/1ps
module clk2x_domain #(
	parameter  FAMILY = "ULTRASCALE_PLUS",	// "7SERIES", "ULTRASCALE", "ULTRASCALE_PLUS", "VERSAL"
	parameter  CLKIN_PERIOD_PS = 10000,	// period of ap_clk
	parameter  M  = 12,	// MMCM feedback multiplier
	parameter  D  = 1,	// MMCM input divider
	parameter  O1 = 12,	// MMCM output divider of clk
	parameter  O2 = 6,	// MMCM output divider of clk2x
	parameter  RST_DEPTH = 16
)(
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	input	ap_clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

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
	localparam real  CLKIN_PERIOD = CLKIN_PERIOD_PS / 1000.0;
	wire  mmcm_rst = !ap_rst_n;
	wire  fb_out;
	wire  fb_in;
	wire  clk_out;
	wire  clk2x_out;

	if(FAMILY == "VERSAL") begin : genMmcm5
		MMCME5 #(
			.CLKIN1_PERIOD(CLKIN_PERIOD),
			.CLKFBOUT_MULT(M), .DIVCLK_DIVIDE(D),
			.CLKOUT0_DIVIDE(O1), .CLKOUT1_DIVIDE(O2),
			.COMPENSATION("AUTO")
		) mmcm (
			.CLKIN1(ap_clk), .CLKIN2(1'b0), .CLKINSEL(1'b1),
			.CLKFBIN(fb_in), .CLKFBOUT(fb_out),
			.CLKOUT0(clk_out), .CLKOUT1(clk2x_out),
			.CLKOUT2(), .CLKOUT3(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
			.LOCKED(locked), .LOCKED_FB(), .LOCKED1_DESKEW(), .LOCKED2_DESKEW(),
			.CLKIN1_DESKEW(1'b0), .CLKFB1_DESKEW(1'b0), .CLKIN2_DESKEW(1'b0), .CLKFB2_DESKEW(1'b0),
			.CLKFBSTOPPED(), .CLKINSTOPPED(),
			.DADDR(7'h00), .DCLK(1'b0), .DEN(1'b0), .DI(16'h0000), .DWE(1'b0), .DO(), .DRDY(),
			.PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(),
			.PWRDWN(1'b0), .RST(mmcm_rst)
		);
	end : genMmcm5
	else if(FAMILY == "7SERIES") begin : genMmcm2
		MMCME2_ADV #(
			.CLKIN1_PERIOD(CLKIN_PERIOD),
			.CLKFBOUT_MULT_F(M), .DIVCLK_DIVIDE(D),
			.CLKOUT0_DIVIDE_F(O1), .CLKOUT1_DIVIDE(O2),
			.COMPENSATION("ZHOLD")
		) mmcm (
			.CLKIN1(ap_clk), .CLKIN2(1'b0), .CLKINSEL(1'b1),
			.CLKFBIN(fb_in), .CLKFBOUT(fb_out), .CLKFBOUTB(),
			.CLKOUT0(clk_out), .CLKOUT0B(), .CLKOUT1(clk2x_out), .CLKOUT1B(),
			.CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
			.LOCKED(locked), .CLKFBSTOPPED(), .CLKINSTOPPED(),
			.DADDR(7'h00), .DCLK(1'b0), .DEN(1'b0), .DI(16'h0000), .DWE(1'b0), .DO(), .DRDY(),
			.PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(),
			.PWRDWN(1'b0), .RST(mmcm_rst)
		);
	end : genMmcm2
	else if(FAMILY == "ULTRASCALE") begin : genMmcm3
		MMCME3_ADV #(
			.CLKIN1_PERIOD(CLKIN_PERIOD),
			.CLKFBOUT_MULT_F(M), .DIVCLK_DIVIDE(D),
			.CLKOUT0_DIVIDE_F(O1), .CLKOUT1_DIVIDE(O2),
			.COMPENSATION("AUTO")
		) mmcm (
			.CLKIN1(ap_clk), .CLKIN2(1'b0), .CLKINSEL(1'b1),
			.CLKFBIN(fb_in), .CLKFBOUT(fb_out), .CLKFBOUTB(),
			.CLKOUT0(clk_out), .CLKOUT0B(), .CLKOUT1(clk2x_out), .CLKOUT1B(),
			.CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
			.LOCKED(locked), .CLKFBSTOPPED(), .CLKINSTOPPED(),
			.CDDCREQ(1'b0), .CDDCDONE(),
			.DADDR(7'h00), .DCLK(1'b0), .DEN(1'b0), .DI(16'h0000), .DWE(1'b0), .DO(), .DRDY(),
			.PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(),
			.PWRDWN(1'b0), .RST(mmcm_rst)
		);
	end : genMmcm3
	else begin : genMmcm4
		MMCME4_ADV #(
			.CLKIN1_PERIOD(CLKIN_PERIOD),
			.CLKFBOUT_MULT_F(M), .DIVCLK_DIVIDE(D),
			.CLKOUT0_DIVIDE_F(O1), .CLKOUT1_DIVIDE(O2),
			.COMPENSATION("AUTO")
		) mmcm (
			.CLKIN1(ap_clk), .CLKIN2(1'b0), .CLKINSEL(1'b1),
			.CLKFBIN(fb_in), .CLKFBOUT(fb_out), .CLKFBOUTB(),
			.CLKOUT0(clk_out), .CLKOUT0B(), .CLKOUT1(clk2x_out), .CLKOUT1B(),
			.CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
			.LOCKED(locked), .CLKFBSTOPPED(), .CLKINSTOPPED(),
			.CDDCREQ(1'b0), .CDDCDONE(),
			.DADDR(7'h00), .DCLK(1'b0), .DEN(1'b0), .DI(16'h0000), .DWE(1'b0), .DO(), .DRDY(),
			.PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(),
			.PWRDWN(1'b0), .RST(mmcm_rst)
		);
	end : genMmcm4

	// Feedback through a global buffer: phase alignment with ap_clk
	BUFG fb_buf (.I(fb_out), .O(fb_in));
	BUFG clk_buf (.I(clk_out), .O(clk));
	BUFG clk2x_buf (.I(clk2x_out), .O(clk2x));
`else
	// Simulation only: ideal clocks, see above.
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

/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief
 *	Tap into a forwarded stream with a customizable repetition of values on
 *	the tapped output.
 *****************************************************************************/

module stream_tap #(
	int unsigned  DATA_WIDTH,
	int unsigned  TAP_REP = 1
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy,

	output	logic [DATA_WIDTH-1:0]  tdat,
	output	logic  tvld,
	input	logic  trdy
);

	localparam int unsigned  CNT_BITS = 1 + $clog2(TAP_REP);
	typedef logic signed [CNT_BITS-1:0]  cnt_t;

	// Input Sidestep Register Stage
	logic [DATA_WIDTH-1:0]  A = 'x;
	logic  AVld = 0;

	// Output Register Stage
	logic [DATA_WIDTH-1:0]  B = 'x;
	logic  OVld = 0;
	cnt_t  TCnt = '{ CNT_BITS-1: 0, default: 'x };
	logic  TLst = 'x;

	// Output Wiring
	assign	irdy = !AVld;
	assign	odat = B;
	assign	ovld = OVld;
	assign	tdat = B;
	assign	tvld = TCnt[$left(TCnt)];

	uwire  bload = (ordy || !ovld) && ((trdy && TLst) || !tvld);
	always_ff @(posedge clk) begin
		if(rst) begin
			A <= 'x;
			AVld <= 0;

			B <= 'x;
			OVld <= 0;
			TCnt <= '{ CNT_BITS-1: 0, default: 'x };
			TLst <= 'x;
		end
		else begin
			automatic logic  iavl = AVld || ivld;

			// A Input Register Control
			if(irdy)  A <= idat;
			AVld <= iavl && !bload;

			// B Output Register Control
			if(bload) begin
				B <= AVld? A : idat;
				OVld <= iavl;
				TCnt <= iavl? -TAP_REP : '{ CNT_BITS-1: 0, default: 'x };
				TLst <= iavl? TAP_REP == 1 : 'x;
			end
			else begin
				automatic logic  tick = tvld && trdy;
				OVld <= OVld && !ordy;
				TCnt <= TCnt + tick;
				TLst <= (TCnt == cnt_t'(-2)) && tick;
			end
		end
	end

endmodule : stream_tap

/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Skid buffer with optional feed stages to ease long-distance routing.
 * @todo
 *	Offer knob for increasing buffer elasticity at the cost of allowable
 *	number of feed stages.
 ***************************************************************************/

module axis_skid_rtl #(
	int unsigned  DATA_WIDTH,
	int unsigned  FEED_STAGES = 0
)(
	input	logic  aclk,
	input	logic  aresetn,

	input	logic [DATA_WIDTH-1:0]  s_axis_tdata,
	input	logic  s_axis_tvalid,
	output	logic  s_axis_tready,

	output	logic [DATA_WIDTH-1:0]  m_axis_tdata,
	output	logic  m_axis_tvalid,
	input	logic  m_axis_tready
);

	typedef logic [DATA_WIDTH-1:0]  dat_t;

	uwire  aload;
	uwire dat_t  adat;
	uwire [3:0]  aptr;
	uwire  bvld;
	uwire  bload;
	if(FEED_STAGES == 0) begin : genNoFeedStages

		// Elasticity Control Logic
		logic [1:0]  AVld = '0;
		logic  ARdy = 1;	// = !AVld[1]
		assign	s_axis_tready = ARdy;
		assign	bvld = |AVld;

		always_ff @(posedge aclk) begin
			if(~aresetn) begin
				AVld <= '0;
				ARdy <= 1;
			end
			else begin
				automatic logic  ardy = !AVld || bload;
				AVld <= '{ !ardy, AVld[1]? AVld[0] : s_axis_tvalid };
				ARdy <= ardy;
			end
		end
		assign	aload = s_axis_tready;
		assign	adat = s_axis_tdata;
		assign	aptr = { 3'b000, AVld[1] };

	end : genNoFeedStages
	else begin : genFeedStages

		//- Allow up to 7 plain-forward FEED_STAGES
		initial begin
			if(FEED_STAGES > 7) begin
				$error("%m: Requested %0d FEED_STAGES exceeds support for up to 7.", FEED_STAGES);
				$finish;
			end
		end

		// Dumb input stages to ease long-distance routing
		uwire  ardy;
		if(1) begin : blkInputFeed
			dat_t  IDat[FEED_STAGES] = '{ default: 'x };
			logic  IVld[FEED_STAGES] = '{ default: 0 };
			logic  IRdy[FEED_STAGES] = '{ default: 1 };
			always_ff @(posedge aclk) begin
				if(~aresetn) begin
					IDat <= '{ default: 'x };
					IVld <= '{ default: 0 };
					IRdy <= '{ default: 1 };
				end
				else begin
					for(int unsigned  i = 0; i < FEED_STAGES-1; i++) begin
						IDat[i] <= IDat[i+1];
						IVld[i] <= IVld[i+1];
						IRdy[i] <= IRdy[i+1];
					end
					IDat[FEED_STAGES-1] <= s_axis_tdata;
					IVld[FEED_STAGES-1] <= s_axis_tvalid && s_axis_tready;
					IRdy[FEED_STAGES-1] <= ardy;
				end
			end
			assign	aload = IVld[0];
			assign	adat = IDat[0];
			assign	s_axis_tready = IRdy[0];
		end : blkInputFeed

		// Elasticity Control Logic
		logic signed [$clog2(2*FEED_STAGES+2):0]  APtr = '1;
		assign	ardy = APtr < 1;
		assign	bvld = !APtr[$left(APtr)];

		always_ff @(posedge aclk) begin
			if(~aresetn)  APtr <= '1;
			else     APtr <= APtr + $signed((aload == (bload && bvld))? 0 : aload? 1 : -1);
		end
		assign	aptr = $unsigned(APtr[$left(APtr)-1:0]);

	end : genFeedStages

	//-----------------------------------------------------------------------
	// Buffer Memory: SRL:2+2*FEED_STAGES + Reg (no reset)

	// Elastic SRL
	uwire dat_t  bdat;
	for(genvar  i = 0; i < DATA_WIDTH; i++) begin : genSRL
		SRL16E srl (
			.CLK(aclk),
			.CE(aload),
			.D(adat[i]),
			.A3(aptr[3]), .A2(aptr[2]), .A1(aptr[1]), .A0(aptr[0]),
			.Q(bdat[i])
		);
	end : genSRL

	// Output Register
	logic  BVld = 0;
	assign	m_axis_tvalid = BVld;
	assign	bload = !BVld || m_axis_tready;

	always_ff @(posedge aclk) begin
		if(~aresetn)  BVld <= 0;
		else     BVld <= bvld || !bload;
	end

	(* EXTRACT_ENABLE = "true" *)
	dat_t  B = 'x;
	assign	m_axis_tdata = B;

	always_ff @(posedge aclk) begin
		if(bload)  B <= bdat;
	end

endmodule : axis_skid_rtl
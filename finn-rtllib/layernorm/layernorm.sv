/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Implements a streaming LayerNorm across N fp32 inputs.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @description
 *	The implemented LayerNorm processes input at the specfied SIMD data
 *	parallelism. The main datapath through the bypass buffer and the
 *	application of the normalization unfold to the corresponding width.
 *	The statistics branch comprises an additive reduction to derive the
 *	the appropriate total normalization statistics.
 *
 *	The LayerNorm is composed of two consecutive normalization diamonds
 *	resembling this structure:
 *
 *		         <-N+Stat Latency->         «──
 *		«──      ┌────────────────┐        ┌───┐
 *		─┬──────►{||||||||||||||||}───────►{-/×}──
 *		 │       └────────────────┘        └───┘
 *		 │                                   ▲
 *		 │  ┌─────┐   ┌─────┐   ┌───────┐    │
 *		 └─►{[()²]}──►{1/N ∑├──►│[1/√()]├────┘
 *		    └─────┘   └─────┘   └───────┘
 *
 *	The overall flow control forwards backpressure through the data buffer.
 *	The statistics pipeline is running freely. Input accepted by the buffer
 *	can always be processed by the statistics pipeline.
 ***************************************************************************/

module layernorm #(
	int unsigned  N,
	int unsigned  SIMD,
	bit  FORCE_BEHAVIORAL = 0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// (Parallel) Input Stream
	input	logic [SIMD-1:0][31:0]  xdat,
	input	logic  xvld,
	output	logic  xrdy,

	// (Parallel) Output Stream
	output	logic [SIMD-1:0][31:0]  ydat,
	output	logic  yvld,
	input	logic  yrdy
);

	localparam int unsigned  NN = N / SIMD;
	initial begin
		if(N%SIMD != 0) begin
			$error("%m: SIMD(%0d) must divide N(%0d).", SIMD, N);
			$finish;
		end
		if(NN <= 12) begin
			$error("%m: N/SIMD must be larger than 12 for rsqrt throughput.");
			$finish;
		end
	end

	typedef logic [31:0]  fp32;
	typedef fp32 [SIMD-1:0] vfp32;
	typedef struct {
		fp32   dat;
		logic  vld;
	} edge_t;
	typedef struct {
		vfp32  dat;
		logic  vld;
		logic  rdy;
	} vedge_t;

	//=======================================================================
	// Build Normalization Diamonds

	// Connectivity: #0 -> Mean Shift -> #1 -> Variance Scaling -> #2
	uwire vedge_t  vedge[3];
	assign	vedge[0].dat = xdat;
	assign	vedge[0].vld = xvld;
	assign	xrdy = vedge[0].rdy;
	assign	ydat = vedge[2].dat;
	assign	yvld = vedge[2].vld;
	assign	vedge[2].rdy = yrdy;

	for(genvar  step = 0; step < 2; step++) begin : genNormDiamonds

		localparam int unsigned  STATISTICS_LATENCY =
			// SIMD adder tree + accumulation + decouple
			$clog2(SIMD) * 2   +     4        +    3 +
			// Variance: *1/N + rsqrt
			step   *    (  3  + 14  );
		localparam int unsigned  VALUE_QUEUE_LEN = NN + STATISTICS_LATENCY;
		localparam int unsigned  STATS_QUEUE_LEN = 2 + (VALUE_QUEUE_LEN-1)/NN;

		//-------------------------------------------------------------------
		// Value bypass Queue
		uwire vedge_t  bypass;
		queue #(.DATA_WIDTH(SIMD*32), .ELASTICITY(VALUE_QUEUE_LEN)) bypass_queue (
			.clk, .rst,
			.idat(vedge[step].dat), .ivld(vedge[step].vld), .irdy(vedge[step].rdy),
			.odat(bypass     .dat), .ovld(bypass     .vld), .ordy(bypass     .rdy)
		);

		//-------------------------------------------------------------------
		// Free-running Statistics Queue
		uwire edge_t  norm;
		if(1) begin : blkStatistics
			//	Input pacing solely by bypass queue
			uwire  avld = vedge[step].vld && vedge[step].rdy;
			uwire vfp32  adat = vedge[step].dat;

			//- Input Aggregation -------

			// Cross-SIMD Reduction tree
			uwire edge_t  part_sum;
			if(1) begin : blkReduceSIMD
				uwire edge_t  tree[2*SIMD-1];

				// Input Refinement for Leaf Feed
				for(genvar  i = 0; i < SIMD; i++) begin : genLeaves
					uwire edge_t  leaf;
					case(step)
					0: /* Mean: straight to sum */ begin
						assign	leaf = '{ vld: avld, dat: adat[i] };
					end
					1: /* Var: square to sum */ begin
						binopf #(.OP("MUL"), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) node (
							.clk, .rst,
							.a(adat[i]),  .avld(avld),
							.b(adat[i]),  .bload(1'b1),
							.r(leaf.dat), .rvld(leaf.vld)
						);
					end
					endcase
					assign	tree[SIMD-1+i] = leaf;
				end : genLeaves

				// Balancing edge delays in trees with incomplete leaf level
				typedef bit edge_delays_t[2*SIMD-1];
				function edge_delays_t INIT_EDGE_DELAYS();
					localparam int unsigned  LEVELS = 1+$clog2(SIMD);
					automatic edge_delays_t  d = '{ default: 0 };
					// Put delay onto leaves that are not on last level
					for(int unsigned  i = SIMD-1; i < 2*SIMD-1; i++) begin
						if($clog2(i+2) == LEVELS)  break;
						d[i] = 1;
					end
					// Move delay shared between children to their parent
					for(int unsigned  i = SIMD-1; i > 0; i--) begin
						if(d[2*i+1]) begin
							d[2*i+1] = 0;
							d[2*i+2] = 0;
							d[i] = 1;
						end
					end
					return  d;
				endfunction : INIT_EDGE_DELAYS
				localparam edge_delays_t  EDGE_DELAYS = INIT_EDGE_DELAYS();

				// Adder Tree
				for(genvar  i = 0; i < SIMD-1; i++) begin : genNodes
					binopf #(
						.OP("ADD"),
						.A_MATCH_OP_DELAY(EDGE_DELAYS[2*i+2]),
						.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
					) node (
						.clk, .rst,
						.r(tree[i]    .dat), .rvld(tree[i]    .vld),
						.b(tree[2*i+1].dat), .bload(1'b1),
						.a(tree[2*i+2].dat), .avld(tree[2*i+2].vld)
					);
				end : genNodes

				assign	part_sum = tree[0];
			end : blkReduceSIMD

			// Scaled Accumulation of parital Sums
			uwire edge_t  total;
			if(1) begin : blkAccumulate

				// Identify last Input Transaction
				uwire  alst;
				if(NN == 1)  assign  alst = 1;
				else begin
					logic signed [$clog2(NN-1):0]  Cnt = NN-2; // NN-2, ..., 1, 0, -1
					always_ff @(posedge clk) begin
						if(rst)  Cnt <= NN-2;
						else     Cnt <= Cnt + (!part_sum.vld? 0 : !alst? -1 : NN-1);
					end
					assign	alst = Cnt[$left(Cnt)];
				end

				// Scaled Accumulation
				accuf #(.SCALE(1.0/N), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) accu (
					.clk, .rst,
					.a(part_sum.dat), .avld(part_sum.vld), .alst,
					.s(total.dat),    .svld(total.vld)
				);

			end : blkAccumulate

			// Output Refinement for norm Extraction
			case(step)
			0: /* Mean: straight out */ begin
				assign	norm = total;
			end
			1: /* Var: inverse square root */ begin
				uwire  vrdy;
				rsqrtf #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) vari_rsqurt (
					.clk, .rst,
					.x(total.dat), .xvld(total.vld), .xrdy(vrdy),
					.r(norm .dat), .rvld(norm .vld)
				);
				always_ff @(posedge clk) begin
					assert(rst || !total.vld || vrdy) else begin
						$error("%m Overrunning rsqrt computation.");
						$stop;
					end
				end
			end
			endcase

		end : blkStatistics

		//-------------------------------------------------------------------
		// Apply Normalization
		if(1) begin : blkApply

			// Statistics Queue catching all possible Computations in Flight
			uwire edge_t  norm0;
			uwire  norm0_rdy;
			if(1) begin : blkMeanCatcher
				uwire  norm_rdy;
				queue #(.DATA_WIDTH(32), .ELASTICITY(STATS_QUEUE_LEN)) catcher (
					.clk, .rst,
					.idat(norm .dat), .ivld(norm .vld), .irdy(norm_rdy),
					.odat(norm0.dat), .ovld(norm0.vld), .ordy(norm0_rdy)
				);
				always_ff @(posedge clk) begin
					assert(rst || !norm.vld || norm_rdy) else begin
						$error("%m: Overrunning statistics queue.");
						$stop;
					end
				end
			end : blkMeanCatcher

			// Free-Running Normalization Operator bracketed by credit-based Flow Control
			localparam int unsigned  CREDIT = 7;
			logic signed [$clog2(CREDIT):0]  Credit = CREDIT-1; // CREDIT-1, ..., 1, 0, -1
			uwire  have_cap = !Credit[$left(Credit)];
			uwire  issue;
			uwire  settle;
			always @(posedge clk) begin
				if(rst)  Credit <= 6;
				else     Credit <= Credit + (issue == settle? 0 : settle? 1 : -1);
			end

			logic signed [$clog2(NN-1):0]  Cnt = 0;	// [-NN,] -NN+1, ..., -1, 0
			assign	norm0_rdy = !Cnt[$left(Cnt)];
			assign	issue = have_cap && (norm0.vld || Cnt[$left(Cnt)]);
			uwire  bload = norm0.vld && norm0_rdy;
			always @(posedge clk) begin
				if(rst)  Cnt <= 0;
				else     Cnt <= Cnt + (bload? -NN : 0) + issue;
			end
			always_ff @(posedge clk) begin
				assert(rst || bypass.vld || !issue) else begin
					$error("%m: Drained bypass.");
					$stop;
				end
			end
			assign	bypass.rdy = issue;

			uwire vfp32  rdat;
			uwire  rvld;
			for(genvar  i = 0; i < SIMD; i++) begin : genOps
				uwire  rvld0;
				binopf #(.OP(step? "MUL" : "SUB"), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) op (
					.clk, .rst,
					.a(bypass.dat[i]), .avld(issue),
					.b(norm0.dat), .bload,
					.r(rdat[i]), .rvld(rvld0)
				);
				if(i == 0)  assign  rvld = rvld0;
			end : genOps

			// Output Queue
			uwire  rrdy;
			queue #(.DATA_WIDTH(SIMD * 32), .ELASTICITY(CREDIT)) decouple (
				.clk, .rst,
				.idat(rdat), .ivld(rvld), .irdy(rrdy),
				.odat(vedge[step+1].dat), .ovld(vedge[step+1].vld), .ordy(vedge[step+1].rdy)
			);
			always_ff @(posedge clk) begin
				assert(rst || !rvld || rrdy) else begin
					$error("%m: Overruning normalization output.");
					$stop;
				end
			end
			assign	settle = vedge[step+1].vld && vedge[step+1].rdy;

		end : blkApply

	end : genNormDiamonds

endmodule : layernorm

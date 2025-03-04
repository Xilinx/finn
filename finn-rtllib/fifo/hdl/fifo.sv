/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FIFO Implementation optimized for AMD PL Fabric.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module fifo #(
	int unsigned  DEPTH,  // Minimum FIFO Depth
	int unsigned  WIDTH,  // Element Bitwidth
	parameter     RAM_STYLE = "auto"
)(
	input	logic  clk,
	input	logic  rst,

	input	logic  ivld,
	output	logic  irdy,
	input	logic [WIDTH-1:0]  idat,

	output	logic  ovld,
	input	logic  ordy,
	output	logic [WIDTH-1:0]  odat
);

	localparam  STYLE = (RAM_STYLE != "auto") || (DEPTH > 64)? RAM_STYLE : "shift";
	typedef logic [WIDTH-1:0]  data_t;

	//-----------------------------------------------------------------------
	if(DEPTH < 2) begin : genTooSmall
		// Error out on impossible FIFO size
		initial begin
			$error("%m: FIFO DEPTH=%0d is smaller than 2.", DEPTH);
			$finish;
		end
	end :genTooSmall
	else if(DEPTH == 2) begin : genMinimal
		// Minimal FIFO
		//	Isolates control domains with all registered output with minimum
		//	capacity of two data registers.
		//
		//                      ┌─────────┐  ┌───┐
		//                      │  ┌────┐ └──┤1  │            ┌────┐
		//                      │  │    │    │   ├──────┐     │    │
		//  idat ───────────────┴──┤  A ├────┤0  │      └─────┤  B ├───► odat
		//                         │    │    └─┬─┘            │    │
		//                         │    │      │      ┌──┐    │    │
		//                         │    │      │ ┌────┤≥1├─┬──┤E   │
		//          ┌───────┐      └────┘      │ │ ┌─o│  │ │  └────┘
		//          │ ┌──┐  └──────────────┐   │ │ │  └──┘ │
		//          └─┤ &│   ┌──┐  ┌────┐  │   │ │ │  ┌──┐ │  ┌────┐
		//  ivld ──┬─o│  ├───┤≥1├──┤ Rdy├─┬┴───┴─┼─┼─o│≥1├─┼──┤ Vld├─┬─► ovld
		//         │  └──┘ ┌─┤  │  │    │ │ ┌────┼─┼──┤  │ └──┤E   │ │
		//         │       │ └──┘  └────┘ │ │    │ │  └──┘    └────┘ │
		//         │       └──────────────┼─┼────┤ └─────────────────┘
		//         └──────────────────────┼─┘    │
		//  irdy ◄────────────────────────┘      └────────────────────── ordy
		//

		data_t  A   = 'x;
		logic   Rdy =  1;
		data_t  B   = 'x;
		logic   Vld =  0;

		always_ff @(posedge clk) begin
			if(rst) begin
				A   <= 'x;
				Rdy <=  1;
				B   <= 'x;
				Vld <=  0;
			end
			else begin
				if(Rdy)  A <= idat;
				Rdy <= ordy || (Rdy && !ivld);
				if(ordy || !Vld) begin
					B   <= Rdy? idat : A;
					Vld <= ivld || !Rdy;
				end
			end
		end

		assign	irdy = Rdy;
		assign	ovld = Vld;
		assign	odat = B;

	end : genMinimal
	else if(STYLE == "shift") begin : genShift // SRL-Based FIFO
		// Link between A (SRL) and B (Output Reg)
		uwire data_t  adat;
		uwire  avld;
		uwire  brdy;

		// A: SRL Shift Register with DEPTH-1 Places
		if(1) begin : blkA
			// Tap identified by Ptr:
			//   Ptr:   -1   | 0, 1, 2, ..., DEPTH-2
			//         empty | (Ptr+1) entries, ^full
			localparam int unsigned  PTR_BITS = 1 + $clog2(DEPTH-1);
			data_t  A[DEPTH-1] = '{ default: 'x };
			logic signed [PTR_BITS-1:0]  Ptr = -1;
			logic  Rdy =  1;

			uwire  inc = ivld && irdy;	// input
			uwire  dec = avld && brdy;	// output
			uwire  lst = /* Ptr[PTR_BITS-2:0] == DEPTH-3 */	// last possible input
				// ignore zero bits of (DEPTH-3) in detection:
				!{Ptr[PTR_BITS-1], ~Ptr[PTR_BITS-2:0] & (DEPTH-3)};
			always_ff @(posedge clk) begin
				// not resettable, contents invalidated by Ptr reset
				if(inc)  A <= { idat, A[0:DEPTH-3] };
			end
			always_ff @(posedge clk) begin
				if(rst) begin
					Ptr <= -1;
					Rdy <=  1;
				end
				else if(dec != inc) begin
					Ptr <= Ptr + $signed(inc? 1 : -1);
					Rdy <= !inc || !lst;
				end
				// Absorbed Alternative: high fanouts for both dec and inc
				//else begin
				//	Ptr <= Ptr + $signed((dec == inc)? 0 : dec? -1 : 1);
				//	Rdy <= dec || (Rdy && (!inc || !lst))
				//end
			end

			assign	avld = !Ptr[PTR_BITS-1];
			assign	adat = A[Ptr[PTR_BITS-2:0]];

			assign	irdy = Rdy;
		end : blkA

		// B: Output Register after SRL
		if(1) begin : blkB
			data_t  B   = 'x;
			logic   Vld =  0;
			always_ff @(posedge clk) begin
				if(rst) begin
					B   <= 'x;
					Vld <=  0;
				end
				else if(brdy) begin
					B   <= adat;
					Vld <= avld;
				end
			end

			assign	brdy = ordy || !ovld;

			assign	ovld = Vld;
			assign	odat = B;
		end : blkB

	end : genShift
	else begin : genMemBased

		// Bulk Memory Storage
		localparam int unsigned  A_BITS = $clog2(DEPTH - 8);
		typedef logic [A_BITS:0]  ptr_t;  // pointers with extra generational bit
		(* RAM_STYLE = STYLE *)
		data_t  Mem[2**A_BITS];
		ptr_t  WPtr = 0;
		ptr_t  RPtr = 0;
		logic  Rdy  = 1;

		assign	irdy = Rdy;

		localparam int unsigned  RD_LATENCY = (STYLE == "block") || (STYLE == "ultra")? 2 : 1;
		uwire  we = ivld && irdy;
		uwire  re;
		data_t  RdDat[RD_LATENCY];
		logic   RdVld[RD_LATENCY] = '{ default: 0 };

		always_ff @(posedge clk) begin
			// Non-resettable OCRAM
			if(we)  Mem[WPtr[A_BITS-1:0]] <= idat;
			RdDat[0] <= Mem[RPtr[A_BITS-1:0]];
			for(int unsigned  i = 1; i < RD_LATENCY; i++)  RdDat[i] <= RdDat[i-1];
		end
		always_ff @(posedge clk) begin
			if(rst) begin
				WPtr <= 0;
				RPtr <= 0;
				Rdy  <= 1;

				RdVld <= '{ default: 0 };
			end
			else begin
				automatic type(WPtr)  wptr = WPtr + we;
				automatic type(RPtr)  rptr = RPtr + re;
				WPtr <= wptr;
				RPtr <= rptr;
				Rdy <= $signed(wptr - rptr) >= 0;

				RdVld[0] <= re;
				for(int unsigned  i = 1; i < RD_LATENCY; i++)  RdVld[i] <= RdVld[i-1];
			end
		end

		// Output Decoupling through SRL16 + Register
		uwire  push = RdVld[RD_LATENCY-1];
		uwire  pop;

		data_t              A[16];
		logic signed [4:0]  Ptr = -1;
		always_ff @(posedge clk) begin
			if(push)  A <= { RdDat[RD_LATENCY-1], A[0:14] };
		end
		always_ff @(posedge clk) begin
			if(rst)               Ptr <= -1;
			else if(push != pop)  Ptr <= Ptr + $signed(push? 1 : -1);
		end

		data_t  B   = 'x;
		logic   Vld =  0;
		uwire   bload = !Vld || ordy;
		always_ff @(posedge clk) begin
			if(rst) begin
				B   <= 'x;
				Vld <=  0;
			end
			else if(bload) begin
				B   <= A[Ptr[3:0]];
				Vld <= !Ptr[4];
			end
		end
		assign	pop = bload && !Ptr[4];
		assign	re  = ($signed(RPtr - WPtr) < 0) && (Ptr[4] == Ptr[3]);

		assign	odat = B;
		assign	ovld = Vld;

	end : genMemBased

endmodule : fifo

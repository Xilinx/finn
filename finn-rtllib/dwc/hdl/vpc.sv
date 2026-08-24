/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Vector pack converter between different lane parallelism.
 *
 * @description
 *  Converts a stream of PI elements/beat to PO elements/beat for vectors
 *  of N elements. Each N-element vector is transported independently
 *  (no cross-vector packing) using ceil(N/PI) input beats and ceil(N/PO)
 *  output beats. Elements are packed in little-endian order; excess lanes
 *  on the last beat are padded with zeros.
 *
 *  Internally, parameters are normalized using gcd(PI,PO) to reduce
 *  buffer depth and counter widths.
 *
 *  Internal buffering ensures full-rate operation of the narrower
 *  interface without a combinatorial path from ordy to irdy.
 ***************************************************************************/

module vpc #(
	int unsigned  W,	// element bit width
	int unsigned  N,	// total elements per vector
	int unsigned  PI,	// input elements per beat
	int unsigned  PO,	// output elements per beat
	bit  RELAX_THROUGHPUT = 0	// allow recovery cycles
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [PI-1:0][W-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [PO-1:0][W-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);
`default_nettype none

	//=======================================================================
	// Parameter Validation
	initial begin
		if((W == 0) || (N == 0) || (PI == 0) || (PO == 0)) begin
			$error("%m: W, N, PI, and PO must all be non-zero.");
			$finish;
		end
	end

	// Extract GCD from in- and output parallelism and map to data path width
	function int unsigned gcd(input int unsigned  a, input int unsigned  b);
		return (b == 0)? a : gcd(b, a % b);
	endfunction
	localparam int unsigned  GCD = gcd(PI, PO);
	localparam int unsigned  W0  = GCD * W;			// normalized element width
	localparam int unsigned  PI0 = PI / GCD;		// normalized input parallelism
	localparam int unsigned  PO0 = PO / GCD;		// normalized output parallelism
	localparam int unsigned  N0  = 1 + (N-1)/GCD;	// normalized vector length

	//=======================================================================
	// Derived Parameters
	localparam int unsigned  TRNI = 1 + (N0-1)/PI0;	// input transactions per vector
	localparam int unsigned  TRNO = 1 + (N0-1)/PO0;	// output transactions per vector

	uwire  itrn = ivld && irdy;
	uwire  otrn = ovld && ordy;
	if((PI0 == 1) && (PO0 == 1)) begin : genWire
		//===============================================================
		// Wire-through: PI0 == PO0 == 1 → no conversion needed.
		assign  odat = idat;
		assign  ovld = ivld;
		assign  irdy = ordy;

	end : genWire
	else if((PO0 == 1) && (N0 <= PI0)) begin : genSer
		//===============================================================
		// Pure serialization: entire vector fits one input beat.
		// Parallel load N0 elements, emit 1/cycle.

		logic [W0-1:0]  Buf[N0] = '{ default: 'x };
		logic signed [$clog2(N0):0]  Cnt = 0;	// -N0, ..., -1 (occupied), 0 (empty)

		if(RELAX_THROUGHPUT) begin : genRelax
			//-----------------------------------------------------------
			// Simple: direct output from Buf[0]. One recovery cycle
			// between loads (N0+1 cycles per N0 elements).

			always_ff @(posedge clk) begin
				if(rst) begin
					Buf <= '{ default: 'x };
					Cnt <= 0;
				end
				else begin
					if(itrn)  foreach(Buf[i])  Buf[i] <= idat[i*GCD +: GCD];
					if(otrn)  for(int unsigned  i = 1; i < N0; i++)  Buf[i-1] <= Buf[i];
					Cnt <= Cnt + (itrn? -N0 : otrn? 1 : 0);
				end
			end

			assign  irdy = !Cnt[$left(Cnt)];
			assign  ovld = Cnt[$left(Cnt)];
			assign  odat = Buf[0];

		end : genRelax
		else begin : genFull
			//-----------------------------------------------------------
			// Full-rate: output side-step register decouples handshakes.
			// No combinatorial paths. Buffer: N0 + 1 elements.

			logic [W0-1:0]  Side = 'x;
			logic  SVld = 0;

			uwire  bypass = !SVld || otrn;
			uwire  shift  = Cnt[$left(Cnt)] && bypass;

			always_ff @(posedge clk) begin
				if(rst) begin
					Buf  <= '{ default: '0 };
					Cnt  <= 0;
					Side <= 'x;
					SVld <= 0;
				end
				else begin
					// Shift: Buf[0] → Side
					if(shift) begin
						Side <= Buf[0];
						SVld <= 1;
						if(N0 > 1)  Buf <= { Buf[1 +: N0-1], W0'('x) };
					end
					else if(otrn)  SVld <= 0;

					// Parallel load
					if(itrn) begin
						if(bypass) begin
							// Element 0 bypasses to Side.
							Side <= idat[0 +: GCD];
							SVld <= 1;
							for(int unsigned  i = 1; i < N0; i++)
								Buf[i-1] <= idat[i*GCD +: GCD];
						end
						else begin
							foreach(Buf[i])  Buf[i] <= idat[i*GCD +: GCD];
						end
					end

					Cnt <= Cnt + (itrn? (bypass? 1-N0 : -N0) : shift? 1 : 0);
				end
			end

			assign  irdy = !Cnt[$left(Cnt)];
			assign  ovld = SVld;
			assign  odat = Side;

		end : genFull

	end : genSer
	else if((PI0 == 1) && (N0 <= PO0)) begin : genDes
		//===============================================================
		// Pure deserialization: entire vector fits one output beat.
		// Shift-register fill of N0 elements, emit PO0 in parallel.

		logic [W0-1:0]  Buf[N0] = '{ default: 'x };
		logic signed [$clog2(N0):0]  Cnt = -N0;	// -N0 (empty), ..., 0 (full)

		if(RELAX_THROUGHPUT) begin : genRelax
			//-----------------------------------------------------------
			// Simple: shift-in fill. One recovery cycle
			// between emit and next fill.

			always_ff @(posedge clk) begin
				if(rst) begin
					Buf <= '{ default: 'x };
					Cnt <= -N0;
				end
				else begin
					if(itrn) begin
						if(N0 > 1)  Buf[0 +: N0-1] <= Buf[1 +: N0-1];
						Buf[N0-1] <= idat[0 +: GCD];
					end
					Cnt <= Cnt + (itrn? 1 : otrn? -N0 : 0);
				end
			end

			assign  irdy = Cnt[$left(Cnt)];
			assign  ovld = !Cnt[$left(Cnt)];

		end : genRelax
		else begin : genFull
			//-----------------------------------------------------------
			// Full-rate: input side-step register decouples handshakes.
			// No combinatorial paths. Buffer: N0 + 1 elements.

			logic [W0-1:0]  Side = 'x;
			logic  SRdy = 1;

			uwire  full = !Cnt[$left(Cnt)];	// Cnt == 0

			always_ff @(posedge clk) begin
				if(rst) begin
					Buf  <= '{ default: 'x };
					Cnt  <= -N0;
					Side <= 'x;
					SRdy <= 1;
				end
				else begin
					if(itrn && otrn) begin
						// Simultaneous accept and drain: shift in first element.
						Buf       <= '{ default: 'x };
						Buf[N0-1] <= idat[0 +: GCD];
					end
					else if(itrn) begin
						if(!full) begin
							if(N0 > 1)  Buf <= { Buf[1 +: N0-1], idat[0 +: GCD] };
							else        Buf[0] <= idat[0 +: GCD];
						end
						else begin
							// Full, downstream stall: park in side-step.
							Side <= idat[0 +: GCD];
							SRdy <= 0;
						end
					end
					else if(otrn) begin
						// Drain. Recover side-step element if present.
						if(!SRdy) begin
							Buf       <= '{ default: 'x };
							Buf[N0-1] <= Side;
							SRdy      <= 1;
						end
					end

					Cnt <= Cnt + ((itrn && !full)? 1 : otrn? ((itrn || !SRdy)? 1-N0 : -N0) : 0);
				end
			end

			assign  irdy = SRdy;
			assign  ovld = full;

		end : genFull

		// Output: N0 valid elements from shift register, zero-padded beyond.
		for(genvar  p = 0; p < PO0; p++) begin : genOdat
			if(p < N0)  assign  odat[p*GCD+:GCD] = Buf[p];
			else        assign  odat[p*GCD+:GCD] = '0;
		end : genOdat

	end : genDes
	else begin : genGeneric
		//===============================================================
		// Full buffered implementation: sustained full-rate operation.

		// Transaction counters only needed for sync when padding differs.
		uwire  nxt;
		uwire  idone;
		uwire  odone;
		if(TRNI * PI0 == TRNO * PO0) begin : genSimple
			assign  nxt   = 0;
			assign  idone = 0;
			assign  odone = 0;
		end : genSimple
		else begin : genTrn
			logic signed [$clog2(TRNI):0]  ITrn = TRNI-1;
			logic signed [$clog2(TRNO):0]  OTrn = TRNO-1;

			uwire signed [$clog2(TRNI):0]  n_itrn = ITrn - itrn;
			uwire signed [$clog2(TRNO):0]  n_otrn = OTrn - otrn;
			assign  nxt = n_itrn[$left(n_itrn)] && n_otrn[$left(n_otrn)];

			always_ff @(posedge clk) begin
				if(rst) begin
					ITrn <= TRNI-1;
					OTrn <= TRNO-1;
				end
				else begin
					ITrn <= nxt? TRNI-1 : n_itrn;
					OTrn <= nxt? TRNO-1 : n_otrn;
				end
			end

			assign  idone = ITrn[$left(ITrn)];
			assign  odone = OTrn[$left(OTrn)];
		end : genTrn

		localparam int unsigned  CAP = PI0 + PO0;
		logic [W0-1:0]  Buf[CAP];
		logic signed [$clog2(CAP):0]  ICap = PO0;	// PO0 (empty), .., 0, .., -PI0; >= 0: room
		logic signed [$clog2(CAP):0]  ORdy = -PO0;	// -PO0 (empty), .., 0, .., PI0; >= 0: ready

		always_ff @(posedge clk) begin
			if(rst) begin
				Buf <= '{ default: 'x };
				ICap <= PO0;
				ORdy <= -PO0;
			end
			else begin
				automatic logic [W0-1:0]  n_buf[CAP];
				automatic logic signed [$clog2(CAP):0]   n_icap;
				automatic logic signed [$clog2(CAP):0]   n_ordy;

				n_buf  = Buf;
				n_icap = ICap;
				n_ordy = ORdy;

				// Phase 1: retire PO0 elements from buffer head.
				if(otrn) begin
					n_buf[0 +: CAP-PO0] = n_buf[PO0 +: CAP-PO0];
					n_icap += PO0;
					n_ordy -= PO0;
				end

				// Phase 2: deposit input at buffer tail (don't-care beyond ORdy).
				if(1) begin
					automatic int unsigned  ofs = ORdy + (otrn? 0 : $signed(PO0));
					for(int unsigned  p = 0; p < PI0; p++)
						if(ofs + p < CAP)  n_buf[ofs + p] = idat[p*GCD +: GCD];
				end
				if(itrn) begin
					n_icap -= PI0;
					n_ordy += PI0;
				end

				// When both input and output vectors are complete, start next vector.
				if(nxt) begin
					n_icap =  PO0;
					n_ordy = -PO0;
				end

				Buf <= n_buf;
				ICap <= n_icap;
				ORdy <= n_ordy;
			end
		end

		assign  irdy = !idone && !ICap[$left(ICap)];
		assign  ovld = !odone && !(ORdy[$left(ORdy)] && !idone);
		for(genvar  p = 0; p < PO0; p++)  assign  odat[p*GCD +: GCD] = Buf[p];

	end : genGeneric

`default_nettype wire
endmodule : vpc

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
	localparam int unsigned  OLAST = N - (TRNO-1)*PO;	// valid output lanes on last beat (1..PO)

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

			if(OLAST == PO)  assign  odat = Buf[0];
			else begin : genOpad
				uwire  olast_beat = &Cnt;
				for(genvar  p = 0; p < PO; p++) begin : genOdat
					if(p < OLAST)  assign  odat[p] = Buf[0][p*W +: W];
					else           assign  odat[p] = olast_beat? '0 : Buf[0][p*W +: W];
				end : genOdat
			end : genOpad

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

			if(OLAST == PO)  assign  odat = Side;
			else begin : genOpad
				uwire  olast_beat = !Cnt[$left(Cnt)];
				for(genvar  p = 0; p < PO; p++) begin : genOdat
					if(p < OLAST)  assign  odat[p] = Side[p*W +: W];
					else           assign  odat[p] = olast_beat? '0 : Side[p*W +: W];
				end : genOdat
			end : genOpad

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
		uwire  olast;
		if(TRNI * PI0 == TRNO * PO0) begin : genSimple
			assign  nxt   = 0;
			assign  idone = 0;
			assign  odone = 0;

			if(OLAST == PO)  assign  olast = 0;
			else begin : genOlast
				logic signed [$clog2((TRNO > 1)? TRNO : 2):0]  OBeat = 1-TRNO;
				assign  olast = !OBeat[$left(OBeat)];
				always_ff @(posedge clk) begin
					if(rst)  OBeat <= 1-TRNO;
					else     OBeat <= OBeat + ((olast && otrn)? -TRNO : 0) + otrn;
				end
			end : genOlast
		end : genSimple
		else begin : genTrn
			localparam int unsigned  ITRN_W = $clog2((TRNI > 2)? TRNI-1 : 2);
			localparam int unsigned  OTRN_W = $clog2((TRNO > 2)? TRNO-1 : 2);
			logic signed [ITRN_W:0]  ITrn = 1-TRNI;	// -TRNI+1, .., 0 (last), 1 (done)
			logic signed [OTRN_W:0]  OTrn = 1-TRNO;	// -TRNO+1, .., 0 (last), 1 (done)

			uwire  ilast = !ITrn[$left(ITrn)] && !ITrn[0];
			assign  olast = !OTrn[$left(OTrn)] && !OTrn[0];
			assign  idone = !ITrn[$left(ITrn)] && ITrn[0];
			assign  odone = !OTrn[$left(OTrn)] && OTrn[0];
			assign  nxt = (idone || (ilast && itrn)) && (odone || (olast && otrn));

			always_ff @(posedge clk) begin
				if(rst) begin
					ITrn <= 1-TRNI;
					OTrn <= 1-TRNO;
				end
				else begin
					ITrn <= ITrn + (nxt? -TRNI : 0) + itrn;
					OTrn <= OTrn + (nxt? -TRNO : 0) + otrn;
				end
			end
		end : genTrn

		localparam int unsigned  CAP  = PI0 + PO0;
		localparam int unsigned  PMAX = (PI0 > PO0)? PI0 : PO0;
		typedef logic signed [$clog2(PMAX+1):0]  cap_t;
		logic [W0-1:0]  Buf[CAP];
		cap_t  ICap = PO0;	// PO0 (empty), .., 0, .., -PI0; >= 0: room
		cap_t  OAvl = -PO0;	// -PO0 (empty), .., 0, .., PI0; >= 0: available

		always_ff @(posedge clk) begin
			if(rst) begin
				Buf <= '{ default: 'x };
				ICap <= PO0;
				OAvl <= -PO0;
			end
			else begin
				// Buffer shift and deposit.
				if(1) begin
					automatic logic [W0-1:0]  n_buf[CAP] = Buf;
					if(otrn)  n_buf[0 +: CAP-PO0] = n_buf[PO0 +: CAP-PO0];
					if(1) begin
						automatic int unsigned  ofs = OAvl + (otrn? 0 : $signed(PO0));
						for(int unsigned  p = 0; p < PI0; p++)
							if(ofs + p < CAP)  n_buf[ofs + p] = idat[p*GCD +: GCD];
					end
					Buf <= n_buf;
				end

				// Capacity counters: base + constant delta.
				ICap <= (nxt? 0 : ICap) + cap_t'(nxt?  PO0 : otrn? (itrn? PO0-PI0 :  PO0) : itrn? -PI0 : 0);
				OAvl <= (nxt? 0 : OAvl) + cap_t'(nxt? -PO0 : otrn? (itrn? PI0-PO0 : -PO0) : itrn?  PI0 : 0);
			end
		end

		assign  irdy = !idone && !ICap[$left(ICap)];
		assign  ovld = !odone && !(OAvl[$left(OAvl)] && !idone);

		for(genvar  p = 0; p < PO; p++) begin : genOdat
			assign  odat[p] = (OLAST == PO || p < OLAST || !olast)? Buf[p/GCD][(p%GCD)*W +: W] : '0;
		end : genOdat

	end : genGeneric

`default_nettype wire
endmodule : vpc

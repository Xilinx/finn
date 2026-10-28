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
 *  on the last beat are padded with zeros (unless PAD_ZEROS=0).
 *
 *  Internally, parameters are normalized using gcd(PI,PO) to reduce
 *  buffer depth and counter widths. Let PI0=PI/GCD, PO0=PO/GCD.
 *
 *  RELAX_THROUGHPUT=0 (default): the bottleneck (higher-beat-count)
 *  interface sustains 1.0 cyc/beat in steady state. No combinatorial
 *  path from ordy to irdy. When both PI0>1 and PO0>1 and N is not
 *  an exact multiple of lcm(PI0,PO0), individual vectors are framed
 *  by a per-vector reset (genTrn). V+1's first input beat is accepted
 *  as soon as V's final output beat is pending, eliminating fill
 *  latency overhead.
 *
 *  RELAX_THROUGHPUT=1: compact implementation with a smaller buffer
 *  (PI0+PO0 vs PI0+PO0+min-1). Allows idle recovery cycles between
 *  load and emit phases, yielding simpler control at the cost of
 *  reduced throughput at non-integer width ratios.
 ***************************************************************************/

module vpc #(
	int unsigned  W,	// element bit width
	int unsigned  N,	// total elements per vector
	int unsigned  PI,	// input elements per beat
	int unsigned  PO,	// output elements per beat
	bit  PAD_ZEROS        = 1,	// zero-pad excess output lanes on last beat
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
	endfunction : gcd
	localparam int unsigned  GCD = gcd(PI, PO);
	localparam int unsigned  W0  = GCD * W; 	// normalized element width
	localparam int unsigned  PI0 = PI / GCD;	// normalized input parallelism
	localparam int unsigned  PO0 = PO / GCD;	// normalized output parallelism
	localparam int unsigned  NN  = 1 + (N-1)/GCD;	// normalized vector length

	// Reduce assumed vector length when plain, padding-free multiple of both PI0 and PO0
	localparam int unsigned  LCM0 = PI0 * PO0;
	localparam int unsigned  N0   = (NN % LCM0 == 0)? LCM0 : NN;

	//=======================================================================
	// Derived Parameters
	localparam int unsigned  TRNI = 1 + (N0-1)/PI0;	// input transactions per vector
	localparam int unsigned  TRNO = 1 + (N0-1)/PO0;	// output transactions per vector
	localparam int unsigned  OLAST = (N0 < NN)? PO : N - (TRNO-1)*PO;	// valid output lanes on last beat (1..PO)

	uwire  itrn = ivld && irdy;
	uwire  otrn = ovld && ordy;

	uwire [W0-1:0]  odat_raw[PO0];
	uwire           olast_beat;

	if((PI0 == 1) && (PO0 == 1)) begin : genWire
		//===============================================================
		// Wire-through: PI0 == PO0 == 1 → no conversion needed.
		assign	odat_raw[0] = idat;
		assign	olast_beat = 1;
		assign	ovld = ivld;
		assign	irdy = ordy;

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

			assign	irdy = !Cnt[$left(Cnt)];
			assign	ovld = Cnt[$left(Cnt)];
			assign	odat_raw[0] = Buf[0];
			assign	olast_beat = &Cnt;

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

			assign	irdy = !Cnt[$left(Cnt)];
			assign	ovld = SVld;
			assign	odat_raw[0] = Side;
			assign	olast_beat = !Cnt[$left(Cnt)];

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

			assign	irdy = Cnt[$left(Cnt)];
			assign	ovld = !Cnt[$left(Cnt)];

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

			assign	irdy = SRdy;
			assign	ovld = full;

		end : genFull

		// Output: N0 valid elements from shift register.
		for(genvar  p = 0; p < PO0; p++) begin : genOraw
			assign	odat_raw[p] = (p < N0)? Buf[p] : 'x;
		end : genOraw
		assign	olast_beat = 1;

	end : genDes
	else begin : genGeneric
		//===============================================================
		// Buffered generic implementation with deposit mask and
		// IRot barrel. RELAX_THROUGHPUT selects between a compact
		// single-counter variant allowing idle recovery cycles
		// and a full-rate dual-counter variant with enlarged buffer.

		//---------------------------------------------------------------
		// Transaction counters only needed for sync when padding differs.
		uwire  nxt;
		uwire  idone;
		uwire  odone;
		uwire  olast;
		uwire  ipre;	// V+1's first beat parked while V's final beat drains
		if(TRNI * PI0 == TRNO * PO0) begin : genSimple
			assign  nxt   = 0;
			assign  idone = 0;
			assign  odone = 0;

			if(OLAST == PO)  assign  olast = 0;
			else begin : genOlast
				logic signed [$clog2((TRNO > 1)? TRNO : 2):0]  OBeat = 1-TRNO;
				always_ff @(posedge clk) begin
					if(rst)  OBeat <= 1-TRNO;
					else     OBeat <= OBeat + ((olast && otrn)? -TRNO : 0) + otrn;
				end
				assign	olast = !OBeat[$left(OBeat)];
			end : genOlast
		end : genSimple
		else begin : genTrn
			localparam int unsigned  ITRN_W = $clog2((TRNI > 2)? TRNI-1 : 2);
			localparam int unsigned  OTRN_W = $clog2((TRNO > 2)? TRNO-1 : 2);
			logic signed [ITRN_W:0]  ITrn = 1-TRNI;
			logic signed [OTRN_W:0]  OTrn = 1-TRNO;

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
					ITrn <= ITrn + (nxt? (itrn? 1-TRNI : -TRNI) : int'(itrn && !idone)) + (nxt && ipre);
					OTrn <= OTrn + (nxt? -TRNO : 0) + otrn;
				end
			end
		end : genTrn

		//---------------------------------------------------------------
		// Rotation logic for input placement.
		// Successive output drains rotate the deposit window relative
		// to the input lanes by PO0 mod PI0 each. SIRot tracks this
		// rotation so that rot[j%PI0] aligns with the otrn deposit
		// target at Buf[j]. Without otrn, a static NO_OTR_OFS into
		// rot compensates for the missing buffer shift.
		localparam int unsigned  NO_OTR_OFS = (PI0 >= 2)? (PI0 - PO0 % PI0) % PI0 : 0;
		uwire [W0-1:0]  rot[PI0];
		if(PI0 == 1)  assign  rot[0] = idat;
		else begin : genRotBarrel
			localparam int unsigned  PO0_MOD = PO0 % PI0;
			localparam int unsigned  IROT_W = $clog2(PI0);

			// SIRot = IRot - NO_OTR_OFS: sign bit replaces the
			// comparator IRot >= NO_OTR_OFS in the modular wrap.
			localparam int  SIROT_INIT = PO0_MOD - NO_OTR_OFS;
			logic signed [IROT_W:0]  SIRot = SIROT_INIT;
			if(PO0_MOD) begin : genIRotDyn
				// At nxt, SIRot jumps to one of two known values
				// depending on whether otrn fires concurrently.
				localparam int  NXT_INC_A = SIROT_INIT - ((TRNO + 1) * PO0_MOD) % PI0 + NO_OTR_OFS;
				localparam int  NXT_INC_B = SIROT_INIT - ( TRNO      * PO0_MOD) % PI0 + NO_OTR_OFS;
				always_ff @(posedge clk) begin
					if(rst)              SIRot <= SIROT_INIT;
					else if(nxt || otrn) SIRot <= SIRot +
						(nxt? (otrn? NXT_INC_B : NXT_INC_A) :
						 !SIRot[$left(SIRot)]? PO0_MOD - PI0 : PO0_MOD);
				end
			end : genIRotDyn

			// V+1's first deposit uses V+1's initial rotation, not the
			// stale end-of-V value. Parked ahead of V's final drain, it
			// lands PO0 positions higher, i.e. one drain step earlier.
			localparam int  SIROT_PRE = -int'(NO_OTR_OFS);
			uwire signed [IROT_W:0]  sirot_eff =
				nxt?   signed'(SIROT_INIT) :
				idone? signed'(SIROT_PRE) : SIRot;

			for(genvar  i = 0; i < PI0; i++) begin : genRot
				localparam int  LUT_LO = -(1 << IROT_W);
				localparam int  LUT_HI =  (1 << IROT_W) - 1;
				uwire [W0-1:0]  lut[LUT_LO:LUT_HI];
				for(genvar  k = LUT_LO; k <= LUT_HI; k++) begin : genLut
					assign	lut[k] = (-int'(NO_OTR_OFS) <= k && k < int'(PI0 - NO_OTR_OFS))? idat[((i + k + NO_OTR_OFS) % PI0) * GCD +: GCD] : 'x;
				end : genLut
				assign	rot[i] = lut[sirot_eff];
			end : genRot

		end : genRotBarrel

		//===============================================================
		// Capacity counter, deposit mask, and buffer.
		// RELAX_THROUGHPUT selects the compact single-counter variant;
		// otherwise the enlarged buffer with dual counters provides
		// sustained full-rate operation at non-integer width ratios.
		if(RELAX_THROUGHPUT) begin : genRelax
			//-----------------------------------------------------------
			// Single capacity counter (signed, count up from negative).
			localparam int unsigned  CAP  = PI0 + PO0;
			localparam int unsigned  PMAX = (PI0 > PO0)? PI0 : PO0;
			typedef logic signed [$clog2(PMAX+1):0]  cap_t;
			cap_t  ICap = PO0;

			always_ff @(posedge clk) begin
				if(rst)  ICap <= PO0;
				else     ICap <= (nxt? 0 : ICap) + cap_t'(nxt? PO0 : otrn? (itrn? PO0-PI0 : PO0) : itrn? -PI0 : 0);
			end

			assign	ipre = 0;

			uwire  icap_positive = !ICap[$left(ICap)] && (|ICap[$left(ICap)-1:0]);
			assign	irdy = !idone && !ICap[$left(ICap)];
			assign	ovld = !odone && !(icap_positive && !idone);

			//-----------------------------------------------------------
			// Deposit mask: CAP-bit rotating register, single step.
			// PI0 ≡ CAP-PO0 (mod CAP), so itrn and otrn share one step.
			localparam bit [CAP-1:0]  DEPMASK_INIT = {(PI0){1'b1}};
			logic [CAP-1:0]  DepMask = DEPMASK_INIT;
			always_ff @(posedge clk) begin
				localparam int unsigned  MSTEP_SINGLE = PI0 % CAP;
				localparam int unsigned  MSTEP_DOUBLE = (2*PI0) % CAP;

				if(rst || nxt)  DepMask <= DEPMASK_INIT;
				else begin
					DepMask <=
						itrn && otrn? {DepMask[CAP-1-MSTEP_DOUBLE:0], DepMask[CAP-1:CAP-MSTEP_DOUBLE]} :
						itrn || otrn? {DepMask[CAP-1-MSTEP_SINGLE:0], DepMask[CAP-1:CAP-MSTEP_SINGLE]} :
						/* else */    DepMask;
				end
			end

			uwire [CAP-1:0]  dep_eff = otrn? (DepMask >> PO0) : DepMask;

			//-----------------------------------------------------------
			// Structural per-position buffer.
			logic [W0-1:0]  Buf[CAP];
			for(genvar  j = 0; j < CAP; j++) begin : genPos
				uwire           deposit = itrn && dep_eff[j];
				uwire           we  = deposit || (otrn && (j < PI0));
				uwire [W0-1:0]  dat = deposit? (otrn? rot[j % PI0] : rot[(j + NO_OTR_OFS) % PI0]) : Buf[(j + PO0) % CAP];
				always_ff @(posedge clk) begin
					if(rst)      Buf[j] <= 'x;
					else if(we)  Buf[j] <= dat;
				end
			end : genPos

			assign	odat_raw = Buf[0:PO0-1];

		end : genRelax
		else begin : genFull
			//-----------------------------------------------------------
			// Dual capacity counters (sign bit = control output).
			//   ICap = fill - ICAP_INIT:  negative ↔ room for input
			//   OCap = (PO0-1) - fill:    negative ↔ output available
			// On nxt, the counter resets to the cold-start value.
			// V+1's first beat, whether concurrent with V's last drain
			// or parked before it (Pre), is folded into the reset via
			// icnt. While parked, it does not affect V's counters.
			localparam int unsigned  P0MIN = (PI0 < PO0)? PI0 : PO0;
			localparam int unsigned  CAP   = PI0 + PO0 + P0MIN - 1;

			localparam int unsigned  ICAP_INIT = CAP - PI0 + 1;
			localparam int unsigned  OCAP_INIT = CAP - PO0 + 1;
			localparam int unsigned  CAP_W = $clog2((ICAP_INIT > OCAP_INIT)? ICAP_INIT : OCAP_INIT);
			typedef logic signed [CAP_W:0]  icap_t;
			typedef logic signed [CAP_W:0]  ocap_t;

			icap_t  ICap = -ICAP_INIT;
			ocap_t  OCap = PO0 - 1;

			// Pre: V+1's first beat was accepted while V's final
			// output beat was still pending. It is parked at
			// [PO0+PI0-1:PO0] and shifted into place by V's final drain.
			logic  Pre = 0;
			always_ff @(posedge clk) begin
				if(rst)  Pre <= 0;
				else     Pre <= !nxt && (Pre || (itrn && idone));
			end
			assign	ipre = Pre;

			uwire  oadj = otrn && !nxt;
			uwire  icnt = nxt? (itrn || Pre) : (itrn && !idone);
			always_ff @(posedge clk) begin
				if(rst) begin
					ICap <= -ICAP_INIT;
					OCap <= PO0 - 1;
				end
				else begin
					ICap <= (nxt? icap_t'(-ICAP_INIT) : ICap) + icap_t'(oadj? (icnt? PI0-PO0-1 : -PO0) : (icnt? PI0-1 : 0)) + icnt;
					OCap <= (nxt? ocap_t'( PO0 - 1)   : OCap) + ocap_t'(icnt? (oadj? PO0-PI0-1 : -PI0) : (oadj? PO0-1 : 0)) + oadj;
				end
			end

			// Registered state only: no combinatorial path from ivld
			// or ordy to irdy. Once V's input is complete, V+1's first
			// beat is accepted as soon as V's final output beat is
			// pending; its residue fits in [PO0-1:0].
			assign	irdy = idone? olast && !Pre : ICap[$left(ICap)];
			assign	ovld = !odone && (OCap[$left(OCap)] || idone);

			//-----------------------------------------------------------
			// Deposit mask: CAP-bit rotating register, separate steps.
			// The mask wraps around CAP only at fill levels where
			// irdy is deasserted, so dep_eff (shift, not rotation)
			// is correct whenever itrn fires.
			// On nxt, warm-start: advance by one itrn step if V+1's
			// first beat fires alongside V's final drain or is parked.
			localparam bit [CAP-1:0]  DEPMASK_INIT = {(PI0){1'b1}};
			localparam bit [CAP-1:0]  DEPMASK_WARM = {DEPMASK_INIT[CAP-1-PI0:0], DEPMASK_INIT[CAP-1:CAP-PI0]};
			localparam bit [CAP-1:0]  DEPMASK_PRE  = DEPMASK_INIT << PO0;
			logic [CAP-1:0]  DepMask = DEPMASK_INIT;

			uwire [CAP-1:0]  dep_rotl = {DepMask[CAP-1-PI0:0], DepMask[CAP-1:CAP-PI0]};
			uwire [CAP-1:0]  dep_rotr = {DepMask[PO0-1:0], DepMask[CAP-1:PO0]};
			localparam int unsigned  MSTEP_B = (PI0 > PO0)? PI0 - PO0 : PO0 - PI0;
			uwire [CAP-1:0]  dep_both = (PI0 > PO0)?
				{DepMask[CAP-1-MSTEP_B:0], DepMask[CAP-1:CAP-MSTEP_B]} :
				{DepMask[MSTEP_B-1:0], DepMask[CAP-1:MSTEP_B]};

			always_ff @(posedge clk) begin
				if(rst)  DepMask <= DEPMASK_INIT;
				else begin
					DepMask <=
						nxt?          (itrn || Pre? DEPMASK_WARM : DEPMASK_INIT) :
						itrn && otrn? dep_both :
						itrn?         dep_rotl :
						otrn?         dep_rotr :
						/* else */    DepMask;
				end
			end

			// V+1's first deposit targets its initial positions
			// [PI0-1:0] in the post-shift buffer frame on nxt, or
			// [PO0+PI0-1:PO0] in the pre-shift frame when parked.
			uwire [CAP-1:0]  dep_eff =
				nxt?   DEPMASK_INIT :
				idone? DEPMASK_PRE  :
				otrn?  (DepMask >> PO0) : DepMask;

			//-----------------------------------------------------------
			// Structural per-position buffer.
			localparam int unsigned  SHIFT_RANGE = CAP - PO0;
			logic [W0-1:0]  Buf[CAP];
			for(genvar  j = 0; j < CAP; j++) begin : genPos
				uwire           deposit = itrn && dep_eff[j];
				uwire           we  = deposit || (otrn && (j < SHIFT_RANGE));
				uwire [W0-1:0]  dat = deposit? ((otrn && !nxt)? rot[j % PI0] : rot[(j + NO_OTR_OFS) % PI0]) : Buf[(j + PO0) % CAP];
				always_ff @(posedge clk) begin
					if(rst)      Buf[j] <= 'x;
					else if(we)  Buf[j] <= dat;
				end
			end : genPos

			assign	odat_raw = Buf[0:PO0-1];

		end : genFull

		//---------------------------------------------------------------
		// Output Assignment
		assign	olast_beat = olast;

	end : genGeneric

	for(genvar  p = 0; p < PO; p++) begin : genOdat
		assign	odat[p] = (!PAD_ZEROS || p < OLAST || !olast_beat)? odat_raw[p/GCD][(p%GCD)*W +: W] : '0;
	end : genOdat

`default_nettype wire
endmodule : vpc

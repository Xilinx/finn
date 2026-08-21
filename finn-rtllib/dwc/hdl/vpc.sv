/******************************************************************************
 * Copyright  Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @author	Thomas B. Preußer <tpreusse@amd.com>
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
 *****************************************************************************/

module vpc #(
	int unsigned  W,	// element bit width
	int unsigned  N,	// total elements per vector
	int unsigned  PI,	// input elements per beat
	int unsigned  PO	// output elements per beat
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

	if(PI == PO) begin : genTrivial

		// No conversion needed: wire input directly to output.
		assign	odat = idat;
		assign	ovld = ivld;
		assign	irdy = ordy;

	end : genTrivial
	else begin : genConvert

		// Extract GCD from in- and output parallelism and map to data path width
		function int unsigned gcd(input int unsigned a, input int unsigned b);
			return (b == 0)? a : gcd(b, a % b);
		endfunction
		localparam int unsigned  GCD = gcd(PI, PO);
		localparam int unsigned  W0  = GCD * W;			// normalized element width
		localparam int unsigned  PI0 = PI / GCD;		// normalized input parallelism
		localparam int unsigned  PO0 = PO / GCD;		// normalized output parallelism
		localparam int unsigned  N0  = 1 + (N-1)/GCD;	// normalized vector length

		//=======================================================================
		// Derived Parameters
		localparam int unsigned  TRNI  = 1 + (N0-1)/PI0;	// input transactions per vector
		localparam int unsigned  TRNO  = 1 + (N0-1)/PO0;	// output transactions per vector
		localparam int unsigned  CAP   = PI0 + PO0;     	// internal buffer capacity

		//=======================================================================
		// Vector Progress Counters
		logic signed [$clog2(TRNI):0]  ITrn = TRNI-1;	// TRNI-1, .., 0, -1 (done)
		logic signed [$clog2(TRNO):0]  OTrn = TRNO-1;	// TRNO-1, .., 0, -1 (done)
		uwire  idone = ITrn[$left(ITrn)];
		uwire  odone = OTrn[$left(OTrn)];
		uwire  itrn = ivld && irdy;
		uwire  otrn = ovld && ordy;

		//=======================================================================
		// Internal Buffer
		//  ICap (input capacity):   ICap += PO0 on pop, -= PI0 on push.  Sign bit 0 → room for PI0.
		//  ORdy (output readiness): ORdy -= PO0 on pop, += PI0 on push.  Sign bit 0 → full beat.
		logic [W0-1:0]  Buf[CAP];
		logic signed [$clog2(CAP):0]  ICap = PO0;	// PO0 (empty), .., 0, .., -PI0; >= 0: room
		logic signed [$clog2(CAP):0]  ORdy = -PO0;	// -PO0 (empty), .., 0, .., PI0; >= 0: ready

		//=======================================================================
		// Sequential State Update (two-phase: otrn first, then itrn)
		always_ff @(posedge clk) begin
			if(rst) begin
				Buf <= '{ default: 'x };
				ICap <= PO0;
				ORdy <= -PO0;
				ITrn <= TRNI-1;
				OTrn <= TRNO-1;
			end
			else begin
				automatic logic [W0-1:0]  n_buf[CAP];
				automatic logic signed [$clog2(CAP):0]   n_icap;
				automatic logic signed [$clog2(CAP):0]   n_ordy;
				automatic logic signed [$clog2(TRNI):0]  n_itrn;
				automatic logic signed [$clog2(TRNO):0]  n_otrn;

				n_buf  = Buf;
				n_icap = ICap;
				n_ordy = ORdy;
				n_itrn = ITrn;
				n_otrn = OTrn;

				// Phase 1: retire PO0 elements from buffer head.
				if(otrn) begin
					n_buf[0 +: CAP-PO0] = n_buf[PO0 +: CAP-PO0];
					n_icap += PO0;
					n_ordy -= PO0;
					n_otrn--;
				end

				// Phase 2: append PI0 input elements at buffer tail.
				if(itrn) begin
					automatic int unsigned  ofs = ORdy + (otrn? 0 : $signed(PO0));
					for(int unsigned  p = 0; p < PI0; p++)
						n_buf[ofs + p] = idat[p*GCD +: GCD];
					n_icap -= PI0;
					n_ordy += PI0;
					n_itrn--;
				end

				// When both input and output vectors are complete, start next vector.
				if(n_itrn[$left(n_itrn)] && n_otrn[$left(n_otrn)]) begin
					n_itrn = TRNI-1;
					n_otrn = TRNO-1;
					n_icap =  PO0;
					n_ordy = -PO0;
				end

				Buf <= n_buf;
				ITrn <= n_itrn;
				OTrn <= n_otrn;
				ICap <= n_icap;
				ORdy <= n_ordy;
			end
		end

		//=======================================================================
		// Ultimate Module Outputs

		// Accept input: vector not complete and buffer has capacity.
		assign  irdy = !idone && !ICap[$left(ICap)];

		// Emit output: full beat available, or input done with residual data.
		assign  ovld = !odone && (!ORdy[$left(ORdy)] || idone);

		// Output: low PO0 buffer elements driven unconditionally.
		for(genvar  p = 0; p < PO0; p++)  assign  odat[p*GCD +: GCD] = Buf[p];

	end : genConvert

`default_nettype wire
endmodule : vpc

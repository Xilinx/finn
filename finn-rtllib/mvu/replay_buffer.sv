/******************************************************************************
 * Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
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
 * @brief	Replay buffer for counted sequences on an AXI-lite stream.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module replay_buffer #(
	int unsigned  LEN,	// Sequence length
	int unsigned  REP,	// Sequence replay count
	int unsigned  W 	// Data width
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [W-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [W-1:0]  odat,
	output	logic  olast,
	output	logic  ofin,
	output	logic  ovld,
	input	logic  ordy
);

	if(LEN == 0)  initial begin
		$error("%m: Illegal zero sequence LEN.");
		$finish;
	end
	if(REP == 0) initial begin
		$error("%m: Illegal zero REP count.");
		$finish;
	end

	// Track position in Sequence
	uwire  last_item;
	uwire  shift;
	if(LEN == 1)  assign  last_item = 1;
	else begin
		typedef logic [$clog2(LEN)-1:0]  count_t;
		count_t  Count = 0;
		logic    Last  = 0;
		always_ff @(posedge clk) begin
			if(rst) begin
				Count <= 0;
				Last  <= 0;
			end
			else if(shift) begin
				Count <= Count + (Last? 2**$clog2(LEN)-LEN+1 : 1);
				Last  <= (((LEN-2) & ~Count) == 0) && ((LEN&1) || !Last);
			end
		end
		assign	last_item = Last;
	end

	if(REP == 1) begin
		assign	shift = ivld && ordy;

		assign	irdy  = ordy;
		assign	odat  = idat;
		assign	olast = last_item;
		assign	ofin  = last_item;
		assign	ovld  = ivld;
	end
	else begin

		// Track Repetitions
		uwire  last_rep;
		if(1) begin : blkRep
			typedef logic [$clog2(REP)-1:0]  rep_t;
			rep_t  RepCnt = 0;
			logic  RepLst = 0;
			always_ff @(posedge clk) begin
				if(rst) begin
					RepCnt <= 0;
					RepLst <= 0;
				end
				else if(last_item && shift) begin
					RepCnt <= RepCnt + (RepLst? 2**$clog2(REP)-REP+1 : 1);
					RepLst <= (((REP-2) & ~RepCnt) == 0) && ((REP&1) || !RepLst);
				end
			end
			assign	last_rep = RepLst;
		end : blkRep

		localparam int unsigned  AWIDTH = LEN < 2? 1 : $clog2(LEN);
		typedef logic [AWIDTH  :0]  ptr_t;	// pointers with additional generational MSB
		typedef logic [W     -1:0]  data_t;

		// Output Registers
		data_t  ODat;
		logic   OVld =  0;
		logic   OLst = 'x;
		logic   OFin = 'x;
		assign	odat  = ODat;
		assign	olast = OLst;
		assign	ofin  = OFin;
		assign	ovld  = OVld;

		// Buffer Memory Management
		data_t  Mem[2**AWIDTH];
		ptr_t  WP = 0;	// Write Pointer
		ptr_t  RP = 0;	// Read Pointer
		ptr_t  FP = 0;	// Free Pointer

		// Operational Guards
		//	Occupancy:    WP-FP
		//	  WP-FP < 2**AWIDTH -> writing allowed
		//		- increments WP
		//	Availability: WP-RP
		//	  WP-RP > 0         -> reading allowed
		//		- increments RP, last in sequence rewinds to FP for non-final repetition
		//		- increments FP in last repetition
		assign	irdy = !((WP-FP) >> AWIDTH);

		uwire  wr = irdy && ivld;
		uwire  rd = !OVld || ordy;
		always_ff @(posedge clk) begin
			if(wr)  Mem[WP[AWIDTH-1:0]] <= idat;
			if(rd)  ODat <= Mem[RP[AWIDTH-1:0]];
		end

		uwire  vld = (RP != WP);
		assign	shift = rd && vld;
		always_ff @(posedge clk) begin
			if(rst) begin
				WP <= 0;
				RP <= 0;
				FP <= 0;

				OVld <=  0;
				OLst <= 'x;
				OFin <= 'x;
			end
			else begin
				if(wr)  WP <= WP + 1;
				if(rd) begin
					if(vld) begin
						automatic logic  rewind = last_item && !last_rep;
						RP <= RP + (rewind? 2**(AWIDTH+1)-LEN+1 : 1);
						FP <= FP + last_rep;
					end

					OVld <= vld;
					OLst <= last_item;
					OFin <= last_rep && last_item;
				end
			end
		end

	end

endmodule : replay_buffer

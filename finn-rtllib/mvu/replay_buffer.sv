/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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

	typedef logic [$clog2(REP)+$clog2(LEN)-1:0]  count_t;
	count_t  Count = 0;
	uwire  done_len = LEN == 1 ? 1 : ((LEN-1) & ~Count[$clog2(LEN)-1:0]) == 0;
	uwire  done_rep;
	uwire  done_all = done_len && done_rep;

	uwire  shift;
	uwire  clr = rst || (done_all && shift);
	always_ff @(posedge clk) begin
		if(clr)         Count <= 0;
		else if(shift)  Count <= Count + ((REP > 1) && done_len? 2**$clog2(LEN)-LEN+1 : 1);
	end

	typedef logic [W-1:0]  data_t;
	uwire data_t  rdat;
	uwire  first_rep;
	if(REP == 1) begin
		assign	done_rep  = 1;
		assign	first_rep = 1;
		assign	rdat = 'x;
	end
	else begin
		assign	done_rep = ((REP-1) & ~Count[$left(Count):$clog2(LEN)]) == 0;

		logic  FirstRep = 1;
		always_ff @(posedge clk) begin
			if(clr)         FirstRep <= 1;
			else if(shift)  FirstRep <= FirstRep && !done_len;
		end
		assign	first_rep = FirstRep;

		data_t  Buf[LEN];
		if(LEN == 1) begin : genTrivial
			always_ff @(posedge clk) begin
				if(shift && FirstRep)  Buf[0] <= idat;
			end
		end : genTrivial
		else begin : genShift
			always_ff @(posedge clk) begin
				if(shift) begin
					Buf[0] <= odat;
					Buf[1:LEN-1] <= Buf[0:LEN-2];
				end
			end
		end : genShift

		assign	rdat = Buf[LEN-1];
	end

	assign  irdy  = ordy && first_rep;
	assign	odat  = first_rep? idat : rdat;
	assign	olast = done_len;
	assign	ofin  = done_all;
	assign	ovld  = first_rep? ivld : 1;
	assign	shift = ovld && ordy;

endmodule : replay_buffer
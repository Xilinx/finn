/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
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
 * @brief	Queue-based unbounded FIFO drop-in for size-gauging simulation.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module fifo_gauge #(
	int unsigned  WIDTH,
	int unsigned  COUNT_WIDTH = 32,
	parameter bit  DEBUG_LOG = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy,

	output	logic [COUNT_WIDTH-1:0]  count,
	output	logic [COUNT_WIDTH-1:0]  maxcount
);

	localparam int  STDERR_FD = 32'h8000_0002;

	//-----------------------------------------------------------------------
	// Monitoring & Debug

	// Transaction counters
	longint unsigned  ITxnCnt = 0;
	longint unsigned  OTxnCnt = 0;

	// The internal Queue serving as data buffer and an output register
	logic [WIDTH-1:0]  Q[$] = {};
	longint unsigned  Count    = 0;
	longint unsigned  MaxCount = 0;

	logic  OVld = 0;
	logic [WIDTH-1:0]  ODat = 'x;

	final begin
		$fwrite(STDERR_FD, "[%m @%0t] MaxFill: %0d; Transactions: in=%0d out=%0d", $time, MaxCount, ITxnCnt, OTxnCnt);
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Q         = {};
			Count    <= 0;
			MaxCount <= 0;
			OVld <= 0;
			ODat <= 'x;

			ITxnCnt <= 0;
			OTxnCnt <= 0;
		end
		else begin
			// Always take input and track Transactions
			if(ivld) begin
				Q.push_back(idat);
				if(DEBUG_LOG)  $fwrite(STDERR_FD, "[FIFOLOG %m @%0t] %0x\n", $time, idat);
				ITxnCnt <= ITxnCnt + 1;
			end
			if(OVld && ordy)  OTxnCnt <= OTxnCnt + 1;

			// Take Count
			Count <= Q.size;
			if(Q.size > MaxCount)  MaxCount <= Q.size;

			// Offer output when available
			if(!OVld || ordy) begin
				if(Q.size == 0) begin
					OVld <= 0;
					ODat <= 'x;
				end
				else begin
					OVld <= 1;
					ODat <= Q.pop_front();
				end
			end
		end
	end
	assign	irdy = 1;
	assign	ovld = OVld;
	assign	odat = ODat;

	assign	count = Count;
	assign	maxcount = MaxCount;

endmodule : fifo_gauge

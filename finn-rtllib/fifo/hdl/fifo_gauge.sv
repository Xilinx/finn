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
	int unsigned WIDTH,

	// Logging controls
	parameter    DATA_LOGFILE = "",
	int unsigned LOG_VERBOSE = 0, // 0: (data_in), 1: (data, direction, cycle)
	int unsigned LOG_FLUSH = 65536
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy,

	output	int unsigned  count,     // mod 2^32
	output	int unsigned  maxcount   // likely count overflow when at 2^32-1
);

	//-----------------------------------------------------------------------
	// Monitoring & Debug

	// Transaction counters
	longint unsigned  ITxnCnt = 0;
	longint unsigned  OTxnCnt = 0;
	int  LogFd = (DATA_LOGFILE != "")? $fopen(DATA_LOGFILE, "w") : 0;

	// Logging: Clock cycle counter
	longint unsigned  Cycle = 0;

	// The internal Queue serving as data buffer and an output register
	logic [WIDTH-1:0]  Q[$] = {};
	int unsigned  Count    = 0;
	int unsigned  MaxCount = 0;

	logic  OVld = 0;
	logic [WIDTH-1:0]  ODat = 'x;

	// Logging: LOG_FLUSH Counter
	int unsigned  Unflushed = 0;
	task automatic note_line();
		Unflushed++;
		if(LOG_FLUSH && (Unflushed >= LOG_FLUSH)) begin
			$fflush(LogFd);
			Unflushed = 0;
		end
	endtask : note_line

	// Logging: Print statement
	initial begin
		if(LogFd) begin
			// Verbose format: data, dir (0=in, 1=out), cycle
			if(LOG_VERBOSE)  $fwrite(LogFd, "# data dir cycle\n");
			else             $fwrite(LogFd, "# data\n");
		end
	end

	// Logging: Final print statement
	final begin
		if(LogFd) begin
			$fwrite(
				LogFd,
				"# [%m @%0t] Cycles: %0d; MaxFill: %0d; Transactions: in=%0d out=%0d\n",
				$time, Cycle, MaxCount, ITxnCnt, OTxnCnt
			);
			$fclose(LogFd);
		end
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Q         = {};
			Count    <= 0;
			MaxCount <= 0;
			OVld <= 0;
			ODat <= 'x;

			Cycle   <= 0;
			ITxnCnt <= 0;
			OTxnCnt <= 0;
		end
		else begin
			automatic int unsigned  count = Count;
			Cycle <= Cycle + 1;

			// Always take input and track Transactions
			if(ivld) begin
				Q.push_back(idat);
				if(LogFd) begin
					if(LOG_VERBOSE)  $fwrite(LogFd, "%0x 0 %0d\n", idat, Cycle);
					else             $fwrite(LogFd, "%0x\n", idat);
					note_line();
				end
				ITxnCnt <= ITxnCnt + 1;
				count++;
			end
			if(OVld && ordy) begin
				if(LogFd && LOG_VERBOSE) begin
					$fwrite(LogFd, "%0x 1 %0d\n", ODat, Cycle);
					note_line();
				end
				OTxnCnt <= OTxnCnt + 1;
				count--;
			end

			// Track Count
			assert((count != 0) || (Count != '1)) else begin
				$error("%m: FIFO fill counter overflowed!");
				$stop;
			end
			Count <= count;
			if(MaxCount < count)  MaxCount <= count;

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

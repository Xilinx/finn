/******************************************************************************
 *  Copyright (c) 2022, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Testbench for checksum component.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 *******************************************************************************/
module checksum_tb;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst;

	//-----------------------------------------------------------------------
	// DUT
	localparam int unsigned  N = 60;	// words per frame
	localparam int unsigned  K = 4;		// subwords per word
	localparam int unsigned  W = 8;		// subword size

	logic [K-1:0][W-1:0]  src_TDATA;
	logic  src_TVALID;
	uwire  src_TREADY;

	uwire [K-1:0][W-1:0]  dst_TDATA;
	uwire  dst_TVALID;
	logic  dst_TREADY;

	uwire [31:0]  chk;
	uwire         chk_vld;

	checksum_top dut (
		.ap_clk(clk), .ap_rst_n(!rst),
		.src_TDATA, .src_TVALID, .src_TREADY,
		.dst_TDATA, .dst_TVALID, .dst_TREADY,
		.chk, .chk_ap_vld(chk_vld),
		.ap_local_block(), .ap_local_deadlock()
	);

	//-----------------------------------------------------------------------
	// Stimulus
	logic [K-1:0][W-1:0]  Bypass  [$] = {};
	logic [31:0]          Checksum[$] = {};
	initial begin
		src_TDATA  = 'x;
		src_TVALID =  0;

		rst = 1;
		repeat(9) @(posedge clk);
		rst <= 0;

		for(int unsigned  r = 0; r < 311; r++) begin
			automatic logic [23:0]  sum = 0;
			src_TVALID <= 1;
			for(int unsigned  i = 0; i < N; i++) begin
				for(int unsigned  k = 0; k < K; k++) begin
					automatic logic [W-1:0]  v = $urandom()>>17;
					src_TDATA[k] <= v;
					sum += ((K*i+k)%3 + 1) * v;
				end
				@(posedge clk iff src_TREADY);
				Bypass.push_back(src_TDATA);
			end
			src_TVALID <= 0;
			$display("Expect: %02x:%06x", r[7:0], sum);
			Checksum.push_back({r, sum});
		end

		repeat(8) @(posedge clk);
		$finish;
	end

	//-----------------------------------------------------------------------
	// Output Validation

	// Drain and check pass-thru stream
	assign	dst_TREADY = 1;
	always_ff @(posedge clk iff dst_TVALID) begin
		assert(Bypass.size()) begin
			automatic logic [K-1:0][W-1:0]  exp = Bypass.pop_front();
			assert(dst_TDATA === exp) else begin
				$error("Unexpected output %0x instead of %0x.", dst_TDATA, exp);
				$stop;
			end
		end
		else begin
			$error("Spurious data output.");
			$stop;
		end
	end

	// Validate checksum reports
	always_ff @(posedge clk iff chk_vld) begin
		$display("Check:  %02x:%06x", chk[31:24], chk[23:0]);
		assert(Checksum.size()) begin
			automatic logic [31:0]  exp = Checksum.pop_front();
			assert(chk === exp) else begin
				$error("Unexpected checksum %0x instead of %0x.", chk, exp);
				$stop;
			end
		end
		else begin
			$error("Spurious checksum output.");
			$stop;
		end
	end

endmodule : checksum_tb

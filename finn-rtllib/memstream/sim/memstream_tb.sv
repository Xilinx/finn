/**
 * Copyright (c) 2023, Xilinx
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of FINN nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 */

module memstream_tb;
	localparam int unsigned  SETS = 3;
	localparam int unsigned  DEPTH = 256;
	localparam int unsigned  DATA_WIDTH = 32;

	localparam int unsigned  SET_BITS = SETS < 2? 1 : $clog2(SETS);

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst;

	// Configuration Interface
	logic [31:0]  config_address;
	logic  config_ce;
	logic  config_we;
	logic [DATA_WIDTH-1:0]  config_d0;
	uwire  config_rack;
	uwire [DATA_WIDTH-1:0]  config_q0;

	logic [SET_BITS-1:0]  sidx;
	logic  svld;
	uwire  srdy;

	// Streamed Output
	logic  ordy;
	uwire  ovld;
	uwire [DATA_WIDTH-1:0]  odat;

	initial begin
		config_address = 'x;
		config_ce = 0;
		config_we = 0;
		config_d0 = 'x;

		sidx = 'x;
		svld = 0;

		ordy = 0;

		rst = 1;
		repeat(16)  @(posedge clk);
		rst <= 0;

		// Write Parameters
		config_we <= 1;
		for(int unsigned  i = 0; i < SETS*DEPTH; i++) begin
			config_address <= i;
			config_d0 <= i;
			while($urandom()%59 == 0) @(posedge clk);

			config_ce <= 1;
			@(posedge clk);
			config_ce <= 0;
		end
		config_address <= 'x;
		config_d0 <= 'x;
		config_we <= 0;

		rst <= 1;
		@(posedge clk);
		rst <= 0;

		// One Round of Reading all Streams in Sequence
		ordy <= 1;
		if(SETS > 1) fork begin
			svld <= 1;
			for(int unsigned  s = 0; s < SETS; s++) begin
				sidx <= s;
				@(posedge clk iff srdy);
			end
			sidx <= 'x;
			svld <= 0;
		end join_none
		for(int unsigned  i = 0; i < SETS*DEPTH; i++) begin
			@(posedge clk iff ovld);
			assert(odat == i) else begin
				$error("Unexpected output: %0d instead of %0d", odat, i);
				$stop;
			end
		end
		ordy <= 0;

		// Full Parameter Readback
		fork
			begin	// Issue Read Requests
				config_ce <= 1;
				for(int unsigned  i = 0; i < SETS*DEPTH; i++) begin
					config_address <= i;
					@(posedge clk);
				end
				config_address <= 'x;
				config_ce <= 0;
			end
			begin	// Collect Replies
				for(int unsigned  i = 0; i < SETS*DEPTH; i++) begin
					@(posedge clk iff config_rack);
					assert(config_q0 == i) else begin
						$error("Readback mismatch: %0d instead of %0d", config_q0, i);
						$stop;
					end
				end
			end
		join

		repeat(6) @(posedge clk);

		// Reading Streams in reverse Order
		ordy <= 1;
		if(SETS > 1) fork begin
			svld <= 1;
			for(int unsigned  s = SETS; s-- > 0;) begin
				sidx <= s;
				@(posedge clk iff srdy);
			end
			sidx <= 'x;
			svld <= 0;
		end join_none
		for(int unsigned  s = SETS; s-- > 0;) begin
			for(int unsigned  i = 0; i < DEPTH; i++) begin
				automatic int unsigned  exp = s*DEPTH + i;
				@(posedge clk iff ovld);
				assert(odat == exp) else begin
					$error("Unexpected output: %0d instead of %0d", odat, exp);
					$stop;
				end
			end
		end
		ordy <= 0;

		// A few randomized Stream Reads with intermittent Read Backs
		if(1) begin
			automatic logic [DATA_WIDTH-1:0]  Q[$] = {};	// Read back reference
			automatic bit  term = 0;

			fork
				// Stream Output Checking
				begin
					repeat(7) begin
						automatic int unsigned  s = $urandom()%SETS;

						if(SETS > 1) fork begin
							sidx <= s;
							svld <= 1;
							@(posedge clk iff srdy);
							svld <= 0;
							sidx <= 'x;
						end join_none

						for(int unsigned  i = 0; i < DEPTH; i++) begin
							automatic int unsigned  exp = s*DEPTH + i;

							while($urandom()%7 == 0) @(posedge clk);
							ordy <= 1;
							@(posedge clk iff ovld);
							assert(odat == exp) else begin
								$error("Unexpected output: %0d instead of %0d", odat, exp);
								$stop;
							end
							ordy <= 0;
						end
					end
					term = 1;
				end

				// Intermittent Readbacks
				forever begin
					while($urandom() % 11) @(posedge clk);
					if(term)  break;

					config_ce <= 1;
					config_address <= $urandom() % (SETS*DEPTH);
					@(posedge clk);
					Q.push_back(config_address);
					config_ce <= 0;
					config_address <= 'x;
				end

				// Readback Checker
				forever begin
					if(Q.size()) begin
						automatic logic [DATA_WIDTH-1:0]  exp = Q.pop_front();
						@(posedge clk iff config_rack);
						assert(config_q0 == exp) else begin
							$error("Readback mismatch: %0d instead of %0d", config_q0, exp);
							$stop;
						end
					end
					else begin
						@(posedge clk);
						assert(!config_rack) else begin
							$error("Spurious readback reply.");
							$stop;
						end
						if(term)  break;
					end
				end
			join
		end

		$display("Test completed.");
		$finish;
	end

	memstream #(
		.SETS(SETS),
		.DEPTH(DEPTH),
		.WIDTH(DATA_WIDTH)
	) dut (
		.clk, .rst,

		.config_address,
		.config_ce,
		.config_we,
		.config_d0,
		.config_q0,
		.config_rack,

		.sidx, .svld, .srdy,
		.odat, .ovld, .ordy
	);

endmodule : memstream_tb

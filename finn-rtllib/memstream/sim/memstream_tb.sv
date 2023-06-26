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
	localparam int unsigned  DEPTH = 256;
	localparam int unsigned  DATA_WIDTH = 32;

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

	// Streamed Output
	logic  ordy;
	uwire  ovld;
	uwire [DATA_WIDTH-1:0]  odat;

	initial begin
		config_address = 'x;
		config_ce = 0;
		config_we = 0;
		config_d0 = 'x;

		ordy = 0;

		rst = 1;
		repeat(16)  @(posedge clk);
		rst <= 0;

		// Write Parameters
		config_ce <= 1;
		config_we <= 1;
		for(int unsigned  i = 0; i < DEPTH; i++) begin
			config_address <= i;
			config_d0 <= i;
			@(posedge clk);
		end
		config_address <= 'x;
		config_ce <= 0;
		config_we <= 0;
		config_d0 <= 'x;

		rst <= 1;
		@(posedge clk);
		rst <= 0;

		// One Round of Stream Read
		ordy <= 1;
		for(int unsigned  i = 0; i < DEPTH; i++) begin
			@(posedge clk iff ovld);
			assert(odat == i) else begin
				$error("Unexpected output: %0d instead of %0d", odat, i);
				$stop;
			end
		end
		ordy <= 0;

		// Full Parameter Readback
		if(1) begin
			automatic logic [DATA_WIDTH-1:0]  Q[$] = {};

			config_ce <= 1;
			for(int unsigned  i = 0; i < DEPTH; i++) begin
				config_address <= i;
				@(posedge clk);
				Q.push_back(i);

				if(config_rack) begin
					automatic logic [DATA_WIDTH-1:0]  exp = Q.pop_front();
					assert(config_q0 == exp) else begin
						$error("Readback mismatch: %0d instead of %0d", config_q0, exp);
						$stop;
					end
				end
			end
			config_address <= 'x;
			config_ce <= 0;

			while(Q.size) begin
				automatic logic [DATA_WIDTH-1:0]  exp = Q.pop_front();

				@(posedge clk iff config_rack);
				assert(config_q0 == exp) else begin
					$error("Readback mismatch: %0d instead of %0d", config_q0, exp);
					$stop;
				end
			end
		end

		repeat(6) @(posedge clk);

		// Another Round of Stream Read
		ordy <= 1;
		for(int unsigned  i = 0; i < DEPTH; i++) begin
			@(posedge clk iff ovld);
			assert(odat == i) else begin
				$error("Unexpected output: %0d instead of %0d", odat, i);
				$stop;
			end
		end
		ordy <= 0;

		// A Round of Stream Read with intermittent Read Backs
		if(1) begin
			automatic logic [DATA_WIDTH-1:0]  Q[$] = {};

			for(int unsigned  i = 0; i < DEPTH; i++) begin
				do begin
					// Randomly delayed Readiness
					if($urandom()%5 != 0)  ordy <= 1;

					// Issue and Check Random Read Backs
					if($urandom()%9 == 0) begin
						automatic int unsigned  addr = $urandom() % DEPTH;
						config_ce <= 1;
						config_address <= addr;
						Q.push_back(addr);
					end
					@(posedge clk);
					config_ce <= 0;
					config_address <= 'x;

					if(config_rack) begin
						automatic logic [DATA_WIDTH-1:0]  exp = Q.pop_front();
						assert(config_q0 == exp) else begin
							$error("Readback mismatch: %0d instead of %0d", config_q0, exp);
							$stop;
						end
					end

				end while(!ovld || !ordy);
				ordy <= 0;

				assert(odat == i) else begin
					$error("Unexpected output: %0d instead of %0d", odat, i);
					$stop;
				end
			end

			while(Q.size) begin
				automatic logic [DATA_WIDTH-1:0]  exp = Q.pop_front();

				@(posedge clk iff config_rack);
				assert(config_q0 == exp) else begin
					$error("Readback mismatch: %0d instead of %0d", config_q0, exp);
					$stop;
				end
			end
		end
		ordy <= 0;

		repeat(2) @(posedge clk);
		$display("Test completed.");
		$finish;
	end

	memstream #(
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

		.ordy,
		.ovld,
		.odat
	);

endmodule : memstream_tb

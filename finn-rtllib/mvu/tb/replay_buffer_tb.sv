/******************************************************************************
 * Copyright (C) 2023, Advanced Micro Devices, Inc.
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
 * @brief	Testbench for replay_buffer module.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module replay_buffer_tb;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	uwire  rst = 0;

	// DUT Geometries
	localparam int unsigned  DIMS[3] = '{ 7, 8, 10 };
	localparam int unsigned  W = 8;
	typedef logic [W-1:0]  data_t;

	bit [2**$size(DIMS)-1:0]  done = 0;
	always_comb begin
		if(&done) begin
			$display("Test completed.");
			$finish;
		end
	end

	// Parallel DUT Instantiations
	for(genvar  r = 0; r < $size(DIMS); r++) begin
		for(genvar  l = 0; l < $size(DIMS); l++) begin
			localparam int unsigned  REP = DIMS[r];
			localparam int unsigned  LEN = DIMS[l];

			data_t  idat;
			logic  ivld;
			uwire  irdy;

			uwire data_t  odat;
			uwire  olast;
			uwire  ofin;
			uwire  ovld;
			logic  ordy;

			replay_buffer #(.LEN(LEN), .REP(REP), .W(W)) dut (
				.clk, .rst,
				.idat, .ivld, .irdy,
				.odat, .olast, .ofin, .ovld, .ordy
			);

			// Input Feed: 0, 1, ..., 10*LEN-1
			initial begin
				idat = 'x;
				ivld =  0;
				@(posedge clk iff !rst);

				for(int unsigned  i = 0; i < 10*LEN; i++) begin
					idat <= i;
					ivld <= 1;
					@(posedge clk iff irdy);
					idat <= 'x;
					ivld <=  0;
					while($urandom()%(REP-1) != 0) @(posedge clk);
				end
			end

			// Output Check
			initial begin
				automatic int unsigned  base = 0;

				ordy = 0;
				@(posedge clk iff !rst);

				for(int unsigned  k = 0; k < 10; k++) begin
					for(int unsigned  j = 0; j < REP; j++) begin
						for(int unsigned  i = 0; i < LEN; i++) begin
							ordy <= 1;
							@(posedge clk iff ovld);
							assert(odat == base+i) else begin
								$error("#%0d.%0d: Data mismatch: %0d instead of %0d.", r, l, odat, base+i);
								$stop;
							end
							assert(olast == (i == LEN-1)) else begin
								$error("#%0d.%0d: Last mismatch.", r, l);
								$stop;
							end
							assert(ofin == ((i == LEN-1) && (j == REP-1))) else begin
								$error("#%0d.%0d: Fin mismatch.", r, l);
								$stop;
							end

							ordy <= 0;
							while($urandom()%13 == 0) @(posedge clk);
						end
					end
					base += LEN;
				end

				done[$size(DIMS)*r + l] <= 1;
			end
		end
	end

endmodule : replay_buffer_tb

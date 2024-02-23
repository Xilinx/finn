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
 *****************************************************************************/
module thresh_gen;
	localparam int unsigned  K = 9;
	localparam int unsigned  N = 4;
	localparam int unsigned  C = 6;

	typedef logic [K-1:0]  thresh_t;
	localparam thresh_t  THRESHOLDS[C][2**N-1] = '{
		'{ 'h00, 'h01, 'h02, 'h03, 'h04, 'h05, 'h06, 'h07, 'h08, 'h09, 'h0a, 'h0b, 'h0c, 'h0d, 'h0e },
		'{ 'h10, 'h11, 'h12, 'h13, 'h14, 'h15, 'h16, 'h17, 'h18, 'h19, 'h1a, 'h1b, 'h1c, 'h1d, 'h1e },
		'{ 'h20, 'h21, 'h22, 'h23, 'h24, 'h25, 'h26, 'h27, 'h28, 'h29, 'h2a, 'h2b, 'h2c, 'h2d, 'h2e },
		'{ 'h30, 'h31, 'h32, 'h33, 'h34, 'h35, 'h36, 'h37, 'h38, 'h39, 'h3a, 'h3b, 'h3c, 'h3d, 'h3e },
		'{ 'h40, 'h41, 'h42, 'h43, 'h44, 'h45, 'h46, 'h47, 'h48, 'h49, 'h4a, 'h4b, 'h4c, 'h4d, 'h4e },
		'{ 'h50, 'h51, 'h52, 'h53, 'h54, 'h55, 'h56, 'h57, 'h58, 'h59, 'h5a, 'h5b, 'h5c, 'h5d, 'h5e }
	};
	localparam  THRESHOLDS_PATH = "./";

	localparam int unsigned  PE = 2;
	localparam int unsigned  CF = C/PE;

	for(genvar  stage = 0; stage < N; stage++) begin
		localparam int unsigned  SN = N-1-stage;
		for(genvar  pe = 0; pe < PE; pe++) begin
			initial begin
				automatic string  file = $sformatf("%sthreshs_%0d_%0d.dat", THRESHOLDS_PATH, pe, stage);

				automatic thresh_t  threshs[CF * 2**stage];
				for(int unsigned  c = 0; c < CF; c++) begin
					for(int unsigned  i = 0; i < 2**stage; i++) begin
						threshs[(c << stage) + i] = THRESHOLDS[c*PE + pe][(i<<(N-stage)) + 2**SN-1];
					end
				end

				$writememh(file, threshs);
			end
		end
	end

    // Quit after running all initializers
	initial begin
		#1ns;
		$display("Generation done.");
		$finish;
	end

endmodule : thresh_gen

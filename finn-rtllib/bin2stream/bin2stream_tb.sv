/******************************************************************************
 *  Copyright (c) 2022, Advanced Micro Devices, Inc.
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
 * @brief	Testbench for bin2stream.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *******************************************************************************/
module bin2stream_tb #(
	parameter     BIN_FILE        = "/dev/urandom",
	int unsigned  WORDS_PER_IMAGE = 100,
	int unsigned  BITS_PER_WORD   = 13
)();
	localparam int unsigned  IMAGES = 2;

	// Global Control
	logic  clk = 0;
	always #5ns  clk = !clk;
	uwire  rst = 0;

	// DUT
	uwire [8*((BITS_PER_WORD+7)/8)-1:0]  tdata;
	uwire  tvalid;
	logic  tready = 0;
	bin2stream #(
		.BIN_FILE(BIN_FILE), .WORDS_PER_IMAGE(WORDS_PER_IMAGE), .BITS_PER_WORD(BITS_PER_WORD)
	) dut (
		.clk, .rst,
		.tdata, .tvalid, .tready
	);

	// Logging Sink
	always_ff @(posedge clk) begin : blockName
		static int  Img = 0;
		static int  Cnt = 0;

		if(rst) begin
			tready <= 0;
			Img = 0;
			Cnt = 0;
		end
		else begin
			if(tvalid || !tready)  tready <= $urandom()%11 > 2;
			if(tvalid &&  tready) begin
				$write("%04X\t", tdata);
				if(++Cnt%8 == 0)  $display();
				if(Cnt == WORDS_PER_IMAGE) begin
					$display();
					Cnt = 0;
					if(++Img == IMAGES)  $finish();
				end
			end
		end
	end

endmodule : bin2stream_tb

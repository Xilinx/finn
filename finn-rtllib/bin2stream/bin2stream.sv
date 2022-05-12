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
 * @brief	Reads a binary file to feed an AXI Stream.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 * @description
 *	The contents of BIN_FILE is read byte by byte and flattened into a bit
 *	stream. The flattening is conducted as if the file contained some
 *	high-precision little-endian integer data:
 *
 *	  ... | B4 | B3 | B2 | B1 | B0  <- start of file
 *
 *	Disregarding the byte boundaries, this bit stream is chopped up, starting
 *	at the LSB, into words of BITS_PER_WORD bits. These words are only then
 *	extended to full byte multiples to be fed to the connected AXI
 *	stream interface.
 *	While words are not aligned at byte boundaries, the image abstraction is.
 *	After the output of WORDS_PER_IMAGE words, possibly remaining bits from the
 *	last partially consumed data byte are discarded. If another byte follows,
 *	it will start the output of the next images. This alignment of images
 *	facilitates the simple concatenation of files of binary input images.
 *******************************************************************************/
module bin2stream #(
	parameter     BIN_FILE,
	int unsigned  WORDS_PER_IMAGE,
	int unsigned  BITS_PER_WORD
)(
	input	logic  clk,
	input	logic  rst,


	output	logic [8*((BITS_PER_WORD+7)/8)-1:0]  tdata,
	output	logic  tvalid,
	input	logic  tready
);
	int unsigned  fd = 0;
	initial begin
		fd = $fopen(BIN_FILE, "rb");
		if(!fd) begin
			$error("Could not read file '%s'", BIN_FILE);
			$finish;
		end
	end
	final begin
		if(fd) begin
			$fclose(fd);
			fd = 0;
		end
	end

	logic [BITS_PER_WORD+6:0]  Buf = 'x;
	int unsigned               Cnt =  0;
	int unsigned               Cyc =  0;
	always_ff @(posedge clk) begin
		if(rst) begin
			automatic int  code = $rewind(fd);
			if(code < 0) begin
				$error("Could not rewind file '%s'", BIN_FILE);
				$stop;
			end
			Buf <= 'x;
			Cnt <=  0;
			Cyc <=  0;
		end
		else begin
			automatic type(Buf)  bof = Buf;
			automatic type(Cnt)  cnt = Cnt;
			automatic type(Cyc)  cyc = Cyc;

			if(tvalid && tready) begin
				if(++cyc < WORDS_PER_IMAGE) begin
					bof >>= BITS_PER_WORD;
					cnt  -= BITS_PER_WORD;
				end
				else begin
					// Next image always aligned to byte boundary
					bof = 'x;
					cnt =  0;
					cyc =  0;
				end
			end
			while(cnt < BITS_PER_WORD) begin : blkFill
				automatic int  code = $fgetc(fd);
				if(code < 0) begin
					if(!cyc && !cnt)  break;	// No more image

					// Incomplete image
					$error("Incomplete input image in file '%s'", BIN_FILE);
					$stop;
				end
				bof[cnt+:8] = code;
				cnt += 8;
			end : blkFill

			Buf <= bof;
			Cnt <= cnt;
			Cyc <= cyc;
		end
	end
	assign	tdata  = Buf[BITS_PER_WORD-1:0];
	assign	tvalid = Cnt >= BITS_PER_WORD;

endmodule : bin2stream

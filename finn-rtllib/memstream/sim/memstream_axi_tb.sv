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

module memstream_axi_tb;
	localparam int unsigned  DEPTH = 912;
	localparam int unsigned  WIDTH =  73;
	localparam bit  PUMPED_MEMORY = 1;

	localparam int unsigned  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2;

	//- Global Control ------------------
	logic  clk = 1;
	logic  clk2x = 1;
	always #5ns clk = !clk;
	always #2.5ns clk2x = !clk2x;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	//- AXI-lite Interface --------------
	// Write
	uwire  awready;
	logic  awvalid;
	logic [AXILITE_ADDR_WIDTH-1:0]  awaddr;

	uwire  wready;
	logic  wvalid;
	logic [31:0]  wdata;

	uwire  bready = 1;
	uwire  bvalid;
	uwire [1:0]  bresp;

	// Read
	uwire  arready;
	logic  arvalid;
	logic [AXILITE_ADDR_WIDTH-1:0]  araddr;

	logic  rready;
	uwire  rvalid;
	uwire [ 1:0]  rresp;
	uwire [31:0]  rdata;

	// Streamed Output
	logic  ordy;
	uwire  ovld;
	uwire [WIDTH-1:0]  odat;

	//-----------------------------------------------------------------------
	// DUT
	memstream_axi #(.DEPTH(DEPTH), .WIDTH(WIDTH), .PUMPED_MEMORY(PUMPED_MEMORY)) dut (
		// Global Control
		.clk, .clk2x, .rst,

		// AXI-lite Write
		.awready, .awvalid, .awaddr, .awprot('x),
		.wready,  .wvalid,  .wdata,  .wstrb('1),
		.bready,  .bvalid,  .bresp,

		// AXI-lite Read
		.arready, .arvalid, .araddr, .arprot('x),
		.rready,  .rvalid,  .rdata,  .rresp,

		// Continuous output stream
		.m_axis_0_tready(ordy), .m_axis_0_tvalid(ovld), .m_axis_0_tdata(odat)
	);

	always_ff @(posedge clk iff !rst) begin
		assert(!bvalid || !bresp) else begin
			$error("Write error.");
			$stop;
		end
	end

	initial begin
		localparam int unsigned  FOLD = 1 + (WIDTH-1)/32;

		awvalid = 0;
		awaddr = 'x;
		wvalid = 0;
		wdata = 'x;
		arvalid = 0;
		araddr = 'x;
		rready = 0;
		ordy = 0;
		@(posedge clk iff !rst);

		// Configuration
		fork
			begin
				awvalid <= 1;
				for(int unsigned  i = 0; i < DEPTH; i++) begin
					automatic type(awaddr)  addr = i << $clog2(FOLD);
					for(int unsigned  j = 0; j < FOLD; j++) begin
						awaddr <= { addr++, 2'b00 };
						@(posedge clk iff awready);
					end
				end
				awvalid <= 0;
			end
			begin
				wvalid <= 1;
				for(int unsigned  i = 0; i < DEPTH; i++) begin
					automatic type(wdata)  data = i << $clog2(FOLD);
					for(int unsigned  j = 0; j < FOLD; j++) begin
						wdata <= data++;
						@(posedge clk iff wready);
					end
				end
				wvalid <= 0;
			end
		join

		// Read Last Entry for Sync
		arvalid <= 1;
		araddr <= { (DEPTH-1) << $clog2(FOLD), 2'b00 };
		@(posedge clk iff arready);
		arvalid <= 0;
		araddr <= 'x;

		rready <= 1;
		@(posedge clk iff rvalid);
		rready <= 0;
		assert(!rresp && (rdata == (DEPTH-1) << $clog2(FOLD))) else begin
			$error("Read back error.");
			$stop;
		end

		// Reset Output Pipeline
		rst <= 1;
		@(posedge clk);
		rst <= 0;

		// One Round of Unimpeded Stream Read
		ordy <= 1;
		for(int unsigned  i = 0; i < DEPTH; i++) begin
			automatic int unsigned  base = i << $clog2(FOLD);
			automatic type(odat)  exp;
			for(int unsigned  j = 0; j < FOLD; j++)  exp[32*j+:32] = base+j;

			@(posedge clk iff ovld);
			assert(odat[WIDTH-1:0] == exp[WIDTH-1:0]) else begin
				$error("Unexpected output: %0x instead of %0x [%0b]", odat, exp, odat == exp);
				$stop;
			end
		end
		ordy <= 0;

		// Another Round with Intermittent Backpressure
		for(int unsigned  i = 0; i < DEPTH; i++) begin
			automatic int unsigned  base = i << $clog2(FOLD);
			automatic type(odat)  exp;
			for(int unsigned  j = 0; j < FOLD; j++)  exp[32*j+:32] = base+j;

			while($urandom()%13 == 0) @(posedge clk);
			ordy <= 1;
			@(posedge clk iff ovld);
			ordy <= 0;

			assert(odat[WIDTH-1:0] == exp[WIDTH-1:0]) else begin
				$error("Unexpected output: %0x instead of %0x", odat, exp);
				$stop;
			end
		end

		// Yet Another Round Adding Intermittent Readbacks
		fork
			automatic bit  done = 0;

			begin
				for(int unsigned  i = 0; i < DEPTH; i++) begin
					automatic logic [WIDTH-1:0]  exp;
					for(int unsigned  j = 0; j < FOLD; j++)  exp[32*j+:32] = (i << $clog2(FOLD)) + j;

					while($urandom()%13 == 0) @(posedge clk);
					ordy <= 1;
					@(posedge clk iff ovld);
					ordy <= 0;

					assert(odat[WIDTH-1:0] == exp[WIDTH-1:0]) else begin
						$error("Unexpected output: %0x instead of %0x", odat, exp);
						$stop;
					end
				end
				done = 1;
			end
			begin
				while(!done) begin
					automatic int  av = $urandom() % DEPTH;
					repeat($urandom()%19) @(posedge clk);

					for(int unsigned  j = 0; j < FOLD; j++) begin
						automatic int unsigned  pt = (av<<$clog2(FOLD)) + j;

						araddr <= { pt, 2'b00 };
						arvalid <= 1;
						@(posedge clk iff arready);
						arvalid <= 0;
						araddr <= 'x;

						rready <= 1;
						@(posedge clk iff rvalid);
						rready <= 0;

						if(j == FOLD-1)  pt = pt[0+:WIDTH-(FOLD-1)*32];
						assert(!rresp && (rdata == pt)) else begin
							$error("Read back error: %0x instead of %0x", rdata, pt);
							$stop;
						end
					end

				end
			end
		join

		repeat(2) @(posedge clk);
		$display("Test completed.");
		$finish;
	end

endmodule : memstream_axi_tb

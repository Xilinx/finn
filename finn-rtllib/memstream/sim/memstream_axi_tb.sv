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

module memstream_axi_tb #(
	int unsigned  SETS = 3,
	int unsigned  DEPTH = 912,
	int unsigned  WIDTH = 73,
	bit  PUMPED_MEMORY = 1,
	bit  HIGH_PERFORMANCE = 0
);

	localparam int unsigned  WORDS = SETS * DEPTH;
	localparam int unsigned  FOLD = 1 + (WIDTH-1)/32;
	localparam int unsigned  AXILITE_ADDR_WIDTH = $clog2(WORDS * (2**$clog2(FOLD))) + 2;
	localparam int unsigned  SET_BITS = SETS > 2? $clog2(SETS) : 1;

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

	// Set Selection
	uwire  srdy;
	logic  svld;
	logic [SET_BITS-1:0]  sidx;
	uwire [((SET_BITS+7)/8)*8-1:0]  sidx_ba = sidx;

	// Streamed Output
	logic  ordy;
	uwire  ovld;
	uwire [((WIDTH+7)/8)*8-1:0]  odat;

	//-----------------------------------------------------------------------
	// DUT
	memstream_axi #(
		.SETS(SETS), .DEPTH(DEPTH), .WIDTH(WIDTH),
		.PUMPED_MEMORY(PUMPED_MEMORY), .HIGH_PERFORMANCE(HIGH_PERFORMANCE)
	) dut (
		// Global Control
		.clk, .clk2x, .rst,

		// AXI-lite Write
		.awready, .awvalid, .awaddr, .awprot('x),
		.wready,  .wvalid,  .wdata,  .wstrb('1),
		.bready,  .bvalid,  .bresp,

		// AXI-lite Read
		.arready, .arvalid, .araddr, .arprot('x),
		.rready,  .rvalid,  .rdata,  .rresp,

		// Set selector stream (ignored for SETS = 1)
		.s_axis_0_tready(srdy), .s_axis_0_tvalid(svld), .s_axis_0_tdata(sidx_ba),

		// Continuous output stream
		.m_axis_0_tready(ordy), .m_axis_0_tvalid(ovld), .m_axis_0_tdata(odat)
	);

	always_ff @(posedge clk iff !rst) begin
		assert(!bvalid || !bresp) else begin
			$error("Write error.");
			$stop;
		end
	end

	//-----------------------------------------------------------------------
	// Stimuli and Checking

	// Memory word w holds the 32-bit values (w << $clog2(FOLD)) + j in its folds j.
	function logic [WIDTH-1:0] word(input int unsigned  w);
		automatic logic [32*FOLD-1:0]  res;
		for(int unsigned  j = 0; j < FOLD; j++)  res[32*j+:32] = (w << $clog2(FOLD)) + j;
		return  res[WIDTH-1:0];
	endfunction : word

	// Stream the given sequence of sets and check the output, optionally with random backpressure.
	task automatic stream_sets(input int unsigned  sets[$], input bit  throttle);
		fork
			if(SETS > 1) begin
				svld <= 1;
				foreach(sets[k]) begin
					sidx <= sets[k];
					@(posedge clk iff srdy);
				end
				svld <= 0;
				sidx <= 'x;
			end
			begin
				ordy <= !throttle;
				foreach(sets[k]) begin
					for(int unsigned  i = 0; i < DEPTH; i++) begin
						automatic logic [WIDTH-1:0]  exp = word(sets[k]*DEPTH + i);
						if(throttle) begin
							while($urandom()%13 == 0) @(posedge clk);
							ordy <= 1;
						end
						@(posedge clk iff ovld);
						if(throttle)  ordy <= 0;
						assert(odat[WIDTH-1:0] == exp) else begin
							$error("Unexpected output of set %0d: %0x instead of %0x", sets[k], odat, exp);
							$stop;
						end
					end
				end
				ordy <= 0;
			end
		join
	endtask : stream_sets

	initial begin
		automatic int unsigned  sets[$];

		awvalid = 0;
		awaddr = 'x;
		wvalid = 0;
		wdata = 'x;
		arvalid = 0;
		araddr = 'x;
		rready = 0;
		svld = 0;
		sidx = 'x;
		ordy = 0;
		@(posedge clk iff !rst);

		// Configuration
		fork
			begin
				awvalid <= 1;
				for(int unsigned  i = 0; i < WORDS; i++) begin
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
				for(int unsigned  i = 0; i < WORDS; i++) begin
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
		araddr <= { (WORDS-1) << $clog2(FOLD), 2'b00 };
		@(posedge clk iff arready);
		arvalid <= 0;
		araddr <= 'x;

		rready <= 1;
		@(posedge clk iff rvalid);
		rready <= 0;
		// Fold 0 of the last entry, which retains only WIDTH bits if WIDTH < 32
		assert(!rresp && (rdata == ((WORDS-1) << $clog2(FOLD) & (FOLD > 1? '1 : 2**WIDTH-1)))) else begin
			$error("Read back error.");
			$stop;
		end

		// Reset Output Pipeline
		rst <= 1;
		@(posedge clk);
		rst <= 0;

		// One Round of Unimpeded Stream Reads of all Sets in Sequence
		sets = {};
		for(int unsigned  s = 0; s < SETS; s++)  sets.push_back(s);
		stream_sets(sets, 0);

		// Another Round in reverse Order with Intermittent Backpressure
		sets.reverse();
		stream_sets(sets, 1);

		// Yet Another Round of Random Sets Adding Intermittent Readbacks
		sets = {};
		repeat(SETS < 2? 1 : 2*SETS)  sets.push_back($urandom()%SETS);
		fork
			automatic bit  done = 0;

			begin
				stream_sets(sets, 1);
				done = 1;
			end
			begin
				while(!done) begin
					automatic int unsigned  av = $urandom() % WORDS;
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

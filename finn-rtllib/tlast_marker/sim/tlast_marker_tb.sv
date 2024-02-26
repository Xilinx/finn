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
 * @brief	Testbench for tlast_marker module.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/
module tlast_marker_tb;
	localparam int unsigned           DATA_WIDTH  =  8;
	localparam int unsigned           PERIOD_BITS =  5;
	localparam bit [PERIOD_BITS-1:0]  PERIOD_INIT = 12;
	localparam bit                    PERIOD_INIT_UPON_RESET = 0;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	// AXI-lite Configuration
	logic  s_axilite_AWVALID;
	uwire  s_axilite_AWREADY;
	logic [0:0]  s_axilite_AWADDR;
	logic  s_axilite_WVALID;
	uwire  s_axilite_WREADY;
	logic [PERIOD_BITS - 1:0]  s_axilite_WDATA;
	uwire  s_axilite_BVALID;
	logic  s_axilite_BREADY;
	uwire [1:0]  s_axilite_BRESP;

	logic  s_axilite_ARVALID;
	uwire  s_axilite_ARREADY;
	logic [0:0]  s_axilite_ARADDR;
	uwire  s_axilite_RVALID;
	logic  s_axilite_RREADY;
	uwire [PERIOD_BITS - 1:0]  s_axilite_RDATA;
	uwire [1:0]  s_axilite_RRESP;

	// Input Stream without TLAST marker
	logic [DATA_WIDTH-1:0]  src_TDATA;
	logic  src_TVALID;
	uwire  src_TREADY;

	// Output Stream with TLAST marker
	uwire [DATA_WIDTH-1:0]  dst_TDATA;
	uwire  dst_TVALID;
	logic  dst_TREADY;
	uwire  dst_TLAST;

	tlast_marker #(.DATA_WIDTH(DATA_WIDTH), .PERIOD_BITS(PERIOD_BITS), .PERIOD_INIT(PERIOD_INIT), .PERIOD_INIT_UPON_RESET(PERIOD_INIT_UPON_RESET)) dut (
		.ap_clk(clk), .ap_rst_n(!rst),

		.s_axilite_AWVALID, .s_axilite_AWREADY, .s_axilite_AWADDR,
		.s_axilite_WVALID,  .s_axilite_WREADY,  .s_axilite_WDATA,
		.s_axilite_BVALID,  .s_axilite_BREADY,  .s_axilite_BRESP,

		.s_axilite_ARVALID, .s_axilite_ARREADY, .s_axilite_ARADDR,
		.s_axilite_RVALID, .s_axilite_RREADY, .s_axilite_RDATA, .s_axilite_RRESP,

		.src_TDATA, .src_TVALID, .src_TREADY,
		.dst_TDATA, .dst_TVALID, .dst_TREADY, .dst_TLAST
	);

	task run(
		input int unsigned  rep0,
		input int unsigned  rep1,
		input int unsigned  period
	);
		repeat(rep0) begin
			fork
				// Feed 3 complete periods
				begin
					for(int unsigned  i = 0; i < rep1*period; i++) begin
						while($urandom()%11 == 0) @(posedge clk);
						src_TDATA <= i;
						src_TVALID <= 1;
						@(posedge clk iff src_TREADY);
						src_TDATA <= 'x;
						src_TVALID <= 0;
					end
				end

				// Consume and check 3 complete periods
				begin
					for(int unsigned  i = 0; i < rep1*period; i++) begin
						while($urandom()%11 == 0) @(posedge clk);
						dst_TREADY <= 1;
						@(posedge clk iff dst_TVALID);
						assert(dst_TDATA == i) else begin
							$error("Unexpected output: %0x instead of %0x", dst_TDATA, i);
							$stop;
						end
						assert(dst_TLAST == ((i+1)%period == 0)) else begin
							$error("Wrong TLAST flag %0b", dst_TLAST);
							$stop;
						end
						dst_TREADY <= 0;
					end
				end
			join
			rst <= 1;
			@(posedge clk);
			rst <= 0;
		end
	endtask


	initial begin
		s_axilite_AWVALID = 0;
		s_axilite_AWADDR = 'x;
		s_axilite_WVALID = 0;
		s_axilite_WDATA = 'x;
		s_axilite_BREADY = 0;
		s_axilite_ARVALID = 0;
		s_axilite_ARADDR = 'x;
		s_axilite_RREADY = 0;
		src_TDATA = 'x;
		src_TVALID = 0;
		dst_TREADY = 0;
		@(posedge clk iff !rst);

		// Two runs seperated by a reset
		run(2, 3, PERIOD_INIT);

		// Reconfigure period to doubled value
		fork
			begin
				s_axilite_AWADDR <= '0;
				s_axilite_AWVALID <= 1;
				@(posedge clk iff s_axilite_AWREADY);
				s_axilite_AWADDR <= 'x;
				s_axilite_AWVALID <= 0;
			end
			begin
				s_axilite_WDATA <= 2*PERIOD_INIT;
				s_axilite_WVALID <= 1;
				@(posedge clk iff s_axilite_WREADY);
				s_axilite_WDATA <= 'x;
				s_axilite_WVALID <= 0;
			end
			begin
				while($urandom()%3 == 0) @(posedge clk);
				s_axilite_BREADY <= 1;
				@(posedge clk iff s_axilite_BVALID);
				assert(s_axilite_BRESP == 0) else begin
					$error("Config write error.");
					$stop;
				end
				s_axilite_BREADY <= 0;
			end
		join

		// Two runs seperated by a reset
		run(2, 3, 2*PERIOD_INIT);

		$display("Test completed.");
		$finish;
	end

endmodule : tlast_marker_tb

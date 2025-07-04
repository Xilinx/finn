/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *	 this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *	 contributors may be used to endorse or promote products derived from
 *	 this software without specific prior written permission.
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

module axilite_tb #(
	int unsigned  AXI_DATA_WIDTH = 32,
	int unsigned  IP_ADDR_WIDTH  = 11,
	int unsigned  IP_DATA_WIDTH  = 83
)();
	localparam int unsigned  IP_DATA_FOLD   = 1 + (IP_DATA_WIDTH-1)/AXI_DATA_WIDTH;
	localparam int unsigned  IP_DATA_STRIDE = 2**$clog2(IP_DATA_FOLD);
	localparam int unsigned  AXI_ADDR_WIDTH = $clog2(IP_DATA_FOLD) + IP_ADDR_WIDTH + 2;
	typedef logic [IP_ADDR_WIDTH-1:0]  addr_t;
	typedef logic [IP_DATA_WIDTH-1:0]  data_t;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// DUT

	// Write Channels
	uwire  awready;
	logic  awvalid;
	logic [AXI_ADDR_WIDTH-1:0]  awaddr;
	uwire  wready;
	logic  wvalid;
	logic [AXI_DATA_WIDTH-1:0]  wdata;
	logic  bready;
	uwire  bvalid;
	uwire [1:0]  bresp;

	// Read Channels
	uwire  arready;
	logic  arvalid;
	logic [AXI_ADDR_WIDTH-1:0]  araddr;
	logic  rready;
	uwire  rvalid;
	uwire [1:0]  rresp;
	uwire [AXI_DATA_WIDTH-1:0]  rdata;

	// IP-side Interface
	uwire  ip_en;
	uwire  ip_wen;
	uwire [IP_ADDR_WIDTH-1:0]  ip_addr;
	uwire [IP_DATA_WIDTH-1:0]  ip_wdata;
	logic  ip_rack;
	logic [IP_DATA_WIDTH-1:0]  ip_rdata;

	axilite #(
		.ADDR_WIDTH(AXI_ADDR_WIDTH),
		.DATA_WIDTH(AXI_DATA_WIDTH),
		.IP_DATA_WIDTH(IP_DATA_WIDTH)
	) dut (
		.aclk(clk), .aresetn(!rst),

		.awready, .awvalid, .awprot('x), .awaddr,
		.wready,  .wvalid,  .wstrb('1),  .wdata,
		.bready, .bvalid, .bresp,

		.arready, .arvalid, .arprot('x), .araddr,
		.rready,  .rvalid,  .rresp,      .rdata,

		.ip_en, .ip_wen,
		.ip_addr, .ip_wdata,
		.ip_rack, .ip_rdata
	);

	//-----------------------------------------------------------------------
	// Client Stub
	if(1) begin : blkClient
		data_t  Mem[2**IP_ADDR_WIDTH] = '{ default: 'x };

		data_t  ReplyQ[$];
		logic   RAck = 0;
		data_t  RData = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				ReplyQ <= {};
				RAck  <= 0;
				RData <= 'x;
			end
			else begin
				// Read Reply
				RAck <= 0;
				RData <= 'x;
				if(ReplyQ.size() && ($urandom()%7 != 0)) begin
					RAck <= 1;
					RData <= ReplyQ.pop_front();
				end

				// Request Issuing
				if(ip_en) begin
					if(ip_wen)  Mem[ip_addr] <= ip_wdata;
					else  ReplyQ.push_back(Mem[ip_addr]);
				end
			end
		end
		assign	ip_rack = RAck;
		assign	ip_rdata = RData;

	end : blkClient

	//-----------------------------------------------------------------------
	// Stimulus
	task issue_write(input addr_t  addr, input data_t  data);
		fork
			// Address
			for(int unsigned  j = 0; j < IP_DATA_FOLD; j++) begin
				while($urandom()%13 == 0) @(posedge clk);
				awvalid <= 1;
				awaddr  <= (addr*IP_DATA_STRIDE + j) << 2;
				@(posedge clk iff awready);
				awvalid <= 0;
				awaddr  <= 'x;
			end
			// Data
			for(int unsigned  j = 0; j < IP_DATA_FOLD; j++) begin
				while($urandom()%13 == 0) @(posedge clk);
				wvalid <= 1;
				wdata  <= data;
				@(posedge clk iff wready);
				data = { {(AXI_DATA_WIDTH){1'bx}}, data } >> AXI_DATA_WIDTH;
				wvalid <= 0;
				wdata  <= 'x;
			end
		join
	endtask : issue_write
	task issue_read(input addr_t  addr);
		// Address
		for(int unsigned  j = 0; j < IP_DATA_FOLD; j++) begin
			while($urandom()%13 == 0) @(posedge clk);
			arvalid <= 1;
			araddr  <= (addr*IP_DATA_STRIDE + j) << 2;
			@(posedge clk iff arready);
			arvalid <= 0;
			araddr  <= 'x;
		end
	endtask : issue_read

	data_t  RefMem[2**IP_ADDR_WIDTH] = '{ default: 'x };
	int unsigned  WrCnt = 0;
	logic [AXI_DATA_WIDTH-1:0]  RdQ[$];
	initial begin

		// Write Channels
		awvalid = 0;
		awaddr = 'x;
		wvalid = 0;
		wdata = 'x;
		bready = 0;

		// Read Channels
		arvalid = 0;
		araddr = 'x;
		rready = 0;

		@(posedge clk iff !rst);

		// Full Initialization
		foreach(RefMem[i]) begin
			automatic data_t  val = i;

			WrCnt += IP_DATA_FOLD;
			issue_write(i, val);
			RefMem[i] = val;
		end

		// Randomized Readbacks and Occassional Contents Update
		repeat(1234) begin
			automatic addr_t  addr;
			automatic data_t  data;
			std::randomize(addr);

			if($urandom()%31 == 0) @(posedge clk);
			randcase
			17: begin // Readback
				data = RefMem[addr];
				for(int unsigned  j = 0; j < IP_DATA_FOLD; j++) begin
					RdQ.push_back(data);
					data >>= AXI_DATA_WIDTH;
				end
				issue_read(addr);
			end
			1: begin // Update
				std::randomize(data);

				WrCnt += IP_DATA_FOLD;
				issue_write(addr, data);
				RefMem[addr] = data;
			end
			endcase
		end

		repeat(20) @(posedge clk);
		assert(WrCnt == 0) else begin
			$error("Missing write confirmation.");
			$stop;
		end
		assert(RdQ.size() == 0) else begin
			$error("Missing reply.");
			$stop;
		end

		$display("Test completed.");
		$finish;
	end

	// Write Confirmation Checker
	initial begin
		bready = 0;
		@(posedge clk iff !rst);

		forever begin
			while($urandom()%23 == 0) @(posedge clk);
			bready <= 1;
			@(posedge clk iff bvalid);
			assert(bresp == '0) else begin
				$error("Write error.");
				$stop;
			end
			assert(WrCnt > 0) else begin
				$error("Spurious write confirmation.");
				$stop;
			end
			bready <= 0;
			WrCnt--;
		end
	end

	// Read Reply Checker
	initial begin
		rready = 0;
		@(posedge clk iff !rst);

		forever begin
			automatic logic [AXI_DATA_WIDTH-1:0]  exp;

			while($urandom()%23 == 0) @(posedge clk);
			rready <= 1;
			@(posedge clk iff rvalid);
			assert(rresp == '0) else begin
				$error("Read error.");
				$stop;
			end
			assert(RdQ.size() > 0) else begin
				$error("Spurious reply.");
				$stop;
			end
			exp = RdQ.pop_front();
			assert(rdata == exp) else begin
				$error("Read back %0x instead of %0x.", rdata, exp);
				$stop;
			end
			rready <= 0;
		end
	end

endmodule : axilite_tb

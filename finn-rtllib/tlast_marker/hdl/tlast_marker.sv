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
 * @brief	TLAST marker insertion.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	Inserts a TLAST marker on an AXI-Stream according to a configured period.
 *	Every period-th stream transaction will be annotated with an asserted
 *	TLAST flag on the `dst` output stream. Otherwise, both the `src` and `dst`
 *	AXI-Stream interfaces execute identical transactions.
 *	The initial period setting is determined by the PERIOD_INIT parameter.
 *	If the parameter PERIOD_INIT_UPON_RESER is set, this value will also be
 *	restored by a reset. Otherwise, a reset will not affect the period
 *	setting. The period setting may be changed via the AXI-lite configuration
 *	interface. Any performed write (irrespective of address) will update the
 *	configured period within the limits of the reserved register width of
 *	PERIOD_BITS bits. This setting will take immediate effect only in a clean
 *	state after reset or when the most recent stream transaction had an
 *	asserted `TLAST` flag. Otherwise, the transmission will continue
 *	eventually asserting the `TLAST` flag according to the period setting
 *	at its beginning. The new period setting (or any update performed in the
 *	meantime) will be adopted for the subsequent transmission.
 *	The current period setting can also be read back via the AXI-lite
 *	interface.
 *****************************************************************************/
module tlast_marker #(
	int unsigned           DATA_WIDTH,
	int unsigned           PERIOD_BITS,
	bit [PERIOD_BITS-1:0]  PERIOD_INIT,
	bit                    PERIOD_INIT_UPON_RESET = 0
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// AXI-lite Configuration
	input	logic  s_axilite_AWVALID,
	output	logic  s_axilite_AWREADY,
	input	logic [0:0]  s_axilite_AWADDR,
	input	logic  s_axilite_WVALID,
	output	logic  s_axilite_WREADY,
	input	logic [PERIOD_BITS - 1:0]  s_axilite_WDATA,
	output	logic  s_axilite_BVALID,
	input	logic  s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	input	logic  s_axilite_ARVALID,
	output	logic  s_axilite_ARREADY,
	input	logic [0:0]  s_axilite_ARADDR,
	output	logic  s_axilite_RVALID,
	input	logic  s_axilite_RREADY,
	output	logic [PERIOD_BITS - 1:0]  s_axilite_RDATA,
	output	logic [1:0]  s_axilite_RRESP,

	// Input Stream without TLAST marker
	input	logic [DATA_WIDTH-1:0]  src_TDATA,
	input	logic  src_TVALID,
	output	logic  src_TREADY,

	// Output Stream with TLAST marker
	output	logic [DATA_WIDTH-1:0]  dst_TDATA,
	output	logic  dst_TVALID,
	input	logic  dst_TREADY,
	output	logic  dst_TLAST
);

	// Just wire data stream and flow control
	assign	dst_TDATA  = src_TDATA;
	assign	dst_TVALID = src_TVALID;
	assign	src_TREADY = dst_TREADY;

	// Configuration Interface
	logic [PERIOD_BITS-1:0]  Period = PERIOD_INIT;
	logic  Update = 0;
	if(1) begin : blkCfg

		// Write Period
		logic  WAddrRdy = 1;
		logic  WDataRdy = 1;
		logic [PERIOD_BITS-1:0]  WData = 'x;
		logic  BVld = 0;

		uwire  wr = !WAddrRdy && !WDataRdy && !BVld;
		always_ff @(posedge ap_clk) begin
			if(!ap_rst_n || wr) begin
				WAddrRdy <= 1;
				WDataRdy <= 1;
				WData <= 'x;
			end
			else begin
				if(WAddrRdy && s_axilite_AWVALID) begin
					WAddrRdy <= 0;
				end
				if(WDataRdy && s_axilite_WVALID) begin
					WData    <= s_axilite_WDATA;
					WDataRdy <= 0;
				end
			end
		end
		always_ff @(posedge ap_clk) begin
			if(!ap_rst_n)              BVld <= 0;
			else if(wr)                BVld <= 1;
			else if(s_axilite_BREADY)  BVld <= 0;
		end
		assign	s_axilite_AWREADY = WAddrRdy;
		assign	s_axilite_WREADY  = WDataRdy;
		assign	s_axilite_BVALID  = BVld;
		assign	s_axilite_BRESP   = 2'b00;

		always_ff @(posedge ap_clk) begin
			if(PERIOD_INIT_UPON_RESET && !ap_rst_n) begin
				Period <= PERIOD_INIT;
				Update <= 0;
			end
			else begin
				Update <= 0;
				if(wr) begin
					Period <= WData;
					Update <= 1;
				end
			end
		end

		// Read back period
		logic  RDataVld = 0;
		always_ff @(posedge ap_clk) begin
			if(!ap_rst_n)  RDataVld <= 0;
			else  RDataVld <= RDataVld? !s_axilite_RREADY : s_axilite_ARVALID;
		end
		assign	s_axilite_ARREADY = !RDataVld;
		assign	s_axilite_RVALID  = RDataVld;
		assign	s_axilite_RDATA   = Period;
		assign	s_axilite_RRESP   = 2'b00;

	end : blkCfg

	// Compute TLAST by counting
	logic signed [PERIOD_BITS:0]  Cnt = PERIOD_INIT;  // Period-2, Period-3, ...,  1, 0, -1 (last)
	logic  Clean = 1;

	uwire  last   = Cnt[$left(Cnt)];
	uwire  cnt_en = !ap_rst_n || (Update && Clean) || (src_TVALID && dst_TREADY);
	always_ff @(posedge ap_clk) begin
		if(cnt_en) begin
			automatic logic signed [PERIOD_BITS:0]  a = Cnt;
			automatic logic signed [PERIOD_BITS:0]  b = -1;

			Clean <= 0;
			if(!ap_rst_n || (Update && Clean) || last) begin
				b[0] = 0;	// b = -2
				a = Period;
				if(PERIOD_INIT_UPON_RESET && !ap_rst_n)  a = PERIOD_INIT;
				Clean <= 1;
			end
			Cnt <= a + b;
		end
	end
	assign	dst_TLAST = last;

endmodule : tlast_marker

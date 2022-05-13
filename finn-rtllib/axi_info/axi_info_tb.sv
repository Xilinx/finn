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
 * @brief	Read-only exposure of compiled-in info data on AXI-lite.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 *******************************************************************************/
module axi_info_tb #(
	int unsigned  N = 9,
	int unsigned  S_AXI_DATA_WIDTH = 32,
	bit [S_AXI_DATA_WIDTH-1:0]  DATA[N] = '{
		32'h2437_9827,
		32'ha639_bf83,
		32'haec2_3ab1,
		32'h5ab3_2b97,
		32'hff1c_4e78,
		32'h10c2_4968,
		32'h4537_a1ec,
		32'h694b_63d1,
		32'h7f77_9af9
	}
)();

	//- Global Control ------------------
	logic  ap_clk = 0;
	always #5ns ap_clk = !ap_clk;
	logic  ap_rst_n = 1;

	//- AXI Lite ------------------------
	// Writing
	uwire                  s_axi_AWVALID = 0;
	uwire                  s_axi_AWREADY;
	uwire [$clog2(N)-1:0]  s_axi_AWADDR = 'x;

	uwire                           s_axi_WVALID = 0;
	uwire                           s_axi_WREADY;
	uwire [S_AXI_DATA_WIDTH  -1:0]  s_axi_WDATA = 'x;
	uwire [S_AXI_DATA_WIDTH/8-1:0]  s_axi_WSTRB = 'x;

	uwire        s_axi_BVALID;
	uwire        s_axi_BREADY = 0;
	uwire [1:0]  s_axi_BRESP;

	// Reading
	logic                  s_axi_ARVALID;
	uwire                  s_axi_ARREADY;
	logic [$clog2(N)-1:0]  s_axi_ARADDR;

	uwire                         s_axi_RVALID;
	logic                         s_axi_RREADY = 0;
	uwire [S_AXI_DATA_WIDTH-1:0]  s_axi_RDATA;
	uwire [                 1:0]  s_axi_RRESP;

	axi_info #(.N(N), .S_AXI_DATA_WIDTH(S_AXI_DATA_WIDTH), .DATA(DATA)) dut (
		.ap_clk, .ap_rst_n,

		.s_axi_AWVALID, .s_axi_AWREADY, .s_axi_AWADDR,
		.s_axi_WVALID, .s_axi_WREADY, .s_axi_WDATA, .s_axi_WSTRB,
		.s_axi_BVALID, .s_axi_BREADY, .s_axi_BRESP,

		.s_axi_ARVALID, .s_axi_ARREADY, .s_axi_ARADDR,
		.s_axi_RVALID, .s_axi_RREADY, .s_axi_RDATA, .s_axi_RRESP
	);

	//-----------------------------------------------------------------------
	// Read address feed
	initial begin
		s_axi_ARVALID =  0;
		s_axi_ARADDR  = 'x;
		@(posedge ap_clk iff ap_rst_n);
		for(int unsigned  i = 0; i < N; i++) begin
			repeat($urandom()%3 > 0);
			s_axi_ARVALID <= 1;
			s_axi_ARADDR  <= i;
			@(posedge ap_clk iff s_axi_ARREADY);
			s_axi_ARVALID <=  0;
			s_axi_ARADDR  <= 'x;
		end
	end

	//-----------------------------------------------------------------------
	// Read reply check
	always_ff @(posedge ap_clk) begin
		static int  Cnt = 0;
		if(!ap_rst_n) begin
			s_axi_RREADY <= 0;
			Cnt = 0;
		end
		else begin
			if(s_axi_RVALID || !s_axi_RREADY)  s_axi_RREADY <= $urandom()%7 > 2;
			if(s_axi_RVALID &&  s_axi_RREADY) begin
				assert(s_axi_RRESP === 2'b00) else begin
					$error("AXI read error indicator: %0d", s_axi_RRESP);
					$stop;
				end
				assert(s_axi_RDATA === DATA[Cnt]) else begin
					$error("Unexpected read reply: 0x%08x instead of 0x%08x", s_axi_RDATA, DATA[Cnt]);
					$stop;
				end
				if(++Cnt == N)  $finish;
			end
		end
	end

endmodule : axi_info_tb

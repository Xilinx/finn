/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	All-AXI interface adapter for thresholding module.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 * @description
 *	This AXI adapter fits the core thresholding functionality:
 *	- with AXI stream data interfaces with flow control
 *	- with implicit round-robin channel rotation as used by FINN, and
 *	- performs aligned byte address to parameter word address translation.
 *****************************************************************************/

module thresholding_axi #(
	int unsigned  N,	// output precision
	int unsigned  K,	// input/threshold precision
	int unsigned  C,	// Channels
	int unsigned  PE,	// Processing Parallelism, requires C = k*PE

	bit  SIGNED = 1,	// signed inputs
	int  BIAS   = 0,	// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	localparam int unsigned  CF = 1 + (C-1)/PE,	// Channel Fold
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(PE) + N + 2,
	localparam int unsigned  O_BITS = BIAS >= 0?
		/* unsigned */ $clog2(2**N+BIAS) :
		/* signed */ 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  s_axilite_AWVALID,
	output	logic                  s_axilite_AWREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,

	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	// Reading
	input	logic        s_axilite_ARVALID,
	output	logic        s_axilite_ARREADY,
	input	logic [0:0]  s_axilite_ARADDR,

	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [((PE*K+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  m_axis_tdata
);
	//- Parameter Constraints Checking --------------------------------------
	initial begin
		if(C%PE != 0) begin
			$error("%m: Channel count C=%0d is not a multiple of PE=%0d.", C, PE);
			$finish;
		end
	end

	//- Global Control ------------------------------------------------------
	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;

	//- AXI Lite: Threshold Configuration -----------------------------------
	uwire  twe[PE];
	uwire [$clog2(CF)+N-1:0]  twa;
	uwire [           K-1:0]  twd;
	if(1) begin : blkAxiLite
		logic  WABusy = 0;
		logic  WDBusy = 0;
		logic  Sel[PE] = '{ default: 'x };
		logic [$clog2(CF)+N-1:0]  Addr = 'x;
		logic [           K-1:0]  Data = 'x;

		for(genvar  pe = 0; pe < PE; pe++) begin
			assign	twe[pe] = WABusy && WDBusy && Sel[pe];
		end
		assign	twa = Addr;
		assign	twd = Data;

		if(PE == 1)  always_comb  Sel[0] = 1;
		else begin
			always_ff @(posedge clk) begin
				if(!WABusy) begin
					foreach(Sel[pe])  Sel[pe] <= s_axilite_AWADDR[N+2+:$clog2(PE)] == pe;
				end
			end
		end

		uwire  clr_wr = rst || (WABusy && WDBusy && s_axilite_BREADY);
		always_ff @(posedge clk) begin
			if(clr_wr) begin
				WABusy <= 0;
				Addr <= 'x;
				WDBusy <= 0;
				Data <= 'x;
			end
			else begin
				if(!WABusy) begin
					WABusy <= s_axilite_AWVALID;
					Addr[0+:N] <= s_axilite_AWADDR[2+:N];
					if(CF > 1)  Addr[N+:$clog2(CF)] <= s_axilite_AWADDR[2+N+$clog2(PE)+:$clog2(CF)];
				end
				if(!WDBusy) begin
					WDBusy <= s_axilite_WVALID;
					Data   <= s_axilite_WDATA[K-1:0];
				end
			end
		end
		assign	s_axilite_AWREADY = !WABusy;
		assign	s_axilite_WREADY  = !WDBusy;
		assign	s_axilite_BVALID  = WABusy && WDBusy;
		assign	s_axilite_BRESP   = '0; // OK

		// Answer all reads with '1
		logic  RValid =  0;
		uwire  clr_rd = rst || (RValid && s_axilite_RREADY);
		always_ff @(posedge clk) begin
			if(clr_rd)        RValid <=  0;
			else if(!RValid)  RValid <= s_axilite_ARVALID;
		end
		assign	s_axilite_ARREADY = !RValid;
		assign	s_axilite_RVALID  = RValid;
		assign	s_axilite_RDATA   = '1;
		assign	s_axilite_RRESP   = '0; // OK

	end : blkAxiLite

	//- IO-Sandwich with two-stage output buffer for containing a local enable
	uwire  en;
	uwire [PE-1:0][O_BITS-1:0]  odat;
	uwire  ovld[PE];
	if(1) begin : blkOutputDecouple
		typedef struct {
			logic  vld;
			logic [PE-1:0][O_BITS-1:0]  dat;
		} buf_t;
		buf_t  A = '{ vld: 0, dat: 'x };
		buf_t  B = '{ vld: 0, dat: 'x };
		always_ff @(posedge clk) begin
			if(rst) begin
				A <= '{ vld: 0, dat: 'x };
				B <= '{ vld: 0, dat: 'x };
			end
			else begin
				if(!B.vld || m_axis_tready) begin
					B <= '{
						vld: A.vld || ovld[0],
						dat: A.vld? A.dat : odat
					};
				end
				A.vld <= B.vld && !m_axis_tready && (A.vld || ovld[0]);
				if(!A.vld)  A.dat <= odat;
			end
		end
		assign	en = !A.vld;

		assign	m_axis_tvalid = B.vld;
		assign	m_axis_tdata  = B.dat;

	end : blkOutputDecouple

	localparam int unsigned  C_BITS = C/PE < 2? 1 : $clog2(C/PE);
	uwire  ivld = s_axis_tvalid;
	uwire [C_BITS-1:0]  icnl;
	uwire [K     -1:0]  idat[PE];
	for(genvar  pe = 0; pe < PE; pe++) begin
		assign	idat[pe] = s_axis_tdata[pe*K+:K];
	end

	assign	s_axis_tready = en;
	if(C == PE)  assign  icnl = 'x;
	else begin
		logic [C_BITS-1:0]  Chnl = 0;
		logic               Last = 0;
		uwire  inc = ivld && en;
		uwire  clr = rst || (Last && inc);
		always_ff @(posedge clk) begin
			if(clr) begin
				Chnl <= 0;
				Last <= 0;
			end
			else if(inc) begin
				Chnl <= Chnl + 1;
				Last <= (~Chnl & (C/PE-2)) == 0;
			end
		end
		assign	icnl = Chnl;
	end

	// Core Thresholding Modules
	for(genvar  pe = 0; pe < PE; pe++) begin : genCores
		thresholding #(.N(N), .K(K), .C(C/PE), .SIGNED(SIGNED), .BIAS(BIAS)) core (
			.clk, .rst,
			.twe(twe[pe]), .twa, .twd,
			.en,
			.ivld, .icnl, .idat(idat[pe]),
			.ovld(ovld[pe]), .ocnl(), .odat(odat[pe])
		);
	end : genCores

endmodule : thresholding_axi

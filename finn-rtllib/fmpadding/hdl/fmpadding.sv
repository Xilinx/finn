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
 * @brief	Feature map padding.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module fmpadding #(
	int unsigned  XCOUNTER_BITS,
	int unsigned  YCOUNTER_BITS,
	int unsigned  NUM_CHANNELS,
	int unsigned  SIMD,
	int unsigned  ELEM_BITS,
	int unsigned  INIT_XON,
	int unsigned  INIT_XOFF,
	int unsigned  INIT_XEND,
	int unsigned  INIT_YON,
	int unsigned  INIT_YOFF,
	int unsigned  INIT_YEND,
	bit           EAGER = 1,

	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Parameter Configuration ----------
	input	logic         we,
	input	logic [ 4:0]  wa,
	input	logic [31:0]  wd,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_tdata
);
	`default_nettype none

	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;

	//=== Parameter Sanity Checking ==========================================
	initial begin
		automatic bit  fail = 0;

		if(XCOUNTER_BITS < $clog2(1+INIT_XEND)) begin
			$error("XCounter size too small to accommodate end count.");
			fail = 1;
		end
		if(XCOUNTER_BITS < $clog2(1+INIT_XON)) begin
			$error("XCounter size too small to accommodate ON count.");
			fail = 1;
		end
		if(XCOUNTER_BITS < $clog2(1+INIT_XOFF)) begin
			$error("XCounter size too small to accommodate OFF count.");
			fail = 1;
		end
		if(YCOUNTER_BITS < $clog2(1+INIT_YEND)) begin
			$error("YCounter size too small to accommodate end count.");
			fail = 1;
		end
		if(YCOUNTER_BITS < $clog2(1+INIT_YON)) begin
			$error("YCounter size too small to accommodate ON count.");
			fail = 1;
		end
		if(YCOUNTER_BITS < $clog2(1+INIT_YOFF)) begin
			$error("YCounter size too small to accommodate OFF count.");
			fail = 1;
		end

		if((INIT_XEND < INIT_XON) || (INIT_XOFF <= INIT_XON)) begin
			$warning("Initial empty X output range.");
		end
		if((INIT_YEND < INIT_YON) || (INIT_YOFF <= INIT_YON)) begin
			$warning("Initial empty Y output range.");
		end

		if(fail)  $finish();
	end

	//=== Dynamically Configurable State =====================================
	typedef logic [XCOUNTER_BITS-1:0]  xcount_t;
	xcount_t  XEnd = INIT_XEND;
	xcount_t  XOn  = INIT_XON;
	xcount_t  XOff = INIT_XOFF;

	typedef logic [YCOUNTER_BITS-1:0]  ycount_t;
	ycount_t  YEnd = INIT_YEND;
	ycount_t  YOn  = INIT_YON;
	ycount_t  YOff = INIT_YOFF;

	always_ff @(posedge clk) begin
		if(we) begin
			unique case(wa)
			0*4:  XOn  <= wd;
			1*4:  XOff <= wd;
			2*4:  XEnd <= wd;
			3*4:  YOn  <= wd;
			4*4:  YOff <= wd;
			5*4:  YEnd <= wd;

			default:  assert(0) else begin
				$error("Illegal write address.");
			end
			endcase
		end
	end

	//=== Padding Core =======================================================
	localparam int unsigned  DATA_BITS = SIMD*ELEM_BITS;
	uwire [SIMD-1:0][ELEM_BITS-1:0]  odat;

	pad #(
		.NUM_CHANNELS(NUM_CHANNELS), .SIMD(SIMD), .ELEM_BITS(ELEM_BITS),
		.XCOUNTER_BITS(XCOUNTER_BITS), .YCOUNTER_BITS(YCOUNTER_BITS),
		.EAGER(EAGER)
	) core (
		.clk, .rst,

		.xon(XOn), .xoff(XOff), .xend(XEnd),
		.yon(YOn), .yoff(YOff), .yend(YEnd),

		.irdy(s_axis_tready), .ivld(s_axis_tvalid), .idat(s_axis_tdata[DATA_BITS-1:0]),
		.ordy(m_axis_tready), .ovld(m_axis_tvalid), .odat
	);
	assign	m_axis_tdata = STREAM_BITS'(odat);

	`default_nettype wire
endmodule : fmpadding

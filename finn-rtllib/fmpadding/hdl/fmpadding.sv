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

	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Parameter Configuration ----------
	input	logic         we,
	input	logic [ 2:0]  wa,
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

	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;

	//-----------------------------------------------------------------------
	// Dynamically configurable state
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
			0:  XOn  <= wd;
			1:  XOff <= wd;
			2:  XEnd <= wd;

			4:  YOn  <= wd;
			5:  YOff <= wd;
			6:  YEnd <= wd;

			default:  assert(0) else begin
				$error("Illegal write address.");
				$stop;
			end
			endcase
		end
	end

	//-----------------------------------------------------------------------
	// Cascaded enables for the nested counters: SCount, XCount, YCount
	uwire  sen;
	uwire  xen;
	uwire  yen;

	//- S-Counter: SIMD fold ------------
	initial begin
		if((NUM_CHANNELS < 1) || (NUM_CHANNELS % SIMD != 0)) begin
			$error("Channel count must be SIMD multiple.");
			$finish;
		end
	end
	// Count SF-2, SF-3, ..., 1, 0, -1
	localparam int unsigned  SF = NUM_CHANNELS/SIMD;
	typedef logic [$clog2(SF-1):0]  scount_t;
	scount_t  SCount = SF-2;

	assign	xen = sen && SCount[$left(SCount)];
	uwire  sclr = rst || xen;
	always_ff @(posedge clk) begin
		if(sclr)      SCount <= SF-2;
		else if(sen)  SCount <= SCount - 1;
	end

	//- X-Counter: image width ----------
	xcount_t  XCount = 0;

	assign	yen = xen && (XCount == XEnd);
	uwire  xclr = rst || yen;
	always_ff @(posedge clk) begin
		if(xclr)      XCount <= 0;
		else if(xen)  XCount <= XCount + 1;
	end
	uwire  xfwd = (XOn <= XCount) && (XCount < XOff);

	//- Y-Counter: image height ---------
	ycount_t  YCount = 0;

	uwire  yclr = rst || (yen && (YCount == YEnd));
	always_ff @(posedge clk) begin
		if(yclr)      YCount <= 0;
		else if(yen)  YCount <= YCount + 1;
	end
	uwire  yfwd = (YOn <= YCount) && (YCount < YOff);

	//-----------------------------------------------------------------------
	// Input forwarding and edge padding
	typedef struct {
		logic  vld;
		logic [STREAM_BITS-1:0]  dat;
	} buf_t;
	buf_t  A = '{ vld: 0, dat: 'x };
	buf_t  B = '{ vld: 0, dat: 'x };

	uwire  fwd = xfwd && yfwd;
	assign	sen = (m_axis_tready || !B.vld) && (s_axis_tvalid || A.vld || !fwd);
	assign	s_axis_tready = !A.vld;
	assign	m_axis_tvalid =  B.vld;
	assign	m_axis_tdata  =  B.dat;

	always_ff @(posedge clk) begin
		if(rst) begin
			B <= '{ vld: 0, dat: 'x };
		end
		else if(m_axis_tready || !B.vld) begin
			B.vld <= s_axis_tvalid || A.vld || !fwd;
			B.dat <= !fwd? '0 : A.vld? A.dat : s_axis_tdata;
		end
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			A <= '{ vld: 0, dat: 'x };
		end
		else begin
			A.vld <= (A.vld || s_axis_tvalid) && ((B.vld && !m_axis_tready) || !fwd);
			if(!A.vld)  A.dat <= s_axis_tdata;
		end
	end

endmodule : fmpadding

/******************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
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
 * @brief	AXI Stream TLAST marker: insert or strip TLAST on a data stream.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/
module lastmarker_axi #(
	bit  DIRECTION_OUT,	// 1: insert TLAST (output marker), 0: strip TLAST (input marker)
	int unsigned  NUM_ITERS = 1,
	int unsigned  DATA_WIDTH,

	localparam int unsigned  AXI_DATA_WIDTH = (DATA_WIDTH+7)/8 * 8
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [AXI_DATA_WIDTH-1:0]  s_axis_tdata,
	input	logic  s_axis_tlast,	// consumed only when !DIRECTION_OUT

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [AXI_DATA_WIDTH-1:0]  m_axis_tdata,
	output	logic  m_axis_tlast
);
`default_nettype none

	//- Data path: straight through -----
	assign	m_axis_tdata  = s_axis_tdata;
	assign	m_axis_tvalid = s_axis_tvalid;
	assign	s_axis_tready = m_axis_tready;

	//=== TLAST generation ==============
	if     (!DIRECTION_OUT)  assign  m_axis_tlast = 'x;
	else if(NUM_ITERS == 1)  assign  m_axis_tlast = 1;
	else begin : genCntLast
		logic signed [$clog2(NUM_ITERS-1):0]  Cnt = NUM_ITERS - 2; // N-2, ..., 0, -1 (last)
		uwire  last = Cnt[$left(Cnt)];
		uwire  beat = s_axis_tvalid & m_axis_tready;
		always_ff @(posedge ap_clk) begin
			if(!ap_rst_n)  Cnt <= NUM_ITERS - 2;
			else if(beat)  Cnt <= Cnt + (last? NUM_ITERS-1 : -1);
		end
		assign	m_axis_tlast = last;
	end : genCntLast

`default_nettype wire
endmodule : lastmarker_axi

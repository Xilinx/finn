/******************************************************************************
 * Copyright (C) 2023, Advanced Micro Devices, Inc.
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
 * @brief	AXI Stream Data Width Converter.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/
module dwc_axi #(
	int unsigned  IBITS,
	int unsigned  OBITS
)(
	//- Global Control ------------------
	input	logic  clk,
	input	logic  rst,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [IBITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OBITS-1:0]  m_axis_tdata
);

	if(IBITS == OBITS) begin : genNoop
		assign	s_axis_tready = m_axis_tready;
		assign	m_axis_tvalid = s_axis_tvalid;
		assign	m_axis_tdata  = s_axis_tdata;
	end : genNoop
	else if(IBITS < OBITS) begin : genUp

		// Sanity Checking: integer upscaling
		initial begin
			if(OBITS % IBITS) begin
				$error("Output width %0d is not a multiple of input width %0d.", OBITS, IBITS);
				$finish;
			end
		end

		// Parallelizing Shift Register A and Sidestep Buffer B on Input Path
		localparam int unsigned  K = OBITS / IBITS;
		typedef logic [IBITS-1:0]  dat_t;
		dat_t       [K-1:0]  ADat = 'x;
		logic [$clog2(K):0]  ACnt = K-1;	// (empty) K-1, ..., 0, -1 (full/valid)
		dat_t  BDat = 'x;
		logic  BRdy =  1;
		always_ff @(posedge clk) begin
			if(rst) begin
				ADat <= 'x;
				ACnt <= K-1;
				BDat <= 'x;
				BRdy <=  1;
			end
			else begin
				automatic type(ACnt)  acnt = (m_axis_tvalid && m_axis_tready)? K-1 : ACnt;
				automatic logic  rdy = !m_axis_tvalid || m_axis_tready;
				if((s_axis_tvalid || !BRdy) && rdy) begin
					ADat <= { BRdy? s_axis_tdata : BDat, ADat[K-1:1] };
					acnt--;
				end
				ACnt <= acnt;

				if(BRdy)  BDat <= s_axis_tdata;
				BRdy <= rdy || (BRdy && !s_axis_tvalid);
			end
		end

		// Output Assignments
		assign  s_axis_tready = BRdy;
		assign	m_axis_tvalid = ACnt[$left(ACnt)];
		assign	m_axis_tdata  = ADat;

	end : genUp
	else begin : genDown

		// Sanity Checking: integer downscaling
		initial begin
			if(IBITS % OBITS) begin
				$error("Input width %0d is not a multiple of output width %0d.", IBITS, OBITS);
				$finish;
			end
		end

		// Serializing Shift Register A and Sidestep Buffer B on Output Path
		localparam int unsigned  K = IBITS / OBITS;
		typedef logic [OBITS-1:0]  dat_t;
		dat_t [      K-1:0]  ADat = 'x;
		logic [$clog2(K):0]  ACnt =  1;	// (full) -K+1, ..., -1, 0, 1 (empty/not valid)
		dat_t  BDat = 'x;
		logic  BRdy =  1;
		dat_t  CDat = 'x;
		logic  CVld =  0;
		always_ff @(posedge clk) begin
			if(rst) begin
				ADat <= 'x;
				ACnt <=  1;
				BDat <= 'x;
				BRdy <=  1;
				CDat <= 'x;
				CVld <=  0;
			end
			else begin
				automatic type(ACnt)  acnt = ACnt;
				automatic logic       ainc = 0;
				if(s_axis_tready) begin
					ADat <= s_axis_tdata;
					acnt = s_axis_tvalid? -K+1 : 1;
				end
				else if(BRdy) begin
					ADat <= { {OBITS{1'bx}}, ADat[K-1:1] };
					ainc = BRdy;
				end;
				ACnt <= acnt + ainc;

				if(BRdy)  BDat <= ADat[0];
				BRdy <= !CVld || m_axis_tready || (BRdy && !ACnt[$left(ACnt)] && ACnt[0]);

				if(!CVld || m_axis_tready)  CDat <= BRdy? ADat[0] : BDat;
				CVld <= (CVld && !m_axis_tready) || !BRdy || ACnt[$left(ACnt)] || !ACnt[0];
			end
		end

		// Output Assignments
		assign  s_axis_tready = BRdy && !ACnt[$left(ACnt)];
		assign	m_axis_tvalid = CVld;
		assign	m_axis_tdata  = CDat;

	end : genDown

endmodule : dwc_axi

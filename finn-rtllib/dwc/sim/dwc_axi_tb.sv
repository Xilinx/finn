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
 * @brief	Testbench for AXI Stream Data Width Converter.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/
module dwc_axi_tb;

	localparam int unsigned  DBITS = 4;
	localparam int unsigned  K     = 3;
	typedef logic [DBITS-1:0]  dat_t;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	if(1) begin : blkUp
		localparam int unsigned  IBITS = DBITS;
		localparam int unsigned  OBITS = K * DBITS;

		//- AXI Stream - Input --------------
		uwire  s_axis_tready;
		logic  s_axis_tvalid;
		dat_t  s_axis_tdata;

		//- AXI Stream - Output -------------
		logic  m_axis_tready;
		uwire  m_axis_tvalid;
		dat_t [K-1:0]  m_axis_tdata;

		dwc_axi #(.IBITS(IBITS), .OBITS(OBITS)) dut (
			.ap_clk(clk), .ap_rst_n(!rst),
			.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
			.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
		);

		// Stimulus: Feed
		dat_t  Q[$];
		initial begin
			s_axis_tvalid = 0;
			s_axis_tdata = 'x;
			@(posedge clk iff !rst);

			repeat(57600) begin
				automatic type(s_axis_tdata)  dat;
				std::randomize(dat);

				while($urandom()%7 < 2) @(posedge clk);

				s_axis_tvalid <= 1;
				s_axis_tdata  <= dat;
				@(posedge clk iff s_axis_tready);
				Q.push_back(dat);

				s_axis_tvalid <= 0;
				s_axis_tdata <= 'x;
			end

			repeat(16) @(posedge clk);
			$finish;
		end

		// Output Sink
		initial begin
			m_axis_tready = 0;
			@(posedge clk iff !rst);

			forever begin
				automatic dat_t [K-1:0]  dat;

				while($urandom()%9 < 1) @(posedge clk);

				m_axis_tready <= 1;
				@(posedge clk iff m_axis_tvalid);
				assert(Q.size >= K) else begin
					$error("Spurious output.");
					$stop;
				end
				for(int unsigned  i = 0; i < K; i++)  dat[i] = Q.pop_front();
				assert(m_axis_tdata == dat) else begin
					$error("Output mismatch.");
					$stop;
				end

				m_axis_tready <= 0;
			end
		end
	end : blkUp

	if(1) begin : blkDown
		localparam int unsigned  IBITS = K * DBITS;
		localparam int unsigned  OBITS = DBITS;

		//- AXI Stream - Input --------------
		uwire  s_axis_tready;
		logic  s_axis_tvalid;
		dat_t [K-1:0]  s_axis_tdata;

		//- AXI Stream - Output -------------
		logic  m_axis_tready;
		uwire  m_axis_tvalid;
		dat_t  m_axis_tdata;

		dwc_axi #(.IBITS(IBITS), .OBITS(OBITS)) dut (
			.ap_clk(clk), .ap_rst_n(!rst),
			.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
			.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
		);

		// Stimulus: Feed
		dat_t  Q[$];
		initial begin
			s_axis_tvalid = 0;
			s_axis_tdata = 'x;
			@(posedge clk iff !rst);

			repeat(57600) begin
				automatic dat_t [K-1:0]  dat;
				std::randomize(dat);

				while($urandom()%7 < 2) @(posedge clk);

				s_axis_tvalid <= 1;
				s_axis_tdata  <= dat;
				@(posedge clk iff s_axis_tready);
				for(int unsigned  i = 0; i < K; i++)  Q.push_back(dat[i]);

				s_axis_tvalid <= 0;
				s_axis_tdata <= 'x;
			end

			repeat(16) @(posedge clk);
			$finish;
		end

		// Output Sink
		initial begin
			m_axis_tready = 0;
			@(posedge clk iff !rst);

			forever begin
				automatic dat_t  dat;

				while($urandom()%9 < 1) @(posedge clk);

				m_axis_tready <= 1;
				@(posedge clk iff m_axis_tvalid);
				assert(Q.size) else begin
					$error("Spurious output.");
					$stop;
				end
				dat = Q.pop_front();
				assert(m_axis_tdata == dat) else begin
					$error("Output mismatch: 0x%0x instead of 0x%0x", m_axis_tdata, dat);
					$stop;
				end

				m_axis_tready <= 0;
			end
		end
	end : blkDown

endmodule : dwc_axi_tb

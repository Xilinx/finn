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
 * @brief	Testbench for MVU core compute kernel.
 *****************************************************************************/

module mvu_accu_tb;

	localparam	IS_MVU = 1;
	localparam	COMPUTE_CORE = "mvu_8sx8u_dsp48";
	localparam	PUMPED_COMPUTE = 0;
	localparam	MW = 6;
	localparam	MH = 32;
	localparam	PE = 1;
	localparam	SIMD = 1;
	localparam	ACTIVATION_WIDTH = 8;
	localparam	WEIGHT_WIDTH = 4;
	localparam	NARROW_WEIGHTS = 1;
	localparam	SIGNED_ACTIVATIONS = 1;
	localparam	SEGMENTLEN = 1;
	localparam	FORCE_BEHAVIORAL = 0;

	// Safely deducible parameters
	localparam  WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8;
	localparam  INPUT_STREAM_WIDTH_BA = ((IS_MVU == 1 ? 1 : PE) * SIMD * ACTIVATION_WIDTH + 7) / 8 * 8;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	logic [WEIGHT_WIDTH-1:0]  WeightMem[MH*MW];
	initial  $readmemh("mvu_accu_tb.dat", WeightMem);

	// Shared Input Feed
	logic [INPUT_STREAM_WIDTH_BA-1:0]  in_TDATA;
	logic  in_TVALID[2];
	uwire  in_TREADY[2];
	initial begin
		in_TDATA = 'x;
		in_TVALID = '{ default: 0 };
		@(posedge clk iff !rst);

		repeat(2161*MW) begin
			automatic logic [ACTIVATION_WIDTH-1:0]  a = $urandom();
			in_TDATA  <= a;
			in_TVALID <= '{ default: 1 };
			fork
				begin
					@(posedge clk iff in_TREADY[0]);
					in_TVALID[0] <= 0;
				end
				begin
					@(posedge clk iff in_TREADY[1]);
					in_TVALID[1] <= 0;
				end
			join
		end

		repeat(MH*MW) @(posedge clk);
		$display("Test completed.");
		$finish;
	end

	// DUTs
	localparam int unsigned  ACCU_WIDTHS[2] = '{ 16, 32 };
	int  OutQ[2][$];
	for(genvar  i = 0; i < $size(ACCU_WIDTHS); i++) begin : genDUTs
		localparam int unsigned  ACCU_WIDTH = ACCU_WIDTHS[i];
		localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (PE*ACCU_WIDTH + 7)/8 * 8;

		// Private Weight Feed
		logic [WEIGHT_STREAM_WIDTH_BA-1:0]  weights_TDATA;
		logic  weights_TVALID;
		uwire  weights_TREADY;
		initial begin
			weights_TDATA  = 'x;
			weights_TVALID = 0;
			@(posedge clk iff !rst);

			weights_TVALID <= 1;
			forever begin
				for(int unsigned  i = 0; i < MH*MW; i++)  begin
					weights_TDATA <= WeightMem[i];
					@(posedge clk iff weights_TREADY);
				end
			end
		end

		// Private Output Capture into Queue
		uwire signed [OUTPUT_STREAM_WIDTH_BA-1:0]  out_TDATA;
		uwire  out_TVALID;
		uwire  out_TREADY = !rst;
		always_ff @(posedge clk iff !rst) begin
			if(out_TVALID)  OutQ[i].push_back(out_TDATA);
		end

		// Actual DUT Instance
		mvu_vvu_axi #(
			.IS_MVU(IS_MVU), .COMPUTE_CORE(COMPUTE_CORE), .PUMPED_COMPUTE(PUMPED_COMPUTE), .MW(MW), .MH(MH), .PE(PE), .SIMD(SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH), .NARROW_WEIGHTS(NARROW_WEIGHTS),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) dut (
			.ap_clk(clk),
			.ap_clk2x(1'b0),
			.ap_rst_n(!rst),
			.s_axis_weights_tdata(weights_TDATA),
			.s_axis_weights_tvalid(weights_TVALID),
			.s_axis_weights_tready(weights_TREADY),
			.s_axis_input_tdata(in_TDATA),
			.s_axis_input_tvalid(in_TVALID[i]),
			.s_axis_input_tready(in_TREADY[i]),
			.m_axis_output_tdata(out_TDATA),
			.m_axis_output_tvalid(out_TVALID),
			.m_axis_output_tready(out_TREADY)
		);
	end : genDUTs

	// Output Equivalence Checker
	always_ff @(posedge clk) begin
		if(OutQ[0].size && OutQ[1].size) begin
			automatic int unsigned  y0 = OutQ[0].pop_front();
			automatic int unsigned  y1 = OutQ[1].pop_front();
			assert(y0 == y1) else begin
				$error("Output Mismatch: %0d vs. %0d", y0, y1);
				$stop;
			end
		end
	end

endmodule : mvu_accu_tb

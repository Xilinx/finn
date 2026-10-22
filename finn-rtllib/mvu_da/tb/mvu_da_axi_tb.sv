/******************************************************************************
 * Copyright (C) 2026, Paderborn University
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
 * @brief	Testbench for the flow-control bracket mvu_da_axi.
 * @details
 *	A fake core (CORE_LATENCY-deep register pipeline adding a constant) stands
 *	in for the adder graph. The testbench drives random tvalid/tready patterns
 *	and checks that every beat arrives exactly once, in order, with the right
 *	value, that the output queue never overflows (assertion inside the DUT)
 *	and that the bracket recovers from a reset in the middle of the traffic.
 *
 *	Queue bound: the lock engages one cycle after the queue becomes non-empty
 *	under back-pressure, so one beat is already queued when the input is
 *	closed, and CORE_LATENCY more beats (accepted up to the lock edge) can
 *	still arrive. The "tight" instances run with MAX_IN_FLIGHT = CORE_LATENCY
 *	+ 1 to confirm; with CORE_LATENCY the overflow assertion fires and beats
 *	are lost (checked manually, not part of the self-check).
 *
 *	Run with:
 *	  xvlog -sv ../mvu_da_axi.sv mvu_da_axi_tb.sv && xelab mvu_da_axi_tb -s sim && xsim sim -R
 *****************************************************************************/

module mvu_da_axi_tb;

	localparam int unsigned  N = 3000;
	localparam int unsigned  WIDTH = 12;	// not a byte multiple: exercises the padding
	localparam int unsigned  CONST = 5;

	logic  clk = 0;
	always #5 clk = !clk;

	tb_instance #(.CORE_LATENCY(1), .WIDTH(WIDTH), .N(N), .CONST(CONST), .SEED(11)) i_l1 (.clk);
	tb_instance #(.CORE_LATENCY(3), .WIDTH(WIDTH), .N(N), .CONST(CONST), .SEED(22)) i_l3 (.clk);
	tb_instance #(.CORE_LATENCY(8), .WIDTH(WIDTH), .N(N), .CONST(CONST), .SEED(33)) i_l8 (.clk);
	tb_instance #(.CORE_LATENCY(3), .WIDTH(WIDTH), .N(N), .CONST(CONST), .SEED(44), .MAX_IN_FLIGHT(4)) i_l3_tight (.clk);
	tb_instance #(.CORE_LATENCY(8), .WIDTH(WIDTH), .N(N), .CONST(CONST), .SEED(55), .MAX_IN_FLIGHT(9)) i_l8_tight (.clk);

	initial begin
		wait(i_l1.done && i_l3.done && i_l8.done && i_l3_tight.done && i_l8_tight.done);
		#20;
		if(i_l1.errors + i_l3.errors + i_l8.errors + i_l3_tight.errors + i_l8_tight.errors == 0)
			$display("PASS: mvu_da_axi_tb");
		else
			$display("FAIL: mvu_da_axi_tb");
		$finish;
	end

endmodule : mvu_da_axi_tb


module tb_instance #(
	int unsigned  CORE_LATENCY,
	int unsigned  WIDTH,
	int unsigned  N,
	int unsigned  CONST,
	int unsigned  SEED,
	int unsigned  MAX_IN_FLIGHT = CORE_LATENCY + 2
)(
	input	logic  clk
);
	localparam int unsigned  WIDTH_BA = (WIDTH + 7)/8*8;

	logic  rst_n = 0;
	logic [WIDTH_BA-1:0]  in_tdata = 'x;
	logic  in_tvalid = 0;
	uwire  in_tready;
	uwire [WIDTH_BA-1:0]  out_tdata;
	uwire  out_tvalid;
	logic  out_tready = 0;

	uwire [WIDTH-1:0]  core_inp;
	logic [WIDTH-1:0]  core_out;
	logic  done = 0;

	mvu_da_axi #(
		.INPUT_STREAM_WIDTH(WIDTH), .OUTPUT_STREAM_WIDTH(WIDTH),
		.CORE_LATENCY(CORE_LATENCY), .SIGNED_OUTPUT(0), .MAX_IN_FLIGHT(MAX_IN_FLIGHT)
	) dut (
		.ap_clk(clk), .ap_rst_n(rst_n),
		.s_axis_input_tdata(in_tdata), .s_axis_input_tvalid(in_tvalid), .s_axis_input_tready(in_tready),
		.m_axis_output_tdata(out_tdata), .m_axis_output_tvalid(out_tvalid), .m_axis_output_tready(out_tready),
		.core_inp(core_inp), .core_out(core_out)
	);

	// Fake core: CORE_LATENCY registers, no reset, no enable, adds CONST.
	logic [WIDTH-1:0]  Pipe[CORE_LATENCY];
	always_ff @(posedge clk) begin
		Pipe[0] <= core_inp + CONST;
		for(int i = 1; i < CORE_LATENCY; i++)  Pipe[i] <= Pipe[i-1];
	end
	assign	core_out = Pipe[CORE_LATENCY-1];

	// Monitors sample at the clock edge (values before any update at this edge)
	int  seed = SEED;
	int  sent = 0, received = 0, errors = 0;
	int  density_in = 0, density_out = 0;
	bit  accepted = 0;
	bit  running = 0;

	function automatic logic [WIDTH_BA-1:0] expected(input int n);
		return WIDTH_BA'((n + CONST) & ((1 << WIDTH) - 1));
	endfunction

	always @(posedge clk) begin
		accepted = rst_n && in_tvalid && in_tready;
		if(accepted)  sent++;
		if(rst_n && out_tvalid && out_tready) begin
			received++;
			if(!running) begin
				errors++;
				if(errors < 5)  $display("%m: spurious output beat (received=%0d)", received);
			end
			else if(out_tdata !== expected(received)) begin
				errors++;
				if(errors < 5)  $display("%m: beat %0d: got %0d, expected %0d", received, out_tdata, expected(received));
			end
		end
	end

	// Drivers act at the falling edge: hold a pending beat until it is accepted
	always @(negedge clk) begin
		if(!rst_n) begin
			in_tvalid <= 0; in_tdata <= 'x; out_tready <= 0;
		end
		else begin
			if(!in_tvalid || accepted) begin
				if(running && sent < N && (($unsigned($random(seed)) % 100) < density_in)) begin
					in_tvalid <= 1;
					in_tdata  <= WIDTH_BA'((sent + 1) & ((1 << WIDTH) - 1));
				end
				else begin
					in_tvalid <= 0;
					in_tdata  <= 'x;
				end
			end
			out_tready <= (($unsigned($random(seed)) % 100) < density_out);
		end
	end

	task automatic run_round(input int din, input int dout);
		sent = 0; received = 0;
		density_in = din; density_out = dout;
		running = 1;
		wait(received == N);
		// drain: nothing else may come out
		density_in = 0; density_out = 100;
		repeat(4*CORE_LATENCY + 16) @(posedge clk);
		running = 0;
		density_out = 0;
		@(posedge clk);
	endtask

	initial begin
		repeat(3) @(posedge clk);
		#1 rst_n = 1;
		repeat(2) @(posedge clk);

		// dense input, choking output: exercises the lock and the queue bound
		run_round(100, 30);
		// balanced
		run_round(60, 60);
		// sparse input, always-ready output
		run_round(30, 100);
		// bursty output ready
		run_round(90, 10);

		// reset in the middle of traffic, then a fresh round
		density_in = 100; density_out = 0; running = 1;
		repeat(3*CORE_LATENCY + 6) @(posedge clk);
		running = 0;
		#1 rst_n = 0;
		repeat(3) @(posedge clk);
		#1 rst_n = 1;
		repeat(2) @(posedge clk);
		run_round(80, 50);

		if(errors == 0)  $display("%m: OK (CORE_LATENCY=%0d, MAX_IN_FLIGHT=%0d)", CORE_LATENCY, MAX_IN_FLIGHT);
		else             $display("%m: %0d errors (CORE_LATENCY=%0d, MAX_IN_FLIGHT=%0d)", errors, CORE_LATENCY, MAX_IN_FLIGHT);
		done = 1;
	end

endmodule : tb_instance

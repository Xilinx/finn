/**
 * Copyright (c) 2023-2024, Xilinx
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of FINN nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 */

module memstream #(
	int unsigned  DEPTH,
	int unsigned  WIDTH,

	parameter  INIT_FILE = "",
	parameter  RAM_STYLE = "auto",
	bit  EXTRA_OUTPUT_REG = 0
)(
	input	logic  clk,
	input	logic  rst,

	// Configuration and readback interface - compatible with ap_memory
	input	logic  config_ce,
	input	logic  config_we,
	input	logic [31     :0]  config_address,
	input	logic [WIDTH-1:0]  config_d0,

	output	logic  config_rack,
	output	logic [WIDTH-1:0]  config_q0,

	// Continuous output stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [WIDTH-1:0]  odat
);

	typedef logic [$clog2(DEPTH)-1:0]  addr_t;
	typedef logic [WIDTH        -1:0]  data_t;

	uwire  en;       // Pipeline enable
	uwire  rollback; // Rollback stream reads if backpressure would block read back

	// Address counter with pre-computed last indication for val == DEPTH-1
	typedef struct {
		addr_t  val;
		logic   lst;
	} ptr_t;
	// 1-Hot pipeline command
	typedef struct {
		logic  wr;  // Write (config)
		logic  rb;  // Read back (config)
		logic  rs;  // Read stream
	} cmd_t;
	localparam cmd_t  NOP = cmd_t'{ default: 0 };
	localparam cmd_t  WR  = cmd_t'{ wr: 1, default: 0 };
	localparam cmd_t  RB  = cmd_t'{ rb: 1, default: 0 };
	localparam cmd_t  RS  = cmd_t'{ rs: 1, default: 0 };

	// Pipeline Stages:
	//	#0 - Streaming pointer run ahead
	//	#1 - Input Feed
	//	#2 - Memory Access
	//	#3 - Absorped Output Register
	//	#4 - Extra Fabric Register [EXTRA_OUTPUT_REG]

	// Counter history to facilitate pipeline rollback
	localparam int unsigned  PIPE_DEPTH = 3 + EXTRA_OUTPUT_REG;
	ptr_t  Ptr[0:PIPE_DEPTH] = '{
		0: '{ val: 0, lst: DEPTH<2 },
		default: '{ default: 'x }
	};
	cmd_t   Cmd [1:PIPE_DEPTH] = '{ default: NOP };
	data_t  Data[1:PIPE_DEPTH] = '{ default: 'x };

	//-----------------------------------------------------------------------
	// Stage #0&1: Address & Op
	if(1) begin : blkStage1
		// Increment for wrapping DEPTH-1 back to zero
		localparam int unsigned  WRAP_INC = 2**$bits(addr_t) - DEPTH + 1;

		uwire ptr_t  ptr_eff = Ptr[rollback? PIPE_DEPTH : 0];
		uwire ptr_t  ptr_nxt;
		assign	ptr_nxt.val = ptr_eff.val + (config_ce? 0 : !ptr_eff.lst? 1 : WRAP_INC);
		assign	ptr_nxt.lst =
			DEPTH < 2?   1 :
			config_ce?   ptr_eff.lst :
			ptr_eff.lst? 0 :
			/* else */   ptr_eff.val == DEPTH-2;

		always_ff @(posedge clk) begin
			if(rst)      Ptr[0] <= '{ val: 0, lst: DEPTH<2 };
			else if(en)  Ptr[0] <= ptr_nxt;
		end

		// Issue next Memory Operation
		always_ff @(posedge clk) begin
			if(rst) begin
				Cmd[1] <= NOP;
				Ptr[1] <= '{ default : 'x };
				Data[1] <= 'x;
			end
			else if(en) begin
				Cmd[1] <= NOP;
				if(config_ce) begin
					Cmd[1] <= config_we? WR : RB;
					Ptr[1] <= '{ val: config_address, lst: 'x };
					Data[1] <= config_d0;
				end
				else begin
					Cmd[1] <= RS;
					Ptr[1] <= ptr_eff;
					Data[1] <= 'x;
				end
			end
		end
	end : blkStage1

	//-----------------------------------------------------------------------
	// Command Memory Bypass Pipeline
	for(genvar  i = 2; i <= PIPE_DEPTH; i++) begin : genCmdPipe
		always_ff @(posedge clk) begin
			if(rst) begin
				Ptr[i] <= '{ default: 'x };
				Cmd[i] <= NOP;
			end
			else begin
				// always reset `rack` after a single clock cycle
				if(i == PIPE_DEPTH)  Cmd[i].rb <= 0;

				if(en) begin
					// copy from previous stage ...
					Ptr[i] <= Ptr[i-1];
					Cmd[i] <= Cmd[i-1];
					// ... but clear streaming pipe upon `rollback`
					if(rollback)  Cmd[i].rs <= 0;
				end
			end
		end
	end : genCmdPipe

	//-----------------------------------------------------------------------
	// Data Readout Pipeline

	// Stage #2: Memory Access
	if(1) begin : blkData2
		(* RAM_STYLE = RAM_STYLE *)
		data_t  Mem[DEPTH];

		// Optional Memory Initialization
		if(INIT_FILE != "")  initial $readmemh(INIT_FILE, Mem);

		// Execute Memory Operation
		uwire  we = Cmd[1].wr;
		uwire addr_t  wa = Ptr[1].val;
		uwire data_t  wd = Data[1];
		data_t  RdOut;
		always_ff @(posedge clk) begin
			if(en) begin
				// NO_CHANGE mode as READ and WRITE never happen together.
				if(we)  Mem[wa] <= wd;
				else  RdOut <= Mem[wa];
			end
		end
		always_comb  Data[2] = RdOut;
	end : blkData2

	// Further Stages as configured
	for(genvar  i = 3; i <= PIPE_DEPTH; i++) begin : genDataPipe
		always_ff @(posedge clk) begin
			if(en)  Data[i] <= Data[i-1];  // just copy
		end
	end : genDataPipe

	//-----------------------------------------------------------------------
	// Output Interfaces & Flow Control
	if(1) begin : blkOutput

		uwire cmd_t  cmd = Cmd[PIPE_DEPTH];
		uwire data_t  data = Data[PIPE_DEPTH];

		// Wire up Outputs
		assign	config_rack = cmd.rb;
		assign	config_q0 = data;

		assign	ovld = cmd.rs;
		assign	odat = data;

		// Flow Control & Pipeline Enablement
		// - Streaming output is allowed to assert backpressure.
		// - New config requests or pending readouts push forward nonetheless,
		//   at the cost of a pipeline rollback for the stream readout.
		uwire  backpressure = cmd.rs && !ordy;
		uwire [PIPE_DEPTH-1:0]  push0;
		assign	push0[0] = config_ce;
		for(genvar  i = 1; i < PIPE_DEPTH; i++)  assign  push0[i] = Cmd[i].rb;
		uwire  push = |push0;
		assign	rollback = backpressure && push;
		assign	en       = !backpressure || push;

	end : blkOutput

endmodule : memstream

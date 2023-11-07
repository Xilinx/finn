/**
 * Copyright (c) 2023, Xilinx
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
	parameter  RAM_STYLE = "auto"
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

	// Counter with pre-computed last indication for val == DEPTH-1
	typedef struct {
		addr_t  val;
		logic   lst;
	} ptr_t;

	// Counter history to facilitate pipeline rollback
	ptr_t  Ptr[3] = '{
		0: '{ val: 0, lst: DEPTH<2 },
		default: '{ default: 'x }
	};

	//-----------------------------------------------------------------------
	// Stage #0: Address & Op
	logic  Wr1 = 0;  // Write
	logic  Rb1 = 0;  // Read back
	logic  Rs1 = 0;  // Read stream
	data_t  Data1 = 'x;
	if(1) begin : blkStage1
		// Increment for wrapping DEPTH-1 back to zero
		localparam int unsigned  WRAP_INC = 2**$bits(addr_t) - DEPTH + 1;

		uwire ptr_t  ptr_eff = rollback? Ptr[2] : Ptr[0];
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
				Wr1 <= 0;
				Rb1 <= 0;
				Rs1 <= 0;
				Ptr[1] <= '{ default : 'x };
				Data1  <= 'x;
			end
			else if(en) begin
				Wr1 <= 0;
				Rb1 <= 0;
				Rs1 <= 0;
				if(config_ce) begin
					if(config_we)  Wr1 <= 1;
					else           Rb1 <= 1;
					Ptr[1] <= '{ val: config_address, lst: 'x };
					Data1  <= config_d0;
				end
				else begin
					Rs1 <= 1;
					Ptr[1] <= ptr_eff;
					Data1  <= 'x;
				end
			end
		end
	end : blkStage1

	//-----------------------------------------------------------------------
	// Stage #2: Memory Access
	logic   Rb2 = 0;
	logic   Rs2 = 0;
	data_t  Data2 = 'x;
	if(1) begin : blkStage2
		(* RAM_STYLE = RAM_STYLE *)
		data_t  Mem[DEPTH];

		// Optional Memory Initialization
		if(INIT_FILE != "")  initial $readmemh(INIT_FILE, Mem);

		// Execute Memory Operation
		uwire addr_t  addr = Ptr[1].val;
		always_ff @(posedge clk) begin
			if(en) begin
				if(Wr1)  Mem[addr] <= Data1;
				Data2 <= Mem[addr];
			end
		end

		// Copy Output Designation
		always_ff @(posedge clk) begin
			if(rst) begin
				Rb2 <= 0;
				Rs2 <= 0;
				Ptr[2] <= '{ default: 'x };
			end
			else if(en) begin
				Rb2 <= Rb1;
				Rs2 <= Rs1 && !rollback;
				Ptr[2] <= Ptr[1];
			end
		end
	end : blkStage2

	//-----------------------------------------------------------------------
	// Output Interfaces
	assign	config_rack = Rb2;
	assign	config_q0 = Data2;

	assign	ovld = Rs2;
	assign	odat = Data2;

	uwire  backpressure = Rs2 && !ordy;
	assign	rollback = backpressure && (Rb1 || config_ce);
	assign	en       = !backpressure || Rb1 || config_ce;

endmodule : memstream

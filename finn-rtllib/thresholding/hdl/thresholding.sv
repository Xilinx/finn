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
 * @brief	Pipelined thresholding by binary search.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *  Produces the N-bit count of those among 2^N-1 thresholds that are not
 *  larger than the corresponding input:
 *     y = Σ(T_i <= x)
 *  The result is computed by binary search. The runtime-configurable
 *  thresholds must be written in ascending order:
 *     i < j => T_i < T_j
 *  The design supports channel folding allowing each input to be processed
 *  with respect to a selectable set of thresholds. The corresponding
 *  threshold configuration relies on a channel address prefix. Inputs are
 *  accompanied by a channel selector.
 *
 *  Parameter Layout as seen on AXI-Lite (row by row):
 *            | Base                \    Offs  |   0    1    2  ...   2^N-2   2^N-1
 *   ---------+--------------------------------+------------------------------------
 *    Chnl #0 |   0                            |  T_0  T_1  T_2 ... T_{2^N-2}  'x
 *    Chnl #1 |   2^N                          |  T_0  T_1  T_2 ... T_{2^N-2}  'x
 *    Chnl #c | ((c/PE)*$clog2(PE) + c%PE)*2^N |  T_0  T_1  T_2 ... T_{2^N-2}  'x
 *
 *****************************************************************************/
module thresholding #(
	int unsigned  N,  // output precision
	int unsigned  K,  // input/threshold precision
	int unsigned  C,  // number of channels
	int unsigned  PE, // parallel processing elements

	bit  SIGNED = 1,  // signed inputs
	bit  FPARG  = 0,  // floating-point inputs: [sign] | exponent | mantissa
	int  BIAS   = 0,  // offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",
	bit  USE_CONFIG = 1,

	// Force Use of On-Chip Memory Blocks
	int unsigned  DEPTH_TRIGGER_URAM = 0,	// if non-zero, local mems of this depth or more go into URAM (prio)
	int unsigned  DEPTH_TRIGGER_BRAM = 0,	// if non-zero, local mems of this depth or more go into BRAM
	bit  DEEP_PIPELINE = 0,

	localparam int unsigned  CF = C/PE,  // Channel fold
	localparam int unsigned  O_BITS = BIAS >= 0?
		/* unsigned */ $clog2(2**N+BIAS) :
		/* signed */ 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Threshold Configuration
	input	logic  cfg_en,
	input	logic  cfg_we,
	input	logic [$clog2(CF)+$clog2(PE)+N-1:0]  cfg_a,
	input	logic [K-1:0]  cfg_d,
	output	logic  cfg_rack,
	output	logic [K-1:0]  cfg_q,

	// Input Stream
	output	logic  irdy,
	input	logic  ivld,
	input	logic [PE-1:0][K-1:0]  idat,

	// Output Stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [PE-1:0][O_BITS-1:0]  odat
);

	// Parameter Constraints Checking
	initial begin
		if(CF*PE != C) begin
			$error("Parallelism PE=%0d is not a multiple of channel count C=%0d.", PE, C);
			$finish;
		end
	end

	// Operations within Pipeline
	typedef enum logic [1:0] {
		NOP = 2'b00, // No operation
		TH  = 2'b01, // Thresholding
		WR  = 2'b11, // Write (initialization)
		RB  = 2'b10, // Readback (validation)
		CFG = 2'b1x  // Config op (pointer-preserving)
	} op_e;

	// Pipeline Link Type
	typedef logic [$clog2(CF)+N-1:0]  ptr_t;
	typedef logic [K           -1:0]  val_t;
	typedef struct packed {
		op_e   op;
		ptr_t  ptr;	// WR/RB: address;         TH: result
		val_t  val;	// WR/RB: threshold value; TH: input value
	} pipe_t;

	//-----------------------------------------------------------------------
	// Pipeline Feed
	//	- configuration always takes precedence
	//	- number of pending thresholding ops capped to N+3
	//	  across pipeline and output FIFO: pipe:N + A:1 + B:1 + 1
	localparam int unsigned  MAX_PENDING = (DEEP_PIPELINE+1)*N + 3;
	pipe_t  pipe[PE][N+1];
	if(1) begin : blkFeed

		// Thresholding Input Guard ensuring Output FIFO is never overrun
		logic signed [$clog2(MAX_PENDING):0]  GuardSem = MAX_PENDING-1;	// MAX_PENDING-1, ..., 0, -1
		uwire  th_full = GuardSem[$left(GuardSem)];
		always_ff @(posedge clk) begin
			if(rst)  GuardSem <= MAX_PENDING-1;
			else begin
				automatic logic  dec = !(USE_CONFIG && cfg_en) && !th_full && ivld;
				automatic logic  inc = ovld && ordy;
				GuardSem <= GuardSem + (inc == dec? 0 : inc? 1 : -1);
			end
		end

		// PE Configuration Address Decoding
		logic  cfg_sel[PE];
		logic  cfg_oob;
		logic [N-1:0]  cfg_ofs;
		if(PE == 1) begin
			assign	cfg_sel[0] = 1;
			assign	cfg_oob = 0;
			assign	cfg_ofs = cfg_a[0+:N];
		end
		else begin
			uwire [$clog2(PE)-1:0]  cfg_pe = cfg_a[N+:$clog2(PE)];
			always_comb begin
				foreach(cfg_sel[pe]) begin
					cfg_sel[pe] = USE_CONFIG && cfg_en && (cfg_pe == pe);
				end
				cfg_oob = (cfg_pe >= PE);
				cfg_ofs = cfg_a[0+:N];
				if(cfg_oob && !cfg_we) begin
					// Map readbacks from padded rows (non-existent PEs) to padded highest threshold index of first PE
					cfg_sel[0] = 1;
					cfg_ofs = '1;
				end
			end
		end

		uwire ptr_t  iptr;
		assign	iptr[0+:N] = cfg_ofs;
		if(CF > 1) begin
			// Channel Fold Rotation
			logic [$clog2(CF)-1:0]  CnlCnt = 0;
			logic                   CnlLst = 0;
			always_ff @(posedge clk) begin
				if(rst) begin
					CnlCnt <= 0;
					CnlLst <= 0;
				end
				else if(!(USE_CONFIG && cfg_en) && !th_full && ivld) begin
					CnlCnt <= CnlCnt + (CnlLst? 1-CF : 1);
					CnlLst <= CnlCnt == CF-2;
				end
			end

			assign  iptr[N+:$clog2(CF)] = USE_CONFIG && cfg_en? cfg_a[N+$clog2(PE)+:$clog2(CF)] : CnlCnt;
		end

		for(genvar  pe = 0; pe < PE; pe++) begin
			assign	pipe[pe][0] = '{
				op:  USE_CONFIG && cfg_en?
					(!cfg_sel[pe]? NOP : cfg_we? WR : RB) :
					(ivld && !th_full? TH : NOP),
				ptr: iptr,
				val: !(USE_CONFIG && cfg_en)? idat[pe] : cfg_we? cfg_d : 0
			};
		end

		assign	irdy = !(USE_CONFIG && cfg_en) && !th_full;
	end : blkFeed

	//-----------------------------------------------------------------------
	// Free-Running Thresholding Pipeline
	for(genvar  stage = 0; stage < N; stage++) begin : genStages

		localparam int unsigned  SN = N-1-stage;
		for(genvar  pe = 0; pe < PE; pe++) begin : genPE
			uwire pipe_t  p = pipe[pe][stage];
			uwire  cs = (p.ptr[SN:0] == 2**SN-1);

			// Threshold Memory
			val_t  Thresh;	// Read-out register
			if(1) begin : blkThresh
				localparam int unsigned  DEPTH = CF * 2**stage;
				localparam  RAM_STYLE =
					DEPTH_TRIGGER_URAM && (DEPTH >= DEPTH_TRIGGER_URAM)? "ultra" :
					DEPTH_TRIGGER_BRAM && (DEPTH >= DEPTH_TRIGGER_BRAM)? "block" :
					// If BRAM trigger defined, force distributed memory below if Vivado may be tempted to use BRAM nonetheless.
					DEPTH_TRIGGER_BRAM && (DEPTH >= 64)? "distributed" : "auto";

				(* RAM_STYLE = RAM_STYLE *)
				val_t  Threshs[DEPTH];
				if(THRESHOLDS_PATH != "") begin
					initial  $readmemh($sformatf("%sthreshs_%0d_%0d.dat", THRESHOLDS_PATH, pe, stage), Threshs);
				end

				if(USE_CONFIG) begin : genThreshMem
					uwire  we = (p.op ==? WR) && cs;
					if((CF == 1) && (stage == 0)) begin
						always @(posedge clk) begin
							if(we)  Threshs[0] <= p.val;
						end
					end
					else begin
						uwire [$clog2(CF)+stage-1:0]  addr = p.ptr[$clog2(CF)+N-1:SN+1];
						always @(posedge clk) begin
							if(we)  Threshs[addr] <= p.val;
						end
					end
				end : genThreshMem

				if((CF == 1) && (stage == 0)) begin
					assign	Thresh = Threshs[0];
				end
				else begin
					uwire [$clog2(CF)+stage-1:0]  addr = p.ptr[$clog2(CF)+N-1:SN+1];
					always_ff @(posedge clk) begin
						Thresh <= Threshs[addr];
					end
				end

			end : blkThresh

			// Pipeline State
			pipe_t  P = '{ op: NOP, default: 'x };
			logic   Reval = 0;
			always_ff @(posedge clk) begin
				if(rst) begin
					P <= '{ op: NOP, default: 'x };
					Reval <= 0;
				end
				else begin
					P <= p;
					Reval <= (p.op ==? RB) && cs;
				end
			end

			logic  cmp;
			if(!SIGNED)		assign	cmp = $unsigned(Thresh) <= $unsigned(P.val);
			else if(!FPARG)	assign	cmp =   $signed(Thresh) <=   $signed(P.val);
			else begin : blkSignedFloat
				uwire  mag_eq = Thresh[K-2:0] == P.val[K-2:0];
				uwire  mag_le = Thresh[K-2:0] <= P.val[K-2:0];
				always_comb begin
					unique case({Thresh[K-1], P.val[K-1]})
					2'b00:  cmp = mag_le;
					2'b01:  cmp = 0;
					2'b10:  cmp = 1;
					2'b11:  cmp = !mag_le || mag_eq;
					default: cmp = 'x;
					endcase
				end
			end : blkSignedFloat

			// Pipeline State Update
			pipe_t  pp;
			always_comb begin
				pp = P;
				if(P.op !=? CFG)  pp.ptr[SN] = cmp;
				if(Reval)         pp.val = Thresh;
			end

			// Pipeline State Forward (potentially additional register)
			pipe_t  pf;
			if(!DEEP_PIPELINE)  assign  pf = pp;
			else begin
				pipe_t  Pf = '{ op: NOP, default: 'x };
				always_ff @(posedge clk) begin
					if(rst)  Pf <= '{ op: NOP, default: 'x };
					else     Pf <= pp;
				end
				assign	pf = Pf;
			end

			assign	pipe[pe][stage+1] = pf;

		end : genPE
	end : genStages

	//-----------------------------------------------------------------------
	// Configuration Readback
	always_comb begin
		cfg_rack = 0;
		cfg_q = 0;
		foreach(pipe[pe]) begin
			automatic pipe_t  p = pipe[pe][N];
			cfg_rack |= p.op ==? RB;
			cfg_q    |= p.val;
		end
	end

	//-----------------------------------------------------------------------
	// Stream Output through FIFO
	//	- Depth of N + Output Reg to allow pipe to drain entirely under backpressure
	//	- Typically mapped to an SRL shift register
	if(1) begin : blkStreamOutput
		localparam int unsigned  A_DEPTH = MAX_PENDING - 1;
		logic        [PE-1 : 0][N-1 : 0]  ADat[A_DEPTH];
		logic signed [$clog2(A_DEPTH):0]  APtr = '1;	// -1, 0, 1, ..., A_DEPTH-1
		uwire  avld = !APtr[$left(APtr)];

		logic [PE-1:0][N-1:0]  BDat = 'x;
		logic  BVld =  0;

		uwire  aload = pipe[0][N].op ==? TH;
		uwire  bload = !BVld || ordy;

		always_ff @(posedge clk) begin
			if(aload) begin
				assert(APtr < $signed(A_DEPTH-1)) else begin
					$error("Overrun after failing stream guard.");
					$stop;
				end
				foreach(pipe[pe])  ADat[0][pe] <= pipe[pe][N].ptr;
				for(int unsigned  i = 1; i < A_DEPTH; i++)  ADat[i] <= ADat[i-1];
			end
		end
		always_ff @(posedge clk) begin
			if(rst)  APtr <= '1;
			else     APtr <= APtr + (aload == (avld && bload)? 0 : aload? 1 : -1);
		end
		always_ff @(posedge clk) begin
			if(rst) begin
				BDat <= 'x;
				BVld <=  0;
			end
			else if(bload) begin
				BDat <= ADat[APtr];
				BVld <= avld;
			end
		end

		assign	ovld = BVld;
		for(genvar  pe = 0; pe < PE; pe++) begin
			assign	odat[pe] = BDat[pe] + BIAS;
		end
	end : blkStreamOutput

endmodule : thresholding

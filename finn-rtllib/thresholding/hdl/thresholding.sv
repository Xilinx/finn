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
 * @brief	Pipelined thresholding by binary search.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
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
 *****************************************************************************/
module thresholding #(
	int unsigned  N,  // output precision
	int unsigned  K,  // input/threshold precision
	int unsigned  C,  // number of channels
	int unsigned  PE, // parallel processing elements

	bit  SIGNED = 1,  // signed inputs
	bit  FPARG  = 0,  // floating-point inputs: [sign] | exponent | mantissa
	int  BIAS   = 0,  // offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

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
	pipe_t  pipe[PE][N+1];
	if(1) begin : blkFeed

		// Thresholding Input Guard ensuring Output FIFO is never overrun
		logic signed [$clog2(N+3):0]  GuardSem = N+2;	// N+2, N+1, ..., 0, -1
		uwire  th_full = GuardSem[$left(GuardSem)];
		always_ff @(posedge clk) begin
			if(rst)  GuardSem <= N+2;
			else begin
				automatic logic  dec = !cfg_en && !th_full && ivld;
				automatic logic  inc = ovld && ordy;
				GuardSem <= GuardSem + (inc == dec? 0 : inc? 1 : -1);
			end
		end

		// PE Configuration Address Decoding
		uwire  cfg_sel[PE];
		if(PE == 1)  assign  cfg_sel[0] = 1;
		else begin
			for(genvar  pe = 0; pe < PE; pe++) begin
				assign	cfg_sel[pe] = cfg_en && (cfg_a[N+:$clog2(PE)] == pe);
			end
		end


		uwire ptr_t  iptr;
		assign	iptr[0+:N] = cfg_a[0+:N];
		if(CF > 1) begin
			// Channel Fold Rotation
			logic [$clog2(CF)-1:0]  CnlCnt = 0;
			logic                   CnlLst = 0;
			always_ff @(posedge clk) begin
				if(rst) begin
					CnlCnt <= 0;
					CnlLst <= 0;
				end
				else if(!cfg_en && !th_full && ivld) begin
					CnlCnt <= CnlCnt + (CnlLst? 1-CF : 1);
					CnlLst <= CnlCnt == CF-2;
				end
			end

			assign  iptr[N+:$clog2(CF)] = cfg_en? cfg_a[N+$clog2(PE)+:$clog2(CF)] : CnlCnt;
		end

		for(genvar  pe = 0; pe < PE; pe++) begin
			assign	pipe[pe][0] = '{
				op:  cfg_en?
					(!cfg_sel[pe]? NOP : cfg_we? WR : RB) :
					(ivld && !th_full? TH : NOP),
				ptr: iptr,
				val: !cfg_en? idat[pe] : cfg_we? cfg_d : 0
			};
		end

		assign	irdy = !cfg_en && !th_full;
	end : blkFeed

	//-----------------------------------------------------------------------
	// Free-Running Thresholding Pipeline
	for(genvar  stage = 0; stage < N; stage++) begin : genStages

		localparam int unsigned  SN = N-1-stage;
		for(genvar  pe = 0; pe < PE; pe++) begin : genPE
			uwire pipe_t  p = pipe[pe][stage];
			uwire  cs = (p.ptr[SN:0] == 2**SN-1);

			// Threshold Memory
			logic [K-1:0]  Thresh = 'x;	// Read-out register
			if(1) begin : blkThreshMem
				uwire  we = (p.op ==? WR) && cs;
				if((CF == 1) && (stage == 0)) begin
					always_ff @(posedge clk) begin
						if(we)  Thresh <= p.val;
					end
				end
				else begin
					logic [K-1:0]  Threshs[CF * 2**stage];
					uwire [$clog2(CF)+stage-1:0]  addr = p.ptr[$clog2(CF)+N-1:SN+1];
					always_ff @(posedge clk) begin
						if(we)  Threshs[addr] <= p.val;
						Thresh <= Threshs[addr];
					end
				end
			end : blkThreshMem

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
			always_comb begin
				automatic pipe_t  pp = P;
				if(P.op !=? CFG)  pp.ptr[SN] = cmp;
				if(Reval)         pp.val = Thresh;
				pipe[pe][stage+1] = pp;
			end

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
		localparam int unsigned  A_DEPTH = N+2;
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

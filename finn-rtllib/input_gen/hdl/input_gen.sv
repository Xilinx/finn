/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief
 *	Generic sliding window / input generator driven by a perfect loop nest.
 *
 *	A loop nest:
 *
 *	  for(i0 = 0; i0 < DIMS[0]; i0++)
 *	    for(i1 = 0; i1 < DIMS[1]; i1++)
 *	      ...
 *	        for(in = 0; in < DIMS[D-1]; in++)
 *	          emit(buf[COEFS[0]*i0 + COEFS[1]*i1 + ... + COEFS[D-1]*in])
 *
 *	is encoded by the array parameters DIMS and COEFS.  The module reads
 *	a linear input stream into a circular buffer and replays elements
 *	according to the loop nest addressing, supporting arbitrary strides,
 *	dilations, and transpositions.
 *
 *	FM_SIZE is the number of input elements per feature map (period of the
 *	input stream).  The olst output exposes the level-completion cascade
 *	term[D-1:0] and odone the per-level exhaustion flags done[D-1:0],
 *	both synchronous with each output beat.
 *
 *	Shared by the MVU/VVU (finn-rtllib/mvu) and Where (finn-rtllib/where)
 *	cores; the buffer sizing is mirrored in Python by
 *	finn.custom_op.fpgadataflow.hwwhere.input_gen_buffer_depth().
 ***************************************************************************/

module input_gen #(
	int unsigned  DATA_WIDTH,
	int unsigned  FM_SIZE,
	int unsigned  D,
	int unsigned  DIMS[D],
	int unsigned  COEFS[D],
	parameter  RAM_STYLE = "auto",	// buffer implementation hint (Vivado RAM_STYLE)
	// Buffers deeper than this are read through a latency pipeline with a
	// read-ahead queue (see Output Stage). Matches the HLS input_gen<>;
	// override (e.g. 0) only to exercise the read-ahead path in tests.
	int unsigned  READ_AHEAD_DEPTH = 262144
)(
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	// Output Stream
	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	output	logic [D-1:0]  olst,
	output	logic [D-1:0]  odone,
	input	logic  ordy
);

	//=== Parameter Validation ==============================================
	initial begin
		if(D == 0) begin
			$error("%m: D must be at least 1.");
			$finish;
		end
		for(int unsigned  i = 0; i < D; i++) begin
			if(DIMS[i] == 0) begin
				$error("%m: DIMS[%0d] must be positive.", i);
				$finish;
			end
		end
	end

	//=== Elaboration-Time Nest Computations ================================
	// Parent coefficient per level (W in the HLS Nest<> encoding):
	// W[0] = FM_SIZE, W[i>0] = COEFS[i-1].
	typedef int unsigned  w_arr_t[D+1];
	function automatic w_arr_t  INIT_W();
		automatic w_arr_t  a;
		a[0] = FM_SIZE;
		for(int unsigned  i = 0; i < D; i++)  a[i+1] = COEFS[i];
		return  a;
	endfunction : INIT_W
	localparam w_arr_t  W = INIT_W();

	// Free-pointer rewind of level i assuming it were responsible for the
	// free pointer (R=1), built inside out. Mirrors Nest<true, ...>::fp_rewind
	// of the HLS input_gen<>.
	typedef int unsigned  fw_arr_t[D+1];
	function automatic fw_arr_t  INIT_FW_IF_R();
		automatic fw_arr_t  a;
		a[D] = 0;
		for(int  i = D-1; i >= 0; i--) begin
			automatic int unsigned  fw = (DIMS[i]-1) * COEFS[i] + a[i+1];
			a[i] = (COEFS[i] > 0) && (fw <= W[i])? fw : 0;
		end
		return  a;
	endfunction : INIT_FW_IF_R
	localparam fw_arr_t  FW_IF_R = INIT_FW_IF_R();

	// Free-pointer responsibility flag per level.
	// R_FLAG[i] is the R flag passed into level i from its parent.
	// Level i may advance the free pointer in lockstep with its iterations if
	// doing so never carries it beyond its period W[i], i.e. its furthest
	// advance (DIMS[i]-1)*COEFS[i] plus the inner levels' rewind fits. The
	// former criterion COEFS[i]*DIMS[i] <= W[i] was stricter than necessary:
	// it failed whenever the last stride period extends past the end of W[i]
	// although no read does (kernel < stride, floor remainder), leaving the
	// free pointer stuck and the buffer to overflow.
	typedef bit  r_flag_arr_t[D+1];
	function automatic r_flag_arr_t  INIT_R_FLAG();
		automatic r_flag_arr_t  a;
		a[0] = 1;
		for(int unsigned  i = 1; i <= D; i++)
		  a[i] = a[i-1] && (COEFS[i-1] > 0)
			&& ((DIMS[i-1]-1) * COEFS[i-1] + FW_IF_R[i] <= W[i-1]);
		return  a;
	endfunction : INIT_R_FLAG
	localparam r_flag_arr_t  R_FLAG = INIT_R_FLAG();

	// Terminal read-pointer increment when level i completes.
	// Index D covers the default innermost-advance case.
	typedef int  rp_inc_arr_t[D+1];
	function automatic rp_inc_arr_t  INIT_RP_INC();
		automatic rp_inc_arr_t  a;
		automatic int unsigned  rw = 0;  // cumulative rp_rewind, built inside out
		for(int  i = D; i >= 0; i--) begin
			if(i < int'(D))  rw = (DIMS[i]-1) * COEFS[i] + rw;
			a[i] = int'(W[i]) - int'(rw);
		end
		return  a;
	endfunction : INIT_RP_INC
	localparam rp_inc_arr_t  TERMINAL_RP_INC = INIT_RP_INC();

	// Negated terminal free-pointer increment when level i completes.
	// Stored negated for direct use in the negated capacity counter.
	// Index D covers the default innermost-advance case.
	typedef int  fp_inc_arr_t[D+1];
	function automatic fp_inc_arr_t  INIT_FP_INC();
		automatic fp_inc_arr_t  a;
		automatic int unsigned  fw = 0;  // cumulative fp_rewind, built inside out
		for(int  i = D; i >= 0; i--) begin
			if(i < int'(D))  fw = R_FLAG[i+1]? (DIMS[i]-1) * COEFS[i] + fw : 0;
			a[i] = R_FLAG[i]? int'(fw) - int'(W[i]) : 0;
		end
		return  a;
	endfunction : INIT_FP_INC
	localparam fp_inc_arr_t  TERMINAL_FP_INC = INIT_FP_INC();

	// Maximum buffer capacity requirement: the larger of the max backward
	// read-pointer retraction and the max read-free pointer gap.
	function automatic int unsigned  INIT_MAX_OCCUPANCY();
		automatic int unsigned  m  = 0;
		automatic int unsigned  rw = 0;
		automatic int unsigned  fw = 0;
		for(int unsigned  i = 0; i < D; i++) begin
			automatic int  t = -TERMINAL_RP_INC[i];
			if(t > int'(m))  m = t;
		end
		for(int  i = D-1; i >= 0; i--) begin
			rw = (DIMS[i]-1) * COEFS[i] + rw;
			fw = R_FLAG[i+1]? (DIMS[i]-1) * COEFS[i] + fw : 0;
			if(rw - fw > m)  m = rw - fw;
		end
		return  m;
	endfunction : INIT_MAX_OCCUPANCY

	//=== Throughput-preserving buffer sizing ===============================
	// Unit: one iteration of the deepest free-pointer level q (R_FLAG[q]=1,
	// R_FLAG[q+1]=0). Between units, the free pointer jumps by
	// -TERMINAL_FP_INC[l+1] (carry into level l<q) or -TERMINAL_FP_INC[0]
	// (frame). Joiner J = jump - Pq, i.e. free-pointer progress minus read
	// cycles spent; the worst lead is a max-subarray over the joiners.
	typedef struct { longint  tot, pre, suf, best; } blk_t;
	localparam longint  BLK_NEG = -(longint'(1) << 60);
	function automatic longint  max2(longint  a, longint  b);  return  a < b? b : a;  endfunction
	function automatic blk_t  compose(blk_t  b, longint  n, longint  j);
		automatic blk_t  r = b;
		if(n > 1) begin
			r.tot  = n*b.tot + (n-1)*j;
			r.pre  = b.pre + max2(0, (n-1)*(b.tot+j));
			r.suf  = b.suf + max2(0, (n-1)*(b.tot+j));
			r.best = max2(b.best, b.suf + j + b.pre + max2(0, (n-2)*(b.tot+j)));
		end
		return  r;
	endfunction : compose
	function automatic longint  INIT_THROUGHPUT_OCC();
		automatic int      q = 0;
		automatic longint  pq = 1, delta = 0, gapq = 0, u, jf, cyc;
		automatic blk_t    bp = '{0, 0, 0, BLK_NEG};
		automatic blk_t    bn = '{0, 0, 0, BLK_NEG};
		for(int i = 0; i <= D; i++)  if(R_FLAG[i])  q = i;
		for(int i = D-1; i >= q; i--) begin
			delta += (DIMS[i]-1) * max2(0, longint'(COEFS[i]) - pq);
			gapq  += (DIMS[i]-1) * COEFS[i];
			pq    *= DIMS[i];
		end
		u = delta + pq - 1;
		for(int i = q-1; i >= 0; i--) begin
			automatic longint  j = -longint'(TERMINAL_FP_INC[i+1]) - pq;
			bp = compose(bp, DIMS[i],  j);
			bn = compose(bn, DIMS[i], -j);
		end
		jf  = -longint'(TERMINAL_FP_INC[0]) - pq;
		cyc = bp.tot + jf;	// = FM_SIZE - reads per frame
		return  cyc <= 0?
			max2(gapq, u + max2(bp.best, bp.suf + jf + bp.pre)) :
			u + max2(0, max2(bn.best, bn.suf - jf + bn.pre));
	endfunction : INIT_THROUGHPUT_OCC
	localparam longint  THROUGHPUT_OCC = INIT_THROUGHPUT_OCC();
	// Writes become visible to has_data after 2 cycles (Wp -> WpZ -> compare)
	// and irdy requires Wp - free pointer < BUF_SIZE-1 (Cap saturates at 0):
	// stall-free iff BUF_SIZE >= THROUGHPUT_OCC + 4.

	//=== Buffer Sizing =====================================================
	// WP_DELAY: write-pointer register delay (WpZ); the effective
	// write-to-read latency is 2 cycles and the usable capacity BUF_SIZE-1.
	localparam int unsigned  WP_DELAY  = 1;
	localparam int unsigned  MAX_OCCUPANCY = INIT_MAX_OCCUPANCY();
	// Stall-free sizing; THROUGHPUT_OCC + 4 always covers the functional
	// bound, which is kept as a safety floor.
	localparam longint  REQ_DEPTH = max2(THROUGHPUT_OCC + 4, MAX_OCCUPANCY + WP_DELAY + 2);
	localparam int unsigned  ADDR_BITS = $clog2(REQ_DEPTH);
	localparam int unsigned  BUF_SIZE  = 1 << ADDR_BITS;

	// Maximum forward jump of the read pointer at any level completion.
	function automatic int unsigned  INIT_MAX_RP_ADVANCE();
		automatic int unsigned  m = 0;
		for(int unsigned  i = 0; i <= D; i++)
			if(TERMINAL_RP_INC[i] > int'(m))  m = TERMINAL_RP_INC[i];
		return  m;
	endfunction : INIT_MAX_RP_ADVANCE
	localparam int unsigned  MAX_RP_ADVANCE = INIT_MAX_RP_ADVANCE();

	// Pointer type: signed differences must stay unambiguous not only across
	// the buffer but also across forward jumps of the read and free pointers
	// over input that no output uses (floor remainder, kernel < stride). Input
	// arriving behind the free pointer is consumed into the buffer as no live
	// data is held then. Bounds:
	//	- Rp only advances while Rp < Wp, so Rp - Wp < MAX_RP_ADVANCE;
	//	- the free pointer never passes Rp, so the same bound applies to it;
	//	- Wp - free pointer <= BUF_SIZE.
	localparam int unsigned  SPAN = BUF_SIZE > MAX_RP_ADVANCE? BUF_SIZE : MAX_RP_ADVANCE;
	typedef logic signed [$clog2(SPAN)+1:0]  ptr_t;

	// Pointer increment type: must accommodate the largest absolute increment.
	function automatic int unsigned  INIT_MAX_ABS_INC();
		automatic int unsigned  m = 0;
		for(int unsigned  i = 0; i <= D; i++) begin
			automatic int unsigned  rp_abs = TERMINAL_RP_INC[i] < 0? -TERMINAL_RP_INC[i] : TERMINAL_RP_INC[i];
			automatic int unsigned  fp_abs = TERMINAL_FP_INC[i] < 0? -TERMINAL_FP_INC[i] : TERMINAL_FP_INC[i];
			if(rp_abs > m)  m = rp_abs;
			if(fp_abs > m)  m = fp_abs;
		end
		return  m;
	endfunction : INIT_MAX_ABS_INC
	localparam int unsigned  INC_BITS = 1 + $clog2(INIT_MAX_ABS_INC() + 1);
	typedef logic signed [INC_BITS-1:0]  inc_t;

	//=== Nest Counters =====================================================
	// done[i]: level i has exhausted its iterations (sign-bit of Cnt).
	// term[i]: level i and all inner levels completed simultaneously.
	uwire [D:0]  done;
	uwire [D:0]  term;
	assign	done[D] = 1;
	assign	term[D] = 1;

	uwire  advance;  // forward-declared, defined in output section

	for(genvar  i = 0; i < D; i++) begin : genCnt
		uwire  step = advance && term[i+1];

		if(DIMS[i] == 1) begin : genTrivial
			assign	done[i] = 1;
		end : genTrivial
		else begin : genCounter
			logic signed [$clog2(DIMS[i]-1):0]  Cnt = DIMS[i]-2;  // DIMS[i]-2, ..., 1, 0, -1 (done)
			always_ff @(posedge clk) begin
				if(rst)        Cnt <= DIMS[i]-2;
				else if(step)  Cnt <= Cnt + (done[i]? $signed(DIMS[i])-1 : -1);
			end
			assign	done[i] = Cnt[$left(Cnt)];
		end : genCounter

		assign	term[i] = term[i+1] && done[i];
	end : genCnt

	//=== Pointer Increment Mux (Combinational) =============================
	inc_t  rp_inc;
	inc_t  fp_inc;
	always_comb begin
		rp_inc = 0;
		fp_inc = 0;
		for(int  i = D; i >= 0; i--) begin
			if(term[i]) begin
				rp_inc = TERMINAL_RP_INC[i];
				if(R_FLAG[i])  fp_inc = TERMINAL_FP_INC[i];
			end
		end
	end

	//=== Circular Buffer and Pointer Management ============================
	(* RAM_STYLE = RAM_STYLE *)
	logic [DATA_WIDTH-1:0]  Buf[BUF_SIZE];
	ptr_t  Wp  = 0;
	ptr_t  WpZ = 0;
	ptr_t  Rp  = 0;
	ptr_t  Cap = -BUF_SIZE+1;  // -BUF_SIZE+1, ..., -1, 0 (full)

	uwire  has_data = $signed(Rp - WpZ) < 0;

	assign	irdy = Cap[$left(Cap)];

	// Buffer memory — one write port, one registered read port.
	//	- Direct path: speculative pre-fetch, i.e. on advance, read from the
	//	  next Rp so that BufRd is ready without a settling cycle.
	//	- Read-ahead path: read from Rp as the read is issued (on advance);
	//	  the result is delayed by RD_LAT register stages, which Vivado can
	//	  absorb into a deep (URAM) memory cascade.
	localparam bit  READ_AHEAD = BUF_SIZE > READ_AHEAD_DEPTH;
	logic [DATA_WIDTH-1:0]  BufRd;
	uwire ptr_t  rd_ptr = READ_AHEAD? Rp : Rp + (advance? ptr_t'(rp_inc) : ptr_t'(0));
	always_ff @(posedge clk) begin
		if(irdy)  Buf[Wp[ADDR_BITS-1:0]] <= idat;
		BufRd <= Buf[rd_ptr[ADDR_BITS-1:0]];
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Wp  <= 0;
			WpZ <= 0;
			Rp  <= 0;
			Cap <= -BUF_SIZE+1;
		end
		else begin
			automatic logic  istep = irdy && ivld;
			WpZ <= Wp;
			Wp  <= Wp + istep;
			Cap <= Cap + (advance? ptr_t'(fp_inc) : ptr_t'(0)) + istep;
			if(advance)  Rp <= Rp + ptr_t'(rp_inc);
		end
	end

	// synthesis translate_off
	// Unbounded shadow pointers validating the dimensioning of ptr_t.
	longint  SWp = 0;
	longint  SRp = 0;
	longint  SFp = 0;
	always_ff @(posedge clk) begin
		if(rst) begin
			SWp <= 0;
			SRp <= 0;
			SFp <= 0;
		end
		else begin
			automatic longint  swp = SWp + (irdy && ivld);
			automatic longint  srp = SRp + (advance? longint'(rp_inc) : 0);
			automatic longint  sfp = SFp - (advance? longint'(fp_inc) : 0);
			assert(srp - swp <= longint'(MAX_RP_ADVANCE)) else $error("%m: Rp leads Wp by %0d > %0d.", srp - swp, MAX_RP_ADVANCE);
			assert(sfp <= srp)                  else $error("%m: free pointer passed Rp.");
			assert(swp - sfp <= longint'(BUF_SIZE)) else $error("%m: buffer overrun.");
			SWp <= swp;
			SRp <= srp;
			SFp <= sfp;
		end
	end
	always_ff @(posedge clk) begin
		if(!rst) begin
			assert(longint'(ptr_t'(Rp - Wp)) == SRp - SWp)        else $error("%m: Rp/Wp difference wrapped.");
			assert(longint'(Cap) == SWp - SFp - longint'(BUF_SIZE) + 1) else $error("%m: capacity counter wrapped.");
		end
	end
	// synthesis translate_on

	//=== Output Stage ======================================================
	logic  OVld = 0;
	logic [DATA_WIDTH-1:0]  OBuf = 'x;
	logic [D-1:0]  OLst  = 'x;
	logic [D-1:0]  ODone = 'x;

	if(!READ_AHEAD) begin : genDirect
		// The output register loads the pre-fetched buffer word directly.
		always_ff @(posedge clk) begin
			if(rst) begin
				OVld  <= 0;
				OBuf  <= 'x;
				OLst  <= 'x;
				ODone <= 'x;
			end
			else if(!OVld || ordy) begin
				OVld  <= has_data;
				OBuf  <= BufRd;
				OLst  <= term[D-1:0];
				ODone <= done[D-1:0];
			end
		end

		assign	advance = has_data && (!OVld || ordy);
	end : genDirect
	else begin : genReadAhead
		// Deep buffers become memory cascades whose read latency grows with
		// depth. As in the HLS input_gen<>, reads are issued ahead of the output
		// into a small queue; credits bound the reads in flight plus those
		// queued, so that the queue never overruns. RD_LAT follows the latency
		// Vitis HLS picks for such cascades: 5 at 512K, 9 at 1M words.
		localparam int unsigned  RD_LAT   = 1 + BUF_SIZE/131072 > 3? 1 + BUF_SIZE/131072 : 3;
		localparam int unsigned  RA_DEPTH = 1 << $clog2(RD_LAT + 4 > 16? RD_LAT + 4 : 16);
		localparam int unsigned  QA_BITS  = $clog2(RA_DEPTH);

		// Read pipeline: BufRd is stage 1, flags travel alongside.
		logic [DATA_WIDTH-1:0]  RdDat[2:RD_LAT];
		logic          RdVld[1:RD_LAT];
		logic [D-1:0]  RdLst[1:RD_LAT];
		logic [D-1:0]  RdDone[1:RD_LAT];
		always_ff @(posedge clk) begin
			if(rst) begin
				for(int unsigned  i = 1; i <= RD_LAT; i++)  RdVld[i] <= 0;
			end
			else begin
				RdVld[1] <= advance;
				for(int unsigned  i = 2; i <= RD_LAT; i++)  RdVld[i] <= RdVld[i-1];
			end
			RdLst [1] <= term[D-1:0];
			RdDone[1] <= done[D-1:0];
			RdDat [2] <= BufRd;
			for(int unsigned  i = 2; i <= RD_LAT; i++) begin
				if(i > 2)  RdDat[i] <= RdDat[i-1];
				RdLst [i] <= RdLst [i-1];
				RdDone[i] <= RdDone[i-1];
			end
		end

		// Read-ahead queue
		logic [DATA_WIDTH-1:0]  QDat [RA_DEPTH];
		logic [D-1:0]           QLst [RA_DEPTH];
		logic [D-1:0]           QDone[RA_DEPTH];
		logic [QA_BITS:0]  QWr = 0;
		logic [QA_BITS:0]  QRd = 0;
		logic [$clog2(RA_DEPTH+1)-1:0]  Credits = RA_DEPTH;
		uwire  qvld = QWr != QRd;
		uwire  pop  = qvld && (!OVld || ordy);
		always_ff @(posedge clk) begin
			if(RdVld[RD_LAT]) begin
				QDat [QWr[QA_BITS-1:0]] <= RdDat [RD_LAT];
				QLst [QWr[QA_BITS-1:0]] <= RdLst [RD_LAT];
				QDone[QWr[QA_BITS-1:0]] <= RdDone[RD_LAT];
			end
		end
		always_ff @(posedge clk) begin
			if(rst) begin
				QWr <= 0;
				QRd <= 0;
				Credits <= RA_DEPTH;
				OVld  <= 0;
				OBuf  <= 'x;
				OLst  <= 'x;
				ODone <= 'x;
			end
			else begin
				QWr <= QWr + RdVld[RD_LAT];
				QRd <= QRd + pop;
				Credits <= Credits + pop - advance;
				if(!OVld || ordy) begin
					OVld  <= qvld;
					OBuf  <= QDat [QRd[QA_BITS-1:0]];
					OLst  <= QLst [QRd[QA_BITS-1:0]];
					ODone <= QDone[QRd[QA_BITS-1:0]];
				end
			end
		end

		assign	advance = has_data && (Credits != 0);

		// synthesis translate_off
		always_ff @(posedge clk) begin
			if(!rst)  assert((QA_BITS+1)'(QWr - QRd) <= RA_DEPTH) else $error("%m: read-ahead queue overrun.");
		end
		// synthesis translate_on
	end : genReadAhead

	assign	odat  = OBuf;
	assign	ovld  = OVld;
	assign	olst  = OLst;
	assign	odone = ODone;

endmodule : input_gen

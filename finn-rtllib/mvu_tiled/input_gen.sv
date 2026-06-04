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
 *	term[D-1:0] synchronous with each output beat.
 ***************************************************************************/

module input_gen #(
	int unsigned  DATA_WIDTH,
	int unsigned  FM_SIZE,
	int unsigned  D,
	int unsigned  DIMS[D],
	int unsigned  COEFS[D]
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

	// Free-pointer responsibility flag per level.
	// R_FLAG[i] is the R flag passed into level i from its parent.
	typedef bit  r_flag_arr_t[D+1];
	function automatic r_flag_arr_t  INIT_R_FLAG();
		automatic r_flag_arr_t  a;
		a[0] = 1;
		for(int unsigned  i = 1; i <= D; i++)
		  a[i] = a[i-1] && (COEFS[i-1] > 0)
			&& (COEFS[i-1] * DIMS[i-1] <= W[i-1]);
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

	//=== Buffer Sizing =====================================================
	localparam int unsigned  WP_DELAY  = 1;
	localparam int unsigned  MAX_OCCUPANCY = INIT_MAX_OCCUPANCY();
	localparam int unsigned  ADDR_BITS = $clog2(MAX_OCCUPANCY + WP_DELAY + 2);
	localparam int unsigned  BUF_SIZE  = 1 << ADDR_BITS;

	// Pointer type: one extra bit for signed wrap-around detection.
	typedef logic signed [ADDR_BITS:0]  ptr_t;

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
	logic [DATA_WIDTH-1:0]  Buf[BUF_SIZE];
	ptr_t  Wp  = 0;
	ptr_t  WpZ = 0;
	ptr_t  Rp  = 0;
	ptr_t  Cap = -BUF_SIZE+1;  // -BUF_SIZE+1, ..., -1, 0 (full)

	uwire  has_data = $signed(Rp - WpZ) < 0;

	assign	irdy = Cap[$left(Cap)];

	// Buffer memory — one write port, one registered read port.
	// Speculative pre-fetch: on advance, read from the next Rp so that
	// BufRd is ready without a settling cycle.
	logic [DATA_WIDTH-1:0]  BufRd;
	uwire ptr_t  rd_ptr = Rp + (advance? ptr_t'(rp_inc) : ptr_t'(0));
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

	//=== Output Stage ======================================================
	logic  OVld = 0;
	logic [DATA_WIDTH-1:0]  OBuf = 'x;
	logic [D-1:0]  OLst  = 'x;
	logic [D-1:0]  ODone = 'x;
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

	assign	odat  = OBuf;
	assign	ovld  = OVld;
	assign	olst  = OLst;
	assign	odone = ODone;

endmodule : input_gen

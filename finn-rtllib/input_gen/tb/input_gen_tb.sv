/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief
 *	Self-checking testbench for input_gen driven by a perfect loop nest:
 *	by default the pooling nest (oh, ow, cf, kh, kw) of sliding-window
 *	generation, or any nest of up to six levels given by NEST_D,
 *	NEST_FM_SIZE, NEST_DIMn and NEST_COEFn (e.g. the replay and reorder
 *	nests of the tiled MVU, the broadcasts of Where).
 *
 *	Streams FRAMES feature maps back-to-back and checks:
 *	  - every emitted element and its level-completion flags (olst, odone)
 *	    against the loop nest,
 *	  - that every input element is accepted,
 *	  - that no extra output appears after the expected sequence,
 *	  - optionally (EXP_BUF_SIZE > 0), the buffer depth the module derives at
 *	    elaboration against an expected value (e.g. the Python mirror
 *	    finn.util.input_gen.input_gen_buffer_depth()), and
 *	  - with FULL_RATE, that the buffer sizing is stall-free: with input
 *	    always valid and output always ready, read k happens at
 *	    max(read k-1 + 1, arrival of its element + latency), the unbounded
 *	    schedule the stall-free sizing promises.
 *	Without FULL_RATE, both handshakes are randomized (~70% active).
 *	The geometry is selected through the scalar top-level parameters so that
 *	a sweep can be run via `xelab -generic_top`. Geometries skipping input
 *	(floor remainder, kernel < stride) are the primary regression target.
 ***************************************************************************/

module input_gen_tb #(
	int unsigned  H  = 9,
	int unsigned  W  = 9,
	int unsigned  CF = 2,
	int unsigned  KH = 1,
	int unsigned  KW = 2,
	int unsigned  SH = 2,
	int unsigned  SW = 3,
	int unsigned  FRAMES = 3,
	int unsigned  SEED = 1,
	int unsigned  READ_AHEAD_DEPTH = 262144,	// 0: force the read-ahead path
	bit  FULL_RATE = 0,	// 1: always valid/ready, check the stall-free schedule
	int unsigned  EXP_BUF_SIZE = 0,	// expected BUF_SIZE, 0: unchecked

	// Generic nest (NEST_D > 0) overriding the pooling geometry
	int unsigned  NEST_D = 0,
	int unsigned  NEST_FM_SIZE = 1,
	int unsigned  NEST_DIM0 = 1, NEST_DIM1 = 1, NEST_DIM2 = 1,
	int unsigned  NEST_DIM3 = 1, NEST_DIM4 = 1, NEST_DIM5 = 1,
	int unsigned  NEST_COEF0 = 0, NEST_COEF1 = 0, NEST_COEF2 = 0,
	int unsigned  NEST_COEF3 = 0, NEST_COEF4 = 0, NEST_COEF5 = 0
);

	localparam int unsigned  OH = (H-KH)/SH + 1;
	localparam int unsigned  OW = (W-KW)/SW + 1;
	localparam bit  POOL = NEST_D == 0;
	localparam int unsigned  FM_SIZE = POOL? H*W*CF : NEST_FM_SIZE;
	localparam int unsigned  D = POOL? 5 : NEST_D;

	typedef int unsigned  nest_t[D];
	function automatic nest_t  INIT_DIMS();
		automatic int unsigned  n[6] = '{ NEST_DIM0, NEST_DIM1, NEST_DIM2, NEST_DIM3, NEST_DIM4, NEST_DIM5 };
		automatic int unsigned  p[5] = '{ OH, OW, CF, KH, KW };
		automatic nest_t  a;
		for(int unsigned  i = 0; i < D; i++)  a[i] = POOL? p[i] : n[i];
		return  a;
	endfunction : INIT_DIMS
	function automatic nest_t  INIT_COEFS();
		automatic int unsigned  n[6] = '{ NEST_COEF0, NEST_COEF1, NEST_COEF2, NEST_COEF3, NEST_COEF4, NEST_COEF5 };
		automatic int unsigned  p[5] = '{ SH*W*CF, SW*CF, 1, W*CF, CF };
		automatic nest_t  a;
		for(int unsigned  i = 0; i < D; i++)  a[i] = POOL? p[i] : n[i];
		return  a;
	endfunction : INIT_COEFS
	localparam nest_t  DIMS  = INIT_DIMS();
	localparam nest_t  COEFS = INIT_COEFS();

	function automatic int unsigned  NEST_SIZE();
		automatic int unsigned  n = 1;
		foreach(DIMS[i])  n *= DIMS[i];
		return  n;
	endfunction : NEST_SIZE

	localparam int unsigned  IN_TOTAL  = FRAMES * FM_SIZE;
	localparam int unsigned  OUT_TOTAL = FRAMES * NEST_SIZE();
	localparam int unsigned  DATA_WIDTH = 32;

	//- Clock & Reset -------------------
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;

	//- DUT -----------------------------
	logic [DATA_WIDTH-1:0]  idat;
	logic  ivld;
	uwire  irdy;
	uwire [DATA_WIDTH-1:0]  odat;
	uwire  ovld;
	uwire [D-1:0]  olst;
	uwire [D-1:0]  odone;
	logic  ordy;

	input_gen #(
		.DATA_WIDTH(DATA_WIDTH),
		.FM_SIZE(FM_SIZE),
		.D(D), .DIMS(DIMS), .COEFS(COEFS),
		.READ_AHEAD_DEPTH(READ_AHEAD_DEPTH)
	) dut (
		.clk, .rst,
		.idat, .ivld, .irdy,
		.odat, .ovld, .olst, .odone, .ordy
	);

	//- Expected Output Sequence --------
	typedef struct {
		logic [DATA_WIDTH-1:0]  dat;
		logic [D-1:0]  lst;	// all levels i and inner at their last iteration
		logic [D-1:0]  done;	// level i at its last iteration
	} exp_t;
	exp_t  Exp[$];
	initial begin
		for(int unsigned  f = 0; f < FRAMES; f++) begin
			automatic int unsigned  idx[D] = '{ default: 0 };
			repeat(NEST_SIZE()) begin
				automatic exp_t  e;
				automatic logic  term = 1;
				e.dat = f*FM_SIZE;
				for(int  i = D-1; i >= 0; i--) begin
					e.dat += COEFS[i]*idx[i];
					e.done[i] = idx[i] == DIMS[i]-1;
					term &= e.done[i];
					e.lst[i] = term;
				end
				Exp.push_back(e);

				// advance the odometer, innermost level first
				for(int  i = D-1; i >= 0; i--) begin
					if(++idx[i] < DIMS[i])  break;
					idx[i] = 0;
				end
			end
		end
	end

	//- Stimulus & Checking -------------
	int unsigned  InCnt  = 0;
	int unsigned  OutCnt = 0;
	int unsigned  Errors = 0;
	int unsigned  Cycle  = 0;

	initial begin
		automatic int unsigned  seed = SEED;
		automatic int unsigned  idle;
		automatic longint  ideal = -(longint'(1) << 40);	// unbounded schedule
		automatic longint  ofs;	// its constant offset to the observed one

		idat = 'x;
		ivld = 0;
		ordy = 0;
		repeat(8) @(posedge clk);
		rst <= 0;

		// Run until all input is accepted and all output observed, then keep
		// going for a while to catch extra output.
		idle = 0;
		while(idle < 20*FM_SIZE + 1000) begin
			if(FULL_RATE) begin
				ivld <= InCnt < IN_TOTAL;
				ordy <= 1;
			end
			else begin
				// randomized handshakes (~70% active)
				if(InCnt < IN_TOTAL) begin
					ivld <= ($urandom(seed) % 10) < 7;
					seed++;
				end
				else  ivld <= 0;
				ordy <= ($urandom(seed) % 10) < 7;
				seed++;
			end
			idat <= InCnt;

			@(posedge clk);
			Cycle++;
			if(ivld && irdy)  InCnt++;
			if(ovld && ordy) begin
				if(OutCnt >= OUT_TOTAL) begin
					if(Errors < 5)  $error("Extra output #%0d: %0d", OutCnt, odat);
					Errors++;
				end
				else begin
					automatic exp_t  e = Exp[OutCnt];
					if((odat !== e.dat) || (olst !== e.lst) || (odone !== e.done)) begin
						if(Errors < 5)  $error("Output #%0d: got %0d/%b/%b, expected %0d/%b/%b (data/olst/odone)", OutCnt, odat, olst, odone, e.dat, e.lst, e.done);
						Errors++;
					end

					// Stall-free: the read of element dat (written at cycle
					// dat) only waits for that write, never for buffer space.
					ideal = (ideal + 1 > longint'(e.dat))? ideal + 1 : longint'(e.dat);
					if(OutCnt == 0)  ofs = Cycle - ideal;
					else if(FULL_RATE && (Cycle - ideal != ofs)) begin
						if(Errors < 5)  $error("Output #%0d at cycle %0d, %0d cycle(s) behind the stall-free schedule.", OutCnt, Cycle, Cycle - ideal - ofs);
						Errors++;
					end
				end
				OutCnt++;
				idle = 0;
			end
			else  idle++;
		end

		if(InCnt  != IN_TOTAL)   begin $error("Accepted %0d of %0d inputs.", InCnt, IN_TOTAL); Errors++; end
		if(OutCnt != OUT_TOTAL)  begin $error("Observed %0d of %0d outputs.", OutCnt, OUT_TOTAL); Errors++; end
		if((EXP_BUF_SIZE != 0) && (dut.BUF_SIZE != EXP_BUF_SIZE)) begin
			$error("BUF_SIZE=%0d, expected %0d.", dut.BUF_SIZE, EXP_BUF_SIZE);
			Errors++;
		end
		if(POOL)  $display("RESULT %s H=%0d W=%0d CF=%0d K=%0dx%0d S=%0dx%0d: %0d/%0d out, %0d/%0d in, %0d error(s), BUF_SIZE=%0d",
			Errors? "FAIL" : "PASS", H, W, CF, KH, KW, SH, SW, OutCnt, OUT_TOTAL, InCnt, IN_TOTAL, Errors, dut.BUF_SIZE);
		else      $display("RESULT %s FM_SIZE=%0d DIMS=%p COEFS=%p: %0d/%0d out, %0d/%0d in, %0d error(s), BUF_SIZE=%0d",
			Errors? "FAIL" : "PASS", FM_SIZE, DIMS, COEFS, OutCnt, OUT_TOTAL, InCnt, IN_TOTAL, Errors, dut.BUF_SIZE);
		$finish;
	end

endmodule : input_gen_tb

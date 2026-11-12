/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief
 *	Self-checking testbench for input_gen driven by a pooling loop nest
 *	(oh, ow, cf, kh, kw) as used for sliding-window generation.
 *
 *	Streams FRAMES feature maps back-to-back under random input and output
 *	backpressure and checks:
 *	  - every emitted element and its window-completion marker against the
 *	    loop nest,
 *	  - that every input element is accepted, and
 *	  - that no extra output appears after the expected sequence.
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
	int unsigned  READ_AHEAD_DEPTH = 262144	// 0: force the read-ahead path
);

	localparam int unsigned  OH = (H-KH)/SH + 1;
	localparam int unsigned  OW = (W-KW)/SW + 1;
	localparam int unsigned  FM_SIZE = H*W*CF;
	localparam int unsigned  D = 5;
	localparam int unsigned  DIMS [D] = '{ OH, OW, CF, KH, KW };
	localparam int unsigned  COEFS[D] = '{ SH*W*CF, SW*CF, 1, W*CF, CF };

	localparam int unsigned  IN_TOTAL  = FRAMES * FM_SIZE;
	localparam int unsigned  OUT_TOTAL = FRAMES * OH*OW*CF*KH*KW;
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
	logic  ordy;

	input_gen #(
		.DATA_WIDTH(DATA_WIDTH),
		.FM_SIZE(FM_SIZE),
		.D(D), .DIMS(DIMS), .COEFS(COEFS),
		.READ_AHEAD_DEPTH(READ_AHEAD_DEPTH)
	) dut (
		.clk, .rst,
		.idat, .ivld, .irdy,
		.odat, .ovld, .olst, .ordy
	);

	//- Expected Output Sequence --------
	typedef struct {
		logic [DATA_WIDTH-1:0]  dat;
		logic  lst;
	} exp_t;
	exp_t  Exp[$];
	initial begin
		for(int unsigned  f = 0; f < FRAMES; f++)
			for(int unsigned  oh = 0; oh < OH; oh++)
				for(int unsigned  ow = 0; ow < OW; ow++)
					for(int unsigned  cf = 0; cf < CF; cf++)
						for(int unsigned  kh = 0; kh < KH; kh++)
							for(int unsigned  kw = 0; kw < KW; kw++)
								Exp.push_back('{
									dat: f*FM_SIZE + ((oh*SH + kh)*W + (ow*SW + kw))*CF + cf,
									lst: (kh == KH-1) && (kw == KW-1)
								});
	end

	//- Stimulus & Checking -------------
	int unsigned  InCnt  = 0;
	int unsigned  OutCnt = 0;
	int unsigned  Errors = 0;

	initial begin
		automatic int unsigned  seed = SEED;
		automatic int unsigned  idle;

		idat = 'x;
		ivld = 0;
		ordy = 0;
		repeat(8) @(posedge clk);
		rst <= 0;

		// Run until all input is accepted and all output observed, then keep
		// going for a while to catch extra output.
		idle = 0;
		while(idle < 20*FM_SIZE + 1000) begin
			// randomized handshakes (~70% active)
			if(InCnt < IN_TOTAL) begin
				ivld <= ($urandom(seed) % 10) < 7;
				seed++;
			end
			else  ivld <= 0;
			idat <= InCnt;
			ordy <= ($urandom(seed) % 10) < 7;
			seed++;

			@(posedge clk);
			if(ivld && irdy)  InCnt++;
			if(ovld && ordy) begin
				if(OutCnt >= OUT_TOTAL) begin
					if(Errors < 5)  $error("Extra output #%0d: %0d", OutCnt, odat);
					Errors++;
				end
				else begin
					automatic exp_t  e = Exp[OutCnt];
					if((odat !== e.dat) || (olst[D-2] !== e.lst)) begin
						if(Errors < 5)  $error("Output #%0d: got %0d/%0b, expected %0d/%0b", OutCnt, odat, olst[D-2], e.dat, e.lst);
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
		$display("RESULT %s H=%0d W=%0d CF=%0d K=%0dx%0d S=%0dx%0d: %0d/%0d out, %0d/%0d in, %0d error(s), BUF_SIZE=%0d",
			Errors? "FAIL" : "PASS", H, W, CF, KH, KW, SH, SW, OutCnt, OUT_TOTAL, InCnt, IN_TOTAL, Errors, dut.BUF_SIZE);
		$finish;
	end

endmodule : input_gen_tb

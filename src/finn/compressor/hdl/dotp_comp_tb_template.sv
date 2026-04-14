/******************************************************************************
 * Testbench for LUT-based dotp_comp module.
 * Exercises the finnlib protocol: clk, rst, en, last, zero, w, a -> vld, p
 *
 * Generated from template for config: PE={pe}, SIMD={simd}, WW={ww}, AW={aw}
 ******************************************************************************/
module dotp_comp_{full_sig}_tb;

	localparam int unsigned  ROUNDS = 217;

	localparam int unsigned  PE   = {pe};
	localparam int unsigned  SIMD = {simd};
	localparam int unsigned  WEIGHT_WIDTH     = {ww};
	localparam int unsigned  ACTIVATION_WIDTH = {aw};
	localparam int unsigned  ACCU_WIDTH       = {accu_width};
	localparam bit  SIGNED_ACTIVATIONS = {signed_act};

	typedef logic signed [WEIGHT_WIDTH    -1:0]  weight_t;
	typedef logic        [ACTIVATION_WIDTH-1:0]  activation_t;
	typedef logic signed [ACCU_WIDTH      -1:0]  accu_t;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// DUT
	logic  en;
	logic  last;
	logic  zero;
	weight_t     [PE-1:0][SIMD-1:0]  w;
	activation_t         [SIMD-1:0]  a;
	uwire  vld;
	accu_t [PE-1:0]  p;

	dotp_comp #(
		.PE(PE), .SIMD(SIMD),
		.WEIGHT_WIDTH(WEIGHT_WIDTH),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH),
		.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
		.COMP_PIPELINE_DEPTH({comp_depth})
	) dut (
		.clk, .rst, .en,
		.last, .zero, .w, .a,
		.vld, .p
	);

	//-----------------------------------------------------------------------
	// Input Feed & Reference Model
	accu_t [PE-1:0]  Q[$];
	int unsigned  RoundsPushed = 0;

	// Drive one dot-product round with given weights/activations on a single
	// enabled cycle (en=1, last=1, zero=0).  Computes the expected accumulator
	// value and pushes it into the checker queue.
	// Results drain naturally when en resumes in subsequent tests or the
	// final flush — no en=1 drain cycles needed.
	task automatic feed_single(
		input weight_t     [PE-1:0][SIMD-1:0]  ww,
		input activation_t         [SIMD-1:0]  aa
	);
		automatic accu_t [PE-1:0]  pp = '{ default: '0 };
		for(int unsigned  pe = 0; pe < PE; pe++) begin
			for(int unsigned  simd = 0; simd < SIMD; simd++) begin
				pp[pe] += $signed(ww[pe][simd])
					* $signed({SIGNED_ACTIVATIONS && aa[simd][ACTIVATION_WIDTH-1], aa[simd]});
			end
		end
		en   <= 1;
		last <= 1;
		zero <= 0;
		w    <= ww;
		a    <= aa;
		@(posedge clk);
		en   <= 0;
		last <= 'x;
		zero <= 'x;
		w    <= 'x;
		a    <= 'x;
		Q.push_back(pp);
		RoundsPushed++;
	endtask : feed_single

	task automatic feed_zero_round();
		automatic accu_t [PE-1:0]  pp = '{ default: '0 };
		en   <= 1;
		last <= 1;
		zero <= 1;
		w    <= '0;
		a    <= '0;
		@(posedge clk);
		en   <= 0;
		last <= 'x;
		zero <= 'x;
		w    <= 'x;
		a    <= 'x;
		Q.push_back(pp);
		RoundsPushed++;
	endtask : feed_zero_round

	initial begin
		en = 0;
		last = 'x;
		zero = 'x;
		w = 'x;
		a = 'x;
		@(posedge clk iff !rst);

		//---------------------------------------------------------------
		// Directed edge-case tests
		//---------------------------------------------------------------

		// All zeros
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww = '0;
			automatic activation_t         [SIMD-1:0]  aa = '0;
			feed_single(ww, aa);
		end

		// Zero round via zero flag
		feed_zero_round();

		// All ones
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww = '1;
			automatic activation_t         [SIMD-1:0]  aa = '1;
			feed_single(ww, aa);
		end

		// Max positive weights, all-ones activations
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww;
			automatic activation_t         [SIMD-1:0]  aa = '1;
			for(int unsigned  pe = 0; pe < PE; pe++)
				for(int unsigned  s = 0; s < SIMD; s++)
					ww[pe][s] = {1'b0, {(WEIGHT_WIDTH-1){1'b1}}};
			feed_single(ww, aa);
		end

		// Single SIMD lane active (first)
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww = '0;
			automatic activation_t         [SIMD-1:0]  aa = '0;
			for(int unsigned  pe = 0; pe < PE; pe++)
				ww[pe][0] = {1'b0, {(WEIGHT_WIDTH-1){1'b1}}};
			aa[0] = '1;
			feed_single(ww, aa);
		end

		// Single SIMD lane active (last)
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww = '0;
			automatic activation_t         [SIMD-1:0]  aa = '0;
			for(int unsigned  pe = 0; pe < PE; pe++)
				ww[pe][SIMD-1] = '1;
			aa[SIMD-1] = '1;
			feed_single(ww, aa);
		end

		// Alternating weights: +max, -max, +max, ...
		begin
			automatic weight_t     [PE-1:0][SIMD-1:0]  ww;
			automatic activation_t         [SIMD-1:0]  aa = '1;
			for(int unsigned  pe = 0; pe < PE; pe++)
				for(int unsigned  s = 0; s < SIMD; s++)
					ww[pe][s] = s[0] ? '1 : {1'b0, {(WEIGHT_WIDTH-1){1'b1}}};
			feed_single(ww, aa);
		end

		// Multi-cycle accumulation: 3 cycles then last
		begin
			automatic accu_t [PE-1:0]  pp = '{ default: '0 };
			for(int unsigned  cyc = 0; cyc < 3; cyc++) begin
				automatic weight_t     [PE-1:0][SIMD-1:0]  ww;
				automatic activation_t         [SIMD-1:0]  aa;
				for(int unsigned  pe = 0; pe < PE; pe++)
					for(int unsigned  s = 0; s < SIMD; s++)
						ww[pe][s] = weight_t'(cyc + 1);
				for(int unsigned  s = 0; s < SIMD; s++)
					aa[s] = activation_t'(s + 1);

				for(int unsigned  pe = 0; pe < PE; pe++)
					for(int unsigned  s = 0; s < SIMD; s++)
						pp[pe] += $signed(ww[pe][s])
							* $signed({SIGNED_ACTIVATIONS && aa[s][ACTIVATION_WIDTH-1], aa[s]});

				en   <= 1;
				last <= (cyc == 2) ? 1 : 0;
				zero <= 0;
				w    <= ww;
				a    <= aa;
				@(posedge clk);
			end
			en   <= 0;
			last <= 'x;
			zero <= 'x;
			w    <= 'x;
			a    <= 'x;
			Q.push_back(pp);
			RoundsPushed++;
		end

		//---------------------------------------------------------------
		// Randomized tests
		//---------------------------------------------------------------
		repeat(ROUNDS) begin
			automatic accu_t [PE-1:0]  pp = '{ default: '0 };
			do begin
				en <= 0;
				last <= 'x;
				zero <= 'x;
				w <= 'x;
				a <= 'x;
				if($urandom()%31 != 0) begin
					en <= 1;
					if($urandom()%19 == 0)  zero <= 1;
					else begin
						automatic weight_t     [PE-1:0][SIMD-1:0]  ww;
						automatic activation_t         [SIMD-1:0]  aa;
						void'(std::randomize(ww, aa));

						for(int unsigned  pe = 0; pe < PE; pe++) begin
							for(int unsigned  simd = 0; simd < SIMD; simd++) begin
								automatic accu_t  m0 = $signed(ww[pe][simd])
									* $signed({SIGNED_ACTIVATIONS && aa[simd][ACTIVATION_WIDTH-1], aa[simd]});
								automatic accu_t  p0 = $signed(pp[pe]) + m0;
								// Avoid overflow by zeroing offending weight
								if(((m0 < 0) == ($signed(pp[pe]) < 0)) && ((m0 < 0) != (p0 < 0)))
									ww[pe][simd] = 0;
								else
									pp[pe] = p0;
							end
						end

						zero <= 0;
						w <= ww;
						a <= aa;
					end
					last <= $urandom() % 137 == 0;
				end
				@(posedge clk);
			end
			while(!en || !last);
			Q.push_back(pp);
			RoundsPushed++;
		end

		// Flush: keep en=1 with zero=1 for pipeline to drain
		en <= 1;
		last <= 0;
		zero <= 1;
		w <= '0;
		a <= '0;
		repeat(20) @(posedge clk);

		assert(Q.size == 0) else begin
			$error("Missing %0d outputs.", Q.size);
			$stop;
		end

		$display("Test completed successfully.");
		$finish;
	end

	//-----------------------------------------------------------------------
	// Output Checker
	int unsigned  Checks = 0;
	always_ff @(posedge clk iff !rst) begin
		if(en && vld) begin
			automatic accu_t [PE-1:0]  exp;

			assert(Q.size > 0) else begin
				$error("Spurious output: %0p.", p);
				$stop;
			end

			exp = Q.pop_front();
			assert(p === exp) else begin
				$error("Output mismatch: got %0p, expected %0p.", p, exp);
				$stop;
			end

			Checks <= Checks + 1;
		end
	end

	final begin
		assert(Checks == RoundsPushed)
			$display("Successfully performed %0d checks (%0d directed + %0d random).",
				Checks, RoundsPushed - ROUNDS, ROUNDS);
		else
			$error("Unexpected number of checks: %0d instead of %0d.",
				Checks, RoundsPushed);
	end

endmodule : dotp_comp_{full_sig}_tb

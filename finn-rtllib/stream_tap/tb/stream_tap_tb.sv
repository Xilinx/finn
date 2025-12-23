module stream_tap_tb;

	localparam int unsigned  ROUNDS = 1357;
	localparam int unsigned  DATA_WIDTH = 9;
	localparam int unsigned  TAP_REP = 7;
	typedef logic [DATA_WIDTH-1:0]  dat_t;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(10) @(posedge clk);
		rst <= 0;
	end

	// DUT
	dat_t  idat;
	logic  ivld;
	uwire  irdy;
	uwire dat_t  odat;
	uwire  ovld;
	logic  ordy;
	uwire dat_t  tdat;
	uwire  tvld;
	logic  trdy;
	stream_tap #(.DATA_WIDTH(DATA_WIDTH), .TAP_REP(TAP_REP)) dut (
		.clk, .rst,
		.idat, .ivld, .irdy,
		.odat, .ovld, .ordy,
		.tdat, .tvld, .trdy
	);

	// Stimuli
	initial begin
		fork
			// Input
			begin
				ivld =  0;
				idat = 'x;
				@(posedge clk iff !rst);

				for(int unsigned  r = 0; r < ROUNDS; r++) begin
					while($urandom()%11 == 0) @(posedge clk);

					ivld <= 1;
					idat <= r;
					@(posedge clk iff irdy);
					ivld <= 0;
					idat <= 'x;
				end
			end

			// Output Sequence Checker
			begin
				ordy = 0;
				@(posedge clk iff !rst);

				for(int unsigned  r = 0; r < ROUNDS; r++) begin
					automatic dat_t  exp = r;

					while($urandom()%7 == 0) @(posedge clk);

					ordy <= 1;
					@(posedge clk iff ovld);
					assert(odat == exp) else begin
						$error("Output mismatch: %0d instead of %0d.", odat, exp);
						$stop;
					end
					ordy <= 0;
				end
			end

			// Tap Sequence Checker
			begin
				trdy = 0;
				@(posedge clk iff !rst);

				for(int unsigned  r = 0; r < ROUNDS; r++) begin
					repeat(TAP_REP) begin
						automatic dat_t  exp = r;

						while($urandom()%31 == 0) @(posedge clk);

						trdy <= 1;
						@(posedge clk iff tvld);
						assert(tdat == exp) else begin
							$error("Tap output mismatch: %0d instead of %0d.", tdat, exp);
							$stop;
						end
						trdy <= 0;
					end
				end
			end

		join
		$display("Test completed.");
		$finish;
	end

endmodule : stream_tap_tb

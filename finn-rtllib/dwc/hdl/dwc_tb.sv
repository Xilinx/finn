module dwc_tb;

	typedef logic [7:0]  T;
	localparam int unsigned  ICNT = 3;
	localparam int unsigned  OCNT = 5;


	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end


	uwire  irdy;
	logic  ivld;
	T [ICNT-1:0]  idat;

	logic  ordy;
	uwire  ovld;
	uwire T [OCNT-1:0]  odat;

	dwc #(.T(T), .ICNT(ICNT), .OCNT(OCNT)) dut (
		.clk, .rst,
		.irdy, .ivld, .idat,
		.ordy, .ovld, .odat
	);

	T  Q[$];
	initial begin
		ivld =  0;
		idat = 'x;
		@(posedge clk iff !rst);

		ivld <= 1;
		repeat(100 * OCNT) begin
			automatic T [ICNT-1:0]  val;
			std::randomize(val);
			idat <= val;
			@(posedge clk iff irdy);

			for(int unsigned  i = 0; i < ICNT; i++)  Q.push_back(val[i]);
		end
		ivld <=  0;
		idat <= 'x;

		repeat(8) @(posedge clk);
		assert(Q.size == 0) else begin
			$error("Missing output: %p", Q);
			$stop;
		end

		$display("Test completed.");
		$finish;
	end

	initial begin
		ordy = 0;
		@(posedge clk iff !rst);

		forever begin
			ordy <= 1;
			@(posedge clk iff ovld);
			assert(Q.size >= OCNT) else begin
				$error("Spurious output.");
				$stop;
			end
			for(int unsigned  i = 0; i < OCNT; i++) begin
				automatic T  exp = Q.pop_front();
				assert(odat[i] == exp) else begin
					$error("Output mismatch on lane #%0d: %0x instead of %0x", i, odat[i], exp);
					$stop;
				end
			end
			ordy <= 0;
			while($urandom()%7 < 2) @(posedge clk);
		end
	end
endmodule : dwc_tb

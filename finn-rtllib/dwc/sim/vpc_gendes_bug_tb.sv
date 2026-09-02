/******************************************************************************
 * @brief	Minimal reproducer for VPC genDes path bug.
 * @details	Parameters: W=8, N=4, PI=1, PO=4
 *		  GCD=1 → PI0=1, PO0=4, N0=4
 *		  Since PI0=1 and N0 <= PO0, this triggers genDes path.
 *
 *		BUG: First element is lost, last element is duplicated.
 *		  Input:  0x30, 0x31, 0x32, 0x33 (4 serial beats)
 *		  Expect: {0x33, 0x32, 0x31, 0x30}
 *		  Actual: {0x33, 0x33, 0x32, 0x31} ← WRONG!
 *****************************************************************************/
module vpc_gendes_bug_tb;

	localparam int unsigned  W  = 8;   // 8-bit elements
	localparam int unsigned  N  = 4;   // 4 elements per vector
	localparam int unsigned  PI = 1;   // narrow input (1 element/beat)
	localparam int unsigned  PO = 4;   // wide output (4 elements/beat)

	typedef logic [W-1:0]  elem_t;

	logic  clk = 0;
	always #5ns clk = !clk;

	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	uwire  irdy;
	logic  ivld = 0;
	elem_t [PI-1:0]  idat;

	logic  ordy = 0;
	uwire  ovld;
	elem_t [PO-1:0]  odat;

	vpc #(.W(W), .N(N), .PI(PI), .PO(PO)) dut (
		.clk, .rst,
		.idat, .ivld, .irdy,
		.odat, .ovld, .ordy
	);

	int errors = 0;

	initial begin
		$display("=== VPC genDes Bug Reproducer ===");
		$display("Parameters: W=%0d, N=%0d, PI=%0d, PO=%0d", W, N, PI, PO);
		$display("Path: genDes (PI0=1, PO0=4, N0=4 <= PO0)");
		$display("");

		ivld = 0;
		@(posedge clk iff !rst);

		// Send 4 elements serially: 0x30, 0x31, 0x32, 0x33
		$display("Sending elements: 0x30, 0x31, 0x32, 0x33");
		for(int i = 0; i < N; i++) begin
			idat[0] = 8'h30 + i;
			ivld <= 1;
			@(posedge clk iff irdy);
			$display("  Sent element %0d: 0x%02x", i, idat[0]);
		end
		ivld <= 0;

		repeat(10) @(posedge clk);

		$display("");
		if(errors == 0)
			$display("TEST PASSED");
		else
			$display("TEST FAILED with %0d errors", errors);

		$finish;
	end

	initial begin
		ordy = 0;
		@(posedge clk iff !rst);

		ordy <= 1;
		@(posedge clk iff ovld);

		$display("");
		$display("Received: {0x%02x, 0x%02x, 0x%02x, 0x%02x}",
			odat[3], odat[2], odat[1], odat[0]);
		$display("Expected: {0x33, 0x32, 0x31, 0x30}");

		if(odat[0] !== 8'h30) begin
			$error("Lane 0: got 0x%02x, expected 0x30", odat[0]);
			errors++;
		end
		if(odat[1] !== 8'h31) begin
			$error("Lane 1: got 0x%02x, expected 0x31", odat[1]);
			errors++;
		end
		if(odat[2] !== 8'h32) begin
			$error("Lane 2: got 0x%02x, expected 0x32", odat[2]);
			errors++;
		end
		if(odat[3] !== 8'h33) begin
			$error("Lane 3: got 0x%02x, expected 0x33", odat[3]);
			errors++;
		end

		ordy <= 0;
	end

endmodule : vpc_gendes_bug_tb

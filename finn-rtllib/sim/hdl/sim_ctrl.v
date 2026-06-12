module sim_ctrl(input ap_clk, input sim_finish);
	always @(posedge sim_finish) $finish;
	// Workaround for XSI bug: final blocks execute prematurely when all
	// initial blocks complete, rather than at $finish. This never-completing
	// initial block prevents that.
	initial forever #1_000_000_000 ;
endmodule

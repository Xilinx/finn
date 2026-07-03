module sim_ctrl(input ap_clk, input sim_finish, output sim_ctrl_out);
	assign sim_ctrl_out = 1'b0;
`ifdef FINN_SIMULATION
	always @(posedge sim_finish) $finish;
	// Workaround for XSI bug: final blocks execute prematurely when all
	// initial blocks complete, rather than at $finish. This never-completing
	// initial block prevents that.
	initial forever #1;
`endif
endmodule

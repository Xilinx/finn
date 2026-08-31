/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Simulation control triggering $finish upon asserting sim_finish.
 * @author	Shane T. Fleming <shane.fleming@amd.com>
 ***************************************************************************/
module sim_ctrl(input ap_clk, input sim_finish, output sim_ctrl_out);
	// DO NOT REMOVE: This output prevents the module from being empty during
	// synthesis. Vivado's hierarchical checkpoint stitching fails on empty
	// modules (reads the checkpoint twice, causing "Failed to stitch checkpoint"
	// errors). The output is unused but must remain to produce a non-empty netlist.
	assign sim_ctrl_out = 1'b0;
`ifdef FINN_SIMULATION
	initial @(posedge sim_finish) $finish;
	// This ensures there is always a pending #delay in the event queue,
	// preventing the kernel from concluding that the simulation is ending.
	initial forever #1_000_000_000;
`endif
endmodule

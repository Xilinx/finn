/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
	parameter	ACTIVATION_WIDTH = $ACTIVATION_WIDTH$,
	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
	parameter	ACCU_WIDTH = $ACCU_WIDTH$,
	parameter   MW = $MW$,
	parameter   MH = $MH$,
	parameter	TH = $TH$,
	parameter	NARROW_WEIGHTS = $NARROW_WEIGHTS$,
	parameter	SIGNED_ACTIVATIONS = $SIGNED_ACTIVATIONS$,
	parameter	PUMPED_COMPUTE = $PUMPED_COMPUTE$,

	// Safely deducible parameters
	parameter   WSIMD = (PE * SIMD) / TH,
	parameter	WEIGHT_STREAM_WIDTH_BA = (WSIMD * WEIGHT_WIDTH + 7)/8 * 8,
	parameter 	INPUT_STREAM_WIDTH_BA = (SIMD * ACTIVATION_WIDTH + 7) / 8 * 8,
	parameter 	OUTPUT_STREAM_WIDTH_BA = (PE * ACCU_WIDTH + 7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in1_V:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	input   ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Weight Stream
	input	[WEIGHT_STREAM_WIDTH_BA-1:0]  in1_V_TDATA,
	input   in1_V_TVALID,
	output  in1_V_TREADY,
	// Input Stream
	input	[INPUT_STREAM_WIDTH_BA-1:0]  in0_V_TDATA,
	input	in0_V_TVALID,
	output	in0_V_TREADY,
	// Output Stream
	output	[OUTPUT_STREAM_WIDTH_BA-1:0]  out0_V_TDATA,
	output	out0_V_TVALID,
	input	out0_V_TREADY
);

mvu_tiled_axi #(
	.PE(PE), .SIMD(SIMD),
	.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
	.MW(MW), .MH(MH), .TH(TH),
	.NARROW_WEIGHTS(NARROW_WEIGHTS), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .PUMPED_COMPUTE(PUMPED_COMPUTE),
	.FORCE_BEHAVIORAL(0)
	) inst (
	.ap_clk(ap_clk),
	.ap_clk2x(ap_clk2x),
	.ap_rst_n(ap_rst_n),
	.s_axis_weights_tdata(in1_V_TDATA),
	.s_axis_weights_tvalid(in1_V_TVALID),
	.s_axis_weights_tready(in1_V_TREADY),
	.s_axis_input_tdata(in0_V_TDATA),
	.s_axis_input_tvalid(in0_V_TVALID),
	.s_axis_input_tready(in0_V_TREADY),
	.m_axis_output_tdata(out0_V_TDATA),
	.m_axis_output_tvalid(out0_V_TVALID),
	.m_axis_output_tready(out0_V_TREADY)
);

endmodule // $MODULE_NAME_AXI_WRAPPER$

/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FINN wrapper of passthru_axi for IP integration.
 *
 * The clock and reset are not used. They are present so that the IP
 * integrator associates the stream interfaces with the clock domain of the
 * surrounding layers.
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
	parameter AXI_WIDTH = $AXI_WIDTH$
)(
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	output	in0_V_TREADY,
	input	in0_V_TVALID,
	input	[AXI_WIDTH-1:0]  in0_V_TDATA,

	input	out0_V_TREADY,
	output	out0_V_TVALID,
	output	[AXI_WIDTH-1:0]  out0_V_TDATA
);

	passthru_axi #(.DATA_WIDTH(AXI_WIDTH)) core (
		.s_axis_tdata(in0_V_TDATA), .s_axis_tvalid(in0_V_TVALID), .s_axis_tready(in0_V_TREADY),
		.m_axis_tdata(out0_V_TDATA), .m_axis_tvalid(out0_V_TVALID), .m_axis_tready(out0_V_TREADY)
	);

endmodule

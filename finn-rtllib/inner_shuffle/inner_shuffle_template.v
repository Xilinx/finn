/****************************************************************************
* Copyright (C) 2025, Advanced Micro Devices, Inc.
* All rights reserved.
*
* SPDX-License-Identifier: BSD-3-Clause

* @author       Shane T. Fleming <shane.fleming@amd.com>
****************************************************************************/

module $TOP_MODULE_NAME$(
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input ap_rst_n,

// -- AXIS input ------------------
output	in0_V_TREADY,
input	in0_V_TVALID,
input	[$STREAM_BITS$-1:0] in0_V_TDATA,


// -- AXIS output ------------------
input	out0_V_TREADY,
output	out0_V_TVALID,
output	[$STREAM_BITS$-1:0] out0_V_TDATA
);

inner_shuffle #(
	.BITS($WIDTH$),
	.I($I$),
	.J($J$),
	.SIMD($SIMD$)
) impl (
	.clk(ap_clk),
	.rst(!ap_rst_n),
	.irdy(in0_V_TREADY),
	.ivld(in0_V_TVALID),
	.idat(in0_V_TDATA),
	.ordy(out0_V_TREADY),
	.ovld(out0_V_TVALID),
	.odat(out0_V_TDATA)
);

endmodule

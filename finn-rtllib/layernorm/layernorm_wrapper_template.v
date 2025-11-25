/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 ***************************************************************************/

module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

//- AXI Stream - Input -------------------
output	in0_V_TREADY,
input	in0_V_TVALID,
input	[$SIMD$-1:0][31:0]  in0_V_TDATA,

//- AXI Stream - Output ------------------
input	out0_V_TREADY,
output	out0_V_TVALID,
output	[$SIMD$-1:0][31:0]  out0_V_TDATA
);


layernorm #(
 .N($N$),
 .SIMD($SIMD$),
 .FORCE_BEHAVIORAL($FORCE_BEHAVIORAL$)
)
impl
(
 .clk(ap_clk),
 .rst(!ap_rst_n),
 .xdat(in0_V_TDATA),
 .xvld(in0_V_TVALID),
 .xrdy(in0_V_TREADY),
 .ydat(out0_V_TDATA),
 .yvld(out0_V_TVALID),
 .yrdy(out0_V_TREADY)
);

endmodule

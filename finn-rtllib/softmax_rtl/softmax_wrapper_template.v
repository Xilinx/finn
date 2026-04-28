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
input	[$SIMD$*$WIDTH$-1:0]  in0_V_TDATA,

//- AXI Stream - Output ------------------
input	out0_V_TREADY,
output	out0_V_TVALID,
output	[$SIMD$*32-1:0]  out0_V_TDATA
);

	// flat carrier between input-conv stage and softmaxf core
	wire [$SIMD$*32-1:0]  xdat_flat;

	generate
		if ($FP32_PASSTHROUGH$) begin : gen_passthrough
			assign  xdat_flat = in0_V_TDATA;
		end
		else begin : gen_int_conv
			genvar  i;
			for (i = 0; i < $SIMD$; i = i + 1) begin : gen_lane
				int_to_fp32 #(
					.WIDTH($WIDTH$),
					.SIGNED($SIGNED$)
				) u_conv (
					.ival(in0_V_TDATA[(i+1)*$WIDTH$-1 -: $WIDTH$]),
					.fval(xdat_flat[(i+1)*32-1 -: 32])
				);
			end
		end
	endgenerate

	softmaxf #(
		.N($N$),
		.SIMD($SIMD$),
		.NR_ITERS(2),
		.TI_WIDTH(32)
	) impl (
		.clk(ap_clk),
		.rst(!ap_rst_n),
		.xdat(xdat_flat),
		.xvld(in0_V_TVALID),
		.xrdy(in0_V_TREADY),
		.zdat(out0_V_TDATA),
		.zvld(out0_V_TVALID),
		.zrdy(out0_V_TREADY)
	);

endmodule

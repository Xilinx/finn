/****************************************************************************
* Copyright (C) 2025, Advanced Micro Devices, Inc.
* All rights reserved.
*
* SPDX-License-Identifier: BSD-3-Clause

* @author       Shane T. Fleming <shane.fleming@amd.com>
****************************************************************************/

module $TOP_MODULE_NAME$(
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:in1_V:out0_V, ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input ap_rst_n,

// -- AXIS input ------------------
output  in0_V_TREADY,
input   in0_V_TVALID,
input   [$STREAM_BITS$-1:0] in0_V_TDATA,

output  in1_V_TREADY,
input   in1_V_TVALID,
input   [$STREAM_BITS$-1:0] in1_V_TDATA,


// -- AXIS output ------------------
input   out0_V_TREADY,
output  out0_V_TVALID,
output  [$STREAM_BITS$-1:0] out0_V_TDATA
);

eltwisef #(
        .PE($PE$),
        .OP($OP$),
        .B_SCALE($B_SCALE$),
        .FORCE_BEHAVIORAL($FORCE_BEHAVIORAL$)
) impl (
        .clk(ap_clk),
        .rst(!ap_rst_n),
        .adat(in0_V_TDATA),
        .avld(in0_V_TVALID),
        .ardy(in0_V_TREADY),
        .bdat(in1_V_TDATA),
        .bvld(in1_V_TVALID),
        .brdy(in1_V_TREADY),
        .odat(out0_V_TDATA),
        .ovld(out0_V_TVALID),
        .ordy(out0_V_TREADY)
);

endmodule

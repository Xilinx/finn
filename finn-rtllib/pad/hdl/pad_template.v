/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	AXI-Stream top of the pad core, filled in by finn.util.pad.
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
    parameter FOLD_WIDTH = $PE$ * $W$,
    parameter AXI_WIDTH = ((FOLD_WIDTH + 7) / 8) * 8
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    input ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input ap_rst_n,

    output in0_V_TREADY,
    input in0_V_TVALID,
    input [AXI_WIDTH-1:0] in0_V_TDATA,

    input out0_V_TREADY,
    output out0_V_TVALID,
    output [AXI_WIDTH-1:0] out0_V_TDATA
);

    localparam [$TOKENS_W_BITS$-1:0] TOKENS_W = $TOKENS_W$;
    localparam [$TOKENS_E_BITS$-1:0] TOKENS_E = $TOKENS_E$;

    wire [FOLD_WIDTH-1:0] core_out;
    assign out0_V_TDATA = core_out;  // zero-extended to the byte-aligned stream

    pad #(
        .C($C$),
        .PE($PE$),
        .W($W$),
        .IH($IH$),
        .IW($IW$),
        .PAD_N($PAD_N$),
        .PAD_S($PAD_S$),
        .PAD_W($PAD_W$),
        .PAD_E($PAD_E$),
        .EAGER($EAGER$),
        .TOKENS($TOKENS$),
        .TOKENS_W(TOKENS_W),
        .TOKENS_E(TOKENS_E)
    ) impl (
        .clk(ap_clk),
        .rst(!ap_rst_n),
        .irdy(in0_V_TREADY),
        .ivld(in0_V_TVALID),
        .idat(in0_V_TDATA[FOLD_WIDTH-1:0]),
        .ordy(out0_V_TREADY),
        .ovld(out0_V_TVALID),
        .odat(core_out)
    );

endmodule

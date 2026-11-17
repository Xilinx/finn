/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/

`default_nettype wire

module $TOP_MODULE_NAME$ #(
    parameter COND_WIDTH = $COND_WIDTH$,
    parameter X_WIDTH = $X_WIDTH$,
    parameter Y_WIDTH = $Y_WIDTH$,
    parameter OUT_WIDTH = $OUT_WIDTH$,
    parameter COND_AXI_WIDTH = ((COND_WIDTH + 7) / 8) * 8,
    parameter X_AXI_WIDTH = ((X_WIDTH + 7) / 8) * 8,
    parameter Y_AXI_WIDTH = ((Y_WIDTH + 7) / 8) * 8,
    parameter OUT_AXI_WIDTH = ((OUT_WIDTH + 7) / 8) * 8
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:in1_V:in2_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    input ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input ap_rst_n,

    output in0_V_TREADY,
    input in0_V_TVALID,
    input [COND_AXI_WIDTH-1:0] in0_V_TDATA,

    output in1_V_TREADY,
    input in1_V_TVALID,
    input [X_AXI_WIDTH-1:0] in1_V_TDATA,

    output in2_V_TREADY,
    input in2_V_TVALID,
    input [Y_AXI_WIDTH-1:0] in2_V_TDATA,

    input out0_V_TREADY,
    output out0_V_TVALID,
    output [OUT_AXI_WIDTH-1:0] out0_V_TDATA
);

    $TOP_MODULE_NAME$_core #(
        .COND_WIDTH(COND_WIDTH),
        .X_WIDTH(X_WIDTH),
        .Y_WIDTH(Y_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .COND_AXI_WIDTH(COND_AXI_WIDTH),
        .X_AXI_WIDTH(X_AXI_WIDTH),
        .Y_AXI_WIDTH(Y_AXI_WIDTH),
        .OUT_AXI_WIDTH(OUT_AXI_WIDTH)
    ) impl (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .in0_V_TREADY(in0_V_TREADY),
        .in0_V_TVALID(in0_V_TVALID),
        .in0_V_TDATA(in0_V_TDATA),
        .in1_V_TREADY(in1_V_TREADY),
        .in1_V_TVALID(in1_V_TVALID),
        .in1_V_TDATA(in1_V_TDATA),
        .in2_V_TREADY(in2_V_TREADY),
        .in2_V_TVALID(in2_V_TVALID),
        .in2_V_TDATA(in2_V_TDATA),
        .out0_V_TREADY(out0_V_TREADY),
        .out0_V_TVALID(out0_V_TVALID),
        .out0_V_TDATA(out0_V_TDATA)
    );

endmodule

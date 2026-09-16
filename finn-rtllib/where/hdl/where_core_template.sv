/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/

`default_nettype wire

module $TOP_MODULE_NAME$_core #(
    parameter COND_WIDTH = $COND_WIDTH$,
    parameter X_WIDTH = $X_WIDTH$,
    parameter Y_WIDTH = $Y_WIDTH$,
    parameter OUT_WIDTH = $OUT_WIDTH$,
    parameter COND_AXI_WIDTH = ((COND_WIDTH + 7) / 8) * 8,
    parameter X_AXI_WIDTH = ((X_WIDTH + 7) / 8) * 8,
    parameter Y_AXI_WIDTH = ((Y_WIDTH + 7) / 8) * 8,
    parameter OUT_AXI_WIDTH = ((OUT_WIDTH + 7) / 8) * 8
)(
    input ap_clk,
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

    wire [OUT_WIDTH-1:0] core_out;

    assign out0_V_TDATA[OUT_WIDTH-1:0] = core_out;

    generate
        if (OUT_AXI_WIDTH > OUT_WIDTH) begin : gen_pad_tdata
            assign out0_V_TDATA[OUT_AXI_WIDTH-1:OUT_WIDTH] = {(OUT_AXI_WIDTH-OUT_WIDTH){1'b0}};
        end
    endgenerate

    where #(
        .DATA_WIDTH($DATA_WIDTH$),
        .PE($PE$),
        .NDIMS($NDIMS$),
        .OUT_SHAPE($OUT_SHAPE$),
        .COND_SHAPE($COND_SHAPE$),
        .X_SHAPE($X_SHAPE$),
        .Y_SHAPE($Y_SHAPE$),
        .RAM_STYLE($RAM_STYLE$)
    ) impl (
        .clk(ap_clk),
        .rst(!ap_rst_n),
        .cdat(in0_V_TDATA[COND_WIDTH-1:0]),
        .cvld(in0_V_TVALID),
        .crdy(in0_V_TREADY),
        .xdat(in1_V_TDATA[X_WIDTH-1:0]),
        .xvld(in1_V_TVALID),
        .xrdy(in1_V_TREADY),
        .ydat(in2_V_TDATA[Y_WIDTH-1:0]),
        .yvld(in2_V_TVALID),
        .yrdy(in2_V_TREADY),
        .odat(core_out),
        .ovld(out0_V_TVALID),
        .ordy(out0_V_TREADY)
    );

endmodule

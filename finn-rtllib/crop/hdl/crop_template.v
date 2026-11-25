/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
    parameter FOLD_WIDTH = $FOLD_WIDTH$,
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

    wire [FOLD_WIDTH-1:0] core_out;

    assign out0_V_TDATA[FOLD_WIDTH-1:0] = core_out;

    generate
        if (AXI_WIDTH > FOLD_WIDTH) begin : gen_pad_tdata
            assign out0_V_TDATA[AXI_WIDTH-1:FOLD_WIDTH] = {(AXI_WIDTH-FOLD_WIDTH){1'b0}};
        end
    endgenerate

    crop #(
        .H($H$),
        .W($W$),
        .CF($CF$),
        .DATA_WIDTH(FOLD_WIDTH),
        .CROP_N($CROP_N$),
        .CROP_E($CROP_E$),
        .CROP_S($CROP_S$),
        .CROP_W($CROP_W$)
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

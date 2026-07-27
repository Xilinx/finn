/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

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

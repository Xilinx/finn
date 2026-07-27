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
        .COND_NDIMS($COND_NDIMS$),
        .X_NDIMS($X_NDIMS$),
        .Y_NDIMS($Y_NDIMS$),
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

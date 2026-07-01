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

    localparam [$PAD_LEFT_DATA_WIDTH$-1:0] PAD_LEFT_DATA = $PAD_LEFT_DATA$;
    localparam [$PAD_RIGHT_DATA_WIDTH$-1:0] PAD_RIGHT_DATA = $PAD_RIGHT_DATA$;

    wire [FOLD_WIDTH-1:0] core_out;

    assign out0_V_TDATA[FOLD_WIDTH-1:0] = core_out;

    generate
        if (AXI_WIDTH > FOLD_WIDTH) begin : gen_pad_tdata
            assign out0_V_TDATA[AXI_WIDTH-1:FOLD_WIDTH] = {(AXI_WIDTH-FOLD_WIDTH){1'b0}};
        end
    endgenerate

    pad1d #(
        .NUM_TOKENS($NUM_TOKENS$),
        .NUM_CHANNELS($NUM_CHANNELS$),
        .SIMD($SIMD$),
        .ELEM_WIDTH($ELEM_WIDTH$),
        .PAD_LEFT($PAD_LEFT$),
        .PAD_RIGHT($PAD_RIGHT$)
    ) impl (
        .clk(ap_clk),
        .rst(!ap_rst_n),
        .irdy(in0_V_TREADY),
        .ivld(in0_V_TVALID),
        .idat(in0_V_TDATA[FOLD_WIDTH-1:0]),
        .ordy(out0_V_TREADY),
        .ovld(out0_V_TVALID),
        .odat(core_out),
        .pad_left_data(PAD_LEFT_DATA),
        .pad_right_data(PAD_RIGHT_DATA)
    );

endmodule

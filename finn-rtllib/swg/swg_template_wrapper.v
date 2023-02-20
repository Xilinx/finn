/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
`timescale 1 ns / 1 ps

module $TOP_MODULE_NAME$ (
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V" *)
input  ap_clk,
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V" *)
input  ap_rst_n,
input  [BUF_IN_WIDTH-1:0] in0_V_TDATA,
input  in0_V_TVALID,
output in0_V_TREADY,
output [BUF_OUT_WIDTH-1:0] out_V_TDATA,
output out_V_TVALID,
input  out_V_TREADY
);

// top-level parameters (set via code-generation)
parameter BIT_WIDTH = $BIT_WIDTH$;
parameter SIMD = $SIMD$;
parameter MMV_IN = $MMV_IN$;
parameter MMV_OUT = $MMV_OUT$;

// derived constants
parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;

$TOP_MODULE_NAME$_impl
#(
    .BIT_WIDTH(BIT_WIDTH),
    .SIMD(SIMD),
    .MMV_IN(MMV_IN),
    .MMV_OUT(MMV_OUT)
)
impl
(
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .in0_V_V_TDATA(in0_V_TDATA),
    .in0_V_V_TVALID(in0_V_TVALID),
    .in0_V_V_TREADY(in0_V_TREADY),
    .out_V_V_TDATA(out_V_TDATA),
    .out_V_V_TVALID(out_V_TVALID),
    .out_V_V_TREADY(out_V_TREADY)
);

endmodule //TOP_MODULE_NAME

/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
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
    parameter T = $IOSTREAM_TYPE$,
    parameter PERIOD_BITS = $PERIOD_BITS$,
    parameter PERIOD_INIT = $PERIOD_INIT$,
    parameter PERIOD_INIT_UPON_RESET = $PERIOD_INIT_UPON_RESET$
)(
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axilite" *)
    input	ap_clk,
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axilite" *)
    input	ap_rst_n,

    // AXI-lite Configuration
	input	logic  s_axilite_AWVALID,
	output	logic  s_axilite_AWREADY,
	input	logic [0:0]  s_axilite_AWADDR,
	input	logic  s_axilite_WVALID,
	output	logic  s_axilite_WREADY,
	input	logic [PERIOD_BITS - 1:0]  s_axilite_WDATA,
	output	logic  s_axilite_BVALID,
	input	logic  s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	input	logic  s_axilite_ARVALID,
	output	logic  s_axilite_ARREADY,
	input	logic [0:0]  s_axilite_ARADDR,
	output	logic  s_axilite_RVALID,
	input	logic  s_axilite_RREADY,
	output	logic [PERIOD_BITS - 1:0]  s_axilite_RDATA,
	output	logic [1:0]  s_axilite_RRESP,

	// Input Stream without TLAST marker
	input	T  in0_V_TDATA,
	input	logic  in0_V_TVALID,
	output	logic  in0_V_TREADY,

	// Output Stream with TLAST marker
	output	T  out_V_TDATA,
	output	logic  out_V_TVALID,
	input	logic  out_V_TREADY,
	output	logic  out_V_TLAST
);


tlast_marker #(
    .T(T),
    .PERIOD_BITS(PERIOD_BITS),
    .PERIOD_INIT(PERIOD_INIT),
    .PERIOD_INIT_UPON_RESET(PERIOD_INIT_UPON_RESET)
)
$TOP_MODULE_NAME$_impl
(
 .ap_clk(ap_clk),
 .ap_rst_n(ap_rst_n),
 .s_axilite_AWVALID(s_axilite_AWVALID),
 .s_axilite_AWREADY(s_axilite_AWREADY),
 .s_axilite_AWADDR(s_axilite_AWADDR),
 .s_axilite_WVALID(s_axilite_WVALID),
 .s_axilite_WREADY(s_axilite_WREADY),
 .s_axilite_WDATA(s_axilite_WDATA),
 .s_axilite_WSTRB(s_axilite_WSTRB),
 .s_axilite_BVALID(s_axilite_BVALID),
 .s_axilite_BREADY(s_axilite_BREADY),
 .s_axilite_BRESP(s_axilite_BRESP),
 .s_axilite_ARVALID(s_axilite_ARVALID),
 .s_axilite_ARREADY(s_axilite_ARREADY),
 .s_axilite_ARADDR(s_axilite_ARADDR),
 .s_axilite_RVALID(s_axilite_RVALID),
 .s_axilite_RREADY(s_axilite_RREADY),
 .s_axilite_RDATA(s_axilite_RDATA),
 .s_axilite_RRESP(s_axilite_RRESP),
 .src_TREADY(in0_V_TREADY),
 .src_TVALID(in0_V_TVALID),
 .src_TDATA(in0_V_TDATA),
 .dst_TREADY(out_V_TREADY),
 .dst_TVALID(out_V_TVALID),
 .dst_TLAST(out_V_TVLAST),
 .dst_TDATA(out_V_TDATA)
);

endmodule

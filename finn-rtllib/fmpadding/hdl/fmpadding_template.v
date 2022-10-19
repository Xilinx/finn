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

module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axilite" *)
input	ap_clk,
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axilite" *)
input	ap_rst_n,

//- AXI Lite ------------------------
// Writing
input	       s_axilite_AWVALID,
output	       s_axilite_AWREADY,
input	[4:0]  s_axilite_AWADDR,

input	        s_axilite_WVALID,
output	        s_axilite_WREADY,
input	[31:0]  s_axilite_WDATA,
input	[ 3:0]  s_axilite_WSTRB,

output	       s_axilite_BVALID,
input	       s_axilite_BREADY,
output	[1:0]  s_axilite_BRESP,

// Reading
input	       s_axilite_ARVALID,
output	       s_axilite_ARREADY,
input	[4:0]  s_axilite_ARADDR,

output	        s_axilite_RVALID,
input	        s_axilite_RREADY,
output	[31:0]  s_axilite_RDATA,
output	[ 1:0]  s_axilite_RRESP,

//- AXI Stream - Input --------------
output	in0_V_TREADY,
input	in0_V_TVALID,
input	[$STREAM_BITS$-1:0]  in0_V_TDATA,

//- AXI Stream - Output -------------
input	out_V_TREADY,
output	out_V_TVALID,
output	[$STREAM_BITS$-1:0]  out_V_TDATA
);


fmpadding_axi #(
.XCOUNTER_BITS($XCOUNTER_BITS$),
.YCOUNTER_BITS($YCOUNTER_BITS$),
.NUM_CHANNELS($NUM_CHANNELS$),
.SIMD($SIMD$),
.ELEM_BITS($ELEM_BITS$),
.INIT_XON($INIT_XON$),
.INIT_XOFF($INIT_XOFF$),
.INIT_XEND($INIT_XEND$),
.INIT_YON($INIT_YON$),
.INIT_YOFF($INIT_YOFF$),
.INIT_YEND($INIT_YEND$)
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
 .s_axis_tready(in0_V_TREADY),
 .s_axis_tvalid(in0_V_TVALID),
 .s_axis_tdata(in0_V_TDATA),
 .m_axis_tready(out_V_TREADY),
 .m_axis_tvalid(out_V_TVALID),
 .m_axis_tdata(out_V_TDATA)
);

endmodule

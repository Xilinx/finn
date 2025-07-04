/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *	 this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *	 contributors may be used to endorse or promote products derived from
 *	 this software without specific prior written permission.
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
    // top-level parameters (set via code-generation)
    parameter BIT_WIDTH = $BIT_WIDTH$,
    parameter SIMD = $SIMD$,
    parameter MMV_IN = $MMV_IN$,
    parameter MMV_OUT = $MMV_OUT$,
    parameter IN_WIDTH_PADDED = $IN_WIDTH_PADDED$,
    parameter OUT_WIDTH_PADDED = $OUT_WIDTH_PADDED$,

    parameter CNTR_BITWIDTH = $CNTR_BITWIDTH$,
    parameter INCR_BITWIDTH = $INCR_BITWIDTH$,

    // derived constants
    parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN,
    parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT,

    parameter integer C_s_axilite_DATA_WIDTH	= 32,
    parameter integer C_s_axilite_ADDR_WIDTH	= 6
)
(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V:s_axilite, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  ap_rst_n,
    input  [IN_WIDTH_PADDED-1:0] in0_V_TDATA,
    input  in0_V_TVALID,
    output in0_V_TREADY,
    output [OUT_WIDTH_PADDED-1:0] out0_V_TDATA,
    output out0_V_TVALID,
    input  out0_V_TREADY,

    // Ports of Axi Slave Bus Interface s_axilite
    input  [C_s_axilite_ADDR_WIDTH-1 : 0] s_axilite_awaddr,
    input  [2 : 0] s_axilite_awprot,
    input  s_axilite_awvalid,
    output s_axilite_awready,
    input  [C_s_axilite_DATA_WIDTH-1 : 0] s_axilite_wdata,
    input  [(C_s_axilite_DATA_WIDTH/8)-1 : 0] s_axilite_wstrb,
    input  s_axilite_wvalid,
    output s_axilite_wready,
    output [1 : 0] s_axilite_bresp,
    output s_axilite_bvalid,
    input  s_axilite_bready,
    input  [C_s_axilite_ADDR_WIDTH-1 : 0] s_axilite_araddr,
    input  [2 : 0] s_axilite_arprot,
    input  s_axilite_arvalid,
    output s_axilite_arready,
    output [C_s_axilite_DATA_WIDTH-1 : 0] s_axilite_rdata,
    output [1 : 0] s_axilite_rresp,
    output s_axilite_rvalid,
    input  s_axilite_rready
);

wire                     cfg_valid;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_simd;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_kw;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_kh;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_w;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_simd;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_kw;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_kh;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_w;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_w;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_last;
wire [31:0]              cfg_last_read;
wire [31:0]              cfg_last_write;

// Instantiation of Axi Bus Interface s_axilite
$TOP_MODULE_NAME$_axilite # (
    .C_S_AXI_DATA_WIDTH(C_s_axilite_DATA_WIDTH),
    .C_S_AXI_ADDR_WIDTH(C_s_axilite_ADDR_WIDTH)
) axilite_cfg_inst (
    .S_AXI_ACLK(ap_clk),
    .S_AXI_ARESETN(ap_rst_n),
    .S_AXI_AWADDR(s_axilite_awaddr),
    .S_AXI_AWPROT(s_axilite_awprot),
    .S_AXI_AWVALID(s_axilite_awvalid),
    .S_AXI_AWREADY(s_axilite_awready),
    .S_AXI_WDATA(s_axilite_wdata),
    .S_AXI_WSTRB(s_axilite_wstrb),
    .S_AXI_WVALID(s_axilite_wvalid),
    .S_AXI_WREADY(s_axilite_wready),
    .S_AXI_BRESP(s_axilite_bresp),
    .S_AXI_BVALID(s_axilite_bvalid),
    .S_AXI_BREADY(s_axilite_bready),
    .S_AXI_ARADDR(s_axilite_araddr),
    .S_AXI_ARPROT(s_axilite_arprot),
    .S_AXI_ARVALID(s_axilite_arvalid),
    .S_AXI_ARREADY(s_axilite_arready),
    .S_AXI_RDATA(s_axilite_rdata),
    .S_AXI_RRESP(s_axilite_rresp),
    .S_AXI_RVALID(s_axilite_rvalid),
    .S_AXI_RREADY(s_axilite_rready),

    .cfg_reg0(cfg_valid),
    .cfg_reg1(cfg_cntr_simd),
    .cfg_reg2(cfg_cntr_kw),
    .cfg_reg3(cfg_cntr_kh),
    .cfg_reg4(cfg_cntr_w),
    .cfg_reg5(cfg_cntr_h),
    .cfg_reg6(cfg_incr_head_simd),
    .cfg_reg7(cfg_incr_head_kw),
    .cfg_reg8(cfg_incr_head_kh),
    .cfg_reg9(cfg_incr_head_w),
    .cfg_reg10(cfg_incr_head_h),
    .cfg_reg11(cfg_incr_tail_w),
    .cfg_reg12(cfg_incr_tail_h),
    .cfg_reg13(cfg_incr_tail_last),
    .cfg_reg14(cfg_last_read),
    .cfg_reg15(cfg_last_write)
);

$TOP_MODULE_NAME$_impl #(
    .BIT_WIDTH(BIT_WIDTH),
    .SIMD(SIMD),
    .MMV_IN(MMV_IN),
    .MMV_OUT(MMV_OUT),
    .CNTR_BITWIDTH(CNTR_BITWIDTH),
    .INCR_BITWIDTH(INCR_BITWIDTH)
) impl (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .in0_V_V_TDATA(in0_V_TDATA[BUF_IN_WIDTH-1:0]),
    .in0_V_V_TVALID(in0_V_TVALID),
    .in0_V_V_TREADY(in0_V_TREADY),
    .out_V_V_TDATA(out0_V_TDATA[BUF_OUT_WIDTH-1:0]),
    .out_V_V_TVALID(out0_V_TVALID),
    .out_V_V_TREADY(out0_V_TREADY),

    .cfg_valid(cfg_valid),
    .cfg_cntr_simd(cfg_cntr_simd),
    .cfg_cntr_kw(cfg_cntr_kw),
    .cfg_cntr_kh(cfg_cntr_kh),
    .cfg_cntr_w(cfg_cntr_w),
    .cfg_cntr_h(cfg_cntr_h),
    .cfg_incr_head_simd(cfg_incr_head_simd),
    .cfg_incr_head_kw(cfg_incr_head_kw),
    .cfg_incr_head_kh(cfg_incr_head_kh),
    .cfg_incr_head_w(cfg_incr_head_w),
    .cfg_incr_head_h(cfg_incr_head_h),
    .cfg_incr_tail_w(cfg_incr_tail_w),
    .cfg_incr_tail_h(cfg_incr_tail_h),
    .cfg_incr_tail_last(cfg_incr_tail_last),
    .cfg_last_read(cfg_last_read),
    .cfg_last_write(cfg_last_write)
);

if (OUT_WIDTH_PADDED > BUF_OUT_WIDTH) begin
       assign out0_V_TDATA[OUT_WIDTH_PADDED-1:BUF_OUT_WIDTH] = {(OUT_WIDTH_PADDED-BUF_OUT_WIDTH){1'b0}};
end

endmodule // $TOP_MODULE_NAME$

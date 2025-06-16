/******************************************************************************
 * Copyright (C) 2023, Advanced Micro Devices, Inc.
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
	parameter  IBITS = $IBITS$,
	parameter  OBITS = $OBITS$,

	parameter  AXI_IBITS = (IBITS+7)/8 * 8,
	parameter  AXI_OBITS = (OBITS+7)/8 * 8
)(
	//- Global Control ------------------
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	//- AXI Stream - Input --------------
	output	in0_V_TREADY,
	input	in0_V_TVALID,
	input	[AXI_IBITS-1:0]  in0_V_TDATA,

	//- AXI Stream - Output -------------
	input	out0_V_TREADY,
	output	out0_V_TVALID,
	output	[AXI_OBITS-1:0]  out0_V_TDATA
);

	dwc_axi #(
		.IBITS(IBITS),
		.OBITS(OBITS)
	) impl (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),
		.s_axis_tready(in0_V_TREADY),
		.s_axis_tvalid(in0_V_TVALID),
		.s_axis_tdata(in0_V_TDATA),
		.m_axis_tready(out0_V_TREADY),
		.m_axis_tvalid(out0_V_TVALID),
		.m_axis_tdata(out0_V_TDATA)
	);

endmodule

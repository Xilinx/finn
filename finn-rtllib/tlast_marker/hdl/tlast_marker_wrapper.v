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
 *
 * @brief	TLAST marker insertion.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	Inserts a TLAST marker on an AXI-Stream according to a configured period.
 *	Every period-th stream transaction will be annotated with an asserted
 *	TLAST flag on the `dst` output stream. Otherwise, both the `src` and `dst`
 *	AXI-Stream interfaces execute identical transactions.
 *	The initial period setting is determined by the PERIOD_INIT parameter.
 *	If the parameter PERIOD_INIT_UPON_RESER is set, this value will also be
 *	restored by a reset. Otherwise, a reset will not affect the period
 *	setting. The period setting may be changed via the AXI-lite configuration
 *	interface. Any performed write (irrespective of address) will update the
 *	configured period within the limits of the reserved register width of
 *	PERIOD_BITS bits. This setting will take immediate effect only in a clean
 *	state after reset or when the most recent stream transaction had an
 *	asserted `TLAST` flag. Otherwise, the transmission will continue
 *	eventually asserting the `TLAST` flag according to the period setting
 *	at its beginning. The new period setting (or any update performed in the
 *	meantime) will be adopted for the subsequent transmission.
 *	The current period setting can also be read back via the AXI-lite
 *	interface.
 *****************************************************************************/
module tlast_marker_wrapper #(
	parameter DATA_WIDTH  = $DATA_WIDTH$,
	parameter PERIOD_BITS = $PERIOD_BITS$,
	parameter PERIOD_INIT = $PERIOD_INIT$,
	parameter PERIOD_INIT_UPON_RESET = 0
)(
	// Global Control
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:in0_V:out_V, ASSOCIATED_RESET ap_rst_n" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// AXI-lite Configuration
	input	s_axilite_AWVALID,
	output	s_axilite_AWREADY,
	input	[0:0]  s_axilite_AWADDR,
	input	s_axilite_WVALID,
	output	s_axilite_WREADY,
	input	[PERIOD_BITS - 1:0]  s_axilite_WDATA,
	output	s_axilite_BVALID,
	input	s_axilite_BREADY,
	output	[1:0]  s_axilite_BRESP,

	input	s_axilite_ARVALID,
	output	s_axilite_ARREADY,
	input	[0:0]  s_axilite_ARADDR,
	output	s_axilite_RVALID,
	input	s_axilite_RREADY,
	output	[PERIOD_BITS - 1:0]  s_axilite_RDATA,
	output	[1:0]  s_axilite_RRESP,

	// Input Stream without TLAST marker
	input	[DATA_WIDTH-1:0]  in0_V_TDATA,
	input	in0_V_TVALID,
	output	in0_V_TREADY,

	// Output Stream with TLAST marker
	output	[DATA_WIDTH-1:0]  out_V_TDATA,
	output	out_V_TVALID,
	input	out_V_TREADY,
	output	out_V_TLAST
);

	tlast_marker_wrapper #(
		.DATA_WIDTH(DATA_WIDTH),
		.PERIOD_BITS(PERIOD_BITS),
		.PERIOD_INIT(PERIOD_INIT),
		.PERIOD_INIT_UPON_RESET(PERIOD_INIT_UPON_RESET)
	) core (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		// AXI-lite Configuration
		.s_axilite_AWVALID(s_axilite_AWVALID),
		.s_axilite_AWREADY(s_axilite_AWREADY),
		.s_axilite_AWADDR(s_axilite_AWADDR),
		.s_axilite_WVALID(s_axilite_WVALID),
		.s_axilite_WREADY(s_axilite_WREADY),
		.s_axilite_WDATA(s_axilite_WDATA),
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

		// Input Stream without TLAST marker
		.src_TDATA(in0_V_TDATA),
		.src_TVALID(in0_V_TVALID),
		.src_TREADY(in0_V_TREADY),

		// Output Stream with TLAST marker
		.dst_TDATA(out_V_TDATA),
		.dst_TVALID(out_V_TVALID),
		.dst_TREADY(out_V_TREADY),
		.dst_TLAST(out_V_TLAST)
	);

endmodule // tlast_marker_wrapper

/******************************************************************************
 * Copyright (C) 2026, Paderborn University
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
 * @brief	Verilog AXI wrapper for the distributed-arithmetic MVU
 *			(MVAU_rtl with mem_mode=internal_embedded, SIMD=MW, PE=MH).
 * @details
 *	Instantiates the shared flow-control bracket (mvu_da_axi), the generated
 *	alkaid core (via its io wrapper with uniform per-element slots) and the
 *	glue that widens every output slot to FINN's ACCU_WIDTH:
 *	  slot pe holds y[pe] >> OUT_SHIFT_LEFT (two's complement if SIGNED_OUTPUT),
 *	  y[pe] = sext/zext(slot) << OUT_SHIFT_LEFT.
 *****************************************************************************/

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	MW = $MW$,
	parameter	MH = $MH$,
	parameter	ACTIVATION_WIDTH = $ACTIVATION_WIDTH$,
	parameter	ACCU_WIDTH = $ACCU_WIDTH$,
	parameter	SIGNED_OUTPUT = $SIGNED_OUTPUT$,
	parameter	CORE_LATENCY = $CORE_LATENCY$,
	parameter	OUT_SLOT_WIDTH = $OUT_SLOT_WIDTH$,
	parameter	OUT_SHIFT_LEFT = $OUT_SHIFT_LEFT$,

	// Safely deducible parameters
	parameter	INPUT_STREAM_WIDTH_BA = (MW * ACTIVATION_WIDTH + 7) / 8 * 8,
	parameter	OUTPUT_STREAM_WIDTH_BA = (MH * ACCU_WIDTH + 7) / 8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Input Stream
	input	[INPUT_STREAM_WIDTH_BA-1:0]  in0_V_TDATA,
	input	in0_V_TVALID,
	output	in0_V_TREADY,
	// Output Stream
	output	[OUTPUT_STREAM_WIDTH_BA-1:0]  out0_V_TDATA,
	output	out0_V_TVALID,
	input	out0_V_TREADY
);

	wire [MW*ACTIVATION_WIDTH-1:0]  core_inp;
	wire [MH*OUT_SLOT_WIDTH-1:0]    core_out_slots;
	wire [MH*ACCU_WIDTH-1:0]        core_out;

	mvu_da_axi #(
		.INPUT_STREAM_WIDTH(MW * ACTIVATION_WIDTH),
		.OUTPUT_STREAM_WIDTH(MH * ACCU_WIDTH),
		.CORE_LATENCY(CORE_LATENCY),
		.SIGNED_OUTPUT(SIGNED_OUTPUT)
	) inst (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),
		.s_axis_input_tdata(in0_V_TDATA),
		.s_axis_input_tvalid(in0_V_TVALID),
		.s_axis_input_tready(in0_V_TREADY),
		.m_axis_output_tdata(out0_V_TDATA),
		.m_axis_output_tvalid(out0_V_TVALID),
		.m_axis_output_tready(out0_V_TREADY),
		.core_inp(core_inp),
		.core_out(core_out)
	);

	// Generated distributed-arithmetic core (registered input and output)
	$DA_CORE_WRAPPER$ core (
		.clk(ap_clk),
		.model_inp(core_inp),
		.model_out(core_out_slots)
	);

	// Widen every output slot to ACCU_WIDTH
	genvar pe;
	generate
		for(pe = 0; pe < MH; pe = pe + 1) begin : genOutSlot
			wire [OUT_SLOT_WIDTH-1:0]  slot = core_out_slots[pe*OUT_SLOT_WIDTH +: OUT_SLOT_WIDTH];
			wire [ACCU_WIDTH-1:0]      ext;
			if(SIGNED_OUTPUT) begin : genSigned
				assign ext = $signed(slot);
			end
			else begin : genUnsigned
				assign ext = slot;
			end
			assign core_out[pe*ACCU_WIDTH +: ACCU_WIDTH] = ext << OUT_SHIFT_LEFT;
		end
	endgenerate

endmodule // $MODULE_NAME_AXI_WRAPPER$

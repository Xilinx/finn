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
 * @brief	Verilog AXI-lite wrapper for MVU & VVU.
 *****************************************************************************/

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	IS_MVU = $IS_MVU$,
	parameter	COMPUTE_CORE = "$COMPUTE_CORE$",
	parameter	PUMPED_COMPUTE = $PUMPED_COMPUTE$,
	parameter	MW = $MW$,
	parameter	MH = $MH$,
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
	parameter	ACTIVATION_WIDTH = $ACTIVATION_WIDTH$,
	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
	parameter	ACCU_WIDTH = $ACCU_WIDTH$,
        parameter       NARROW_WEIGHTS = $NARROW_WEIGHTS$,
	parameter	SIGNED_ACTIVATIONS = $SIGNED_ACTIVATIONS$,
	parameter	SEGMENTLEN = $SEGMENTLEN$,
	parameter	FORCE_BEHAVIORAL = $FORCE_BEHAVIORAL$,

	// Safely deducible parameters
	parameter	WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
	parameter 	INPUT_STREAM_WIDTH_BA = ((IS_MVU == 1 ? 1 : PE) * SIMD * ACTIVATION_WIDTH + 7) / 8 * 8,
	parameter 	OUTPUT_STREAM_WIDTH_BA = (PE*ACCU_WIDTH + 7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in1_V:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	input   ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Weight Stream
	input	[WEIGHT_STREAM_WIDTH_BA-1:0]  in1_V_TDATA,
	input   in1_V_TVALID,
	output  in1_V_TREADY,
	// Input Stream
	input	[INPUT_STREAM_WIDTH_BA-1:0]  in0_V_TDATA,
	input	in0_V_TVALID,
	output	in0_V_TREADY,
	// Output Stream
	output	[OUTPUT_STREAM_WIDTH_BA-1:0]  out0_V_TDATA,
	output	out0_V_TVALID,
	input	out0_V_TREADY
);

mvu_vvu_axi #(
	.IS_MVU(IS_MVU), .COMPUTE_CORE(COMPUTE_CORE), .PUMPED_COMPUTE(PUMPED_COMPUTE), .MW(MW), .MH(MH), .PE(PE), .SIMD(SIMD),
	.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH), .NARROW_WEIGHTS(NARROW_WEIGHTS),
	.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
	) inst (
	.ap_clk(ap_clk),
	.ap_clk2x(ap_clk2x),
	.ap_rst_n(ap_rst_n),
	.s_axis_weights_tdata(in1_V_TDATA),
	.s_axis_weights_tvalid(in1_V_TVALID),
	.s_axis_weights_tready(in1_V_TREADY),
	.s_axis_input_tdata(in0_V_TDATA),
	.s_axis_input_tvalid(in0_V_TVALID),
	.s_axis_input_tready(in0_V_TREADY),
	.m_axis_output_tdata(out0_V_TDATA),
	.m_axis_output_tvalid(out0_V_TVALID),
	.m_axis_output_tready(out0_V_TREADY)
);

endmodule // $MODULE_NAME_AXI_WRAPPER$

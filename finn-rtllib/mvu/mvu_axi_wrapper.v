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
 *
 * @brief	Verilog AXI-lite wrapper for MVU.
 *****************************************************************************/

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter 	MW = $MW$,
	parameter	MH = $MH$,
	parameter 	PE = $PE$,
	parameter 	SIMD = $SIMD$,
	parameter 	ACTIVATION_WIDTH = $ACTIVATION_WIDTH$,
	parameter 	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
	parameter 	ACCU_WIDTH = $ACCU_WIDTH$,
	parameter 	SIGNED_ACTIVATIONS = $SIGNED_ACTIVATIONS$,
	parameter 	SEGMENTLEN = $SEGMENTLEN$,
	parameter 	RAM_STYLE = "$IBUF_RAM_STYLE$",

	// Safely deducible parameters
	parameter 	WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
	parameter 	INPUT_STREAM_WIDTH_BA = (SIMD*ACTIVATION_WIDTH+7)/8 * 8,
	parameter 	OUTPUT_LANES = PE,
	parameter 	OUTPUT_STREAM_WIDTH_BA = (OUTPUT_LANES*ACCU_WIDTH + 7)/8 * 8
)(
  	// Global Control
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_weights_tdata,
	input	logic  s_axis_weights_tvalid,
	output	logic  s_axis_weights_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata,
	input	logic  s_axis_input_tvalid,
	output	logic  s_axis_input_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready
);

mvu_axi #(
	.MW(MW), .MH(MH), .PE(PE), .SIMD(SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
	.WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
	.SEGMENTLEN(SEGMENTLEN), .RAM_STYLE(RAM_STYLE)
	) inst (
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),
	.s_axis_weights_tdata(s_axis_weights_tdata),
	.s_axis_weights_tvalid(s_axis_weights_tvalid),
	.s_axis_weights_tready(s_axis_weights_tready),
	.s_axis_input_tdata(s_axis_input_tdata),
	.s_axis_input_tvalid(s_axis_input_tvalid),
	.s_axis_input_tready(s_axis_input_tready),
	.m_axis_output_tdata(m_axis_output_tdata),
	.m_axis_output_tvalid(m_axis_output_tvalid),
	.m_axis_output_tready(m_axis_output_tready)
);

endmodule : $MODULE_NAME_AXI_WRAPPER$
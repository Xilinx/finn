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
 *****************************************************************************/

module en_global #(
	int unsigned PE,
	int unsigned CLEN,
	int unsigned CU_SIMD = 3,

	int unsigned WEIGHT_WIDTH,
	int unsigned ACTIVATION_WIDTH,

	int unsigned N_DCPL_STAGES = 2
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,
	input   logic  en,

	// Activation Stream
	input	logic  s_act_tvalid,
	output	logic  s_act_tready,
	input	logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0]  s_act_tdata,
	input   logic [CLEN-1:0] s_act_tlast,

	output	logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0]  m_act_tdata,
	output  logic [CLEN-1:0] m_act_tlast,

	// Weight Stream
	input	logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0]  s_wgt_tdata,
	input	logic  s_wgt_tvalid,
	output	logic  s_wgt_tready,

	output	logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0]  m_wgt_tdata,

	output  logic m_tvalid
);

// Global enable
// ---------------------------------------------------------------------
logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] act_tdata;
logic [CLEN-1:0] act_tlast;
logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0] wgt_tdata;
logic ovld;
logic ordy;

assign ovld = en && s_act_tvalid && s_wgt_tvalid;
assign s_act_tready = en && s_wgt_tvalid;
assign s_wgt_tready = en && s_act_tvalid;


assign act_tdata = ovld ? s_act_tdata : '0;
assign act_tlast = ovld ? s_act_tlast : '0;
assign wgt_tdata = ovld ? s_wgt_tdata : '0;


// Output
// ---------------------------------------------------------------------
skid #(.DATA_WIDTH(PE*CU_SIMD*WEIGHT_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg_weights (
	.clk(clk), .rst(rst),
	.idat(wgt_tdata), .ivld(ovld), .irdy(),
	.odat(m_wgt_tdata), .ovld(), .ordy(1'b1)
);

skid #(.DATA_WIDTH(CLEN*CU_SIMD*ACTIVATION_WIDTH+CLEN), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg_activations (
	.clk(clk), .rst(rst),
	.idat({act_tlast, act_tdata}), .ivld(ovld), .irdy(ordy),
	.odat({m_act_tlast, m_act_tdata}), .ovld(m_tvalid), .ordy(1'b1)
);

endmodule
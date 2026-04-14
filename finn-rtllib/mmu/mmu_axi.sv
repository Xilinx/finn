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

module mmu_axi #(
	string GEMM_TYPE = "mmau",
	int unsigned PE,
	int unsigned SIMD,
	int unsigned CU_SIMD = 3,

	int unsigned MW,
	int unsigned MH,
	int unsigned N_VECTORS,

	int unsigned WEIGHT_WIDTH,
	int unsigned ACTIVATION_WIDTH,
	int unsigned ACCU_WIDTH,

	int unsigned IN_TILED = 0,
	int unsigned OUT_TILED = 0,

	int unsigned DSP_STAGES = 3,
	bit SIGNED_ACTIVATIONS = 1,
	bit PUMPED_COMPUTE = 0, // Not used
	bit FORCE_BEHAVIOURAL = 0,

	int unsigned N_DCPL_STAGES = 2,
    
	// Safely deducible parameters
	localparam int unsigned  CLEN = (SIMD + CU_SIMD-1)/ CU_SIMD,
	localparam int unsigned  WSIMD = PE * CU_SIMD,
	localparam int unsigned  ASIMD = CLEN * CU_SIMD,
	
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = WSIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8
)(
	// Global Control
	input	logic  ap_clk,
	input   logic  ap_clk2x,
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

// Checks and params
// ---------------------------------------------------------------------
	initial begin
		if (SIMD != CLEN * CU_SIMD) begin
			$error("%m: SIMD (%0d) should be a multiple of CU_SIMD and CLEN. (TODO: Needs testing)", SIMD);
			$finish;
		end
		if (MW % SIMD != 0) begin
			$error("%m: MW (%0d) is not a multiple of SIMD (%0d).", MW, SIMD);
			$finish;
		end
		if (MH % PE != 0) begin
			$error("%m: MH (%0d) is not a multiple of PE (%0d).", MH, PE);
			$finish;
		end
		if (WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if (ACTIVATION_WIDTH > 8) begin
			$error("Activation width of %0d-bits exceeds maximum of 8-bits", ACTIVATION_WIDTH);
			$finish;
		end
	end

	localparam int unsigned SF = MW / SIMD;
	localparam int unsigned NF = MH / PE;
	localparam int unsigned N_TRS_OP = SF * NF * N_VECTORS;
    localparam int unsigned N_TRS_EP = (GEMM_TYPE == "mmau_1d") ? CLEN-1 + CLEN-1 + DSP_STAGES + 2 :
                                                                  CLEN-1 + CLEN-1 + DSP_STAGES + PE;
    
// Input replay
// ---------------------------------------------------------------------
	logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] adat_s0;
	logic [ASIMD-1:0][ACTIVATION_WIDTH-1:0] adat_s0_wd;
	logic alast_s0;
	logic avld_s0, ardy_s0;

	logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] act_s0_tdata;
	logic [ASIMD-1:0][ACTIVATION_WIDTH-1:0] act_s0_tdata_mod;
	logic act_s0_tlast;
	logic act_s0_tvalid, act_s0_tready;

	replay_buff_mmau #(.XC(SF), .YC(CLEN), .W(SIMD*ACTIVATION_WIDTH), .N_REPS(NF), .IO_TILED(IN_TILED)) activation_replay (
		.clk(ap_clk), .rst(~ap_rst_n),
		.ivld(s_axis_input_tvalid), .irdy(s_axis_input_tready), .idat(s_axis_input_tdata),
		.ovld(act_s0_tvalid), .ordy(act_s0_tready), .odat(act_s0_tdata), .olast(act_s0_tlast)
	);

    if (ASIMD > SIMD) 
        assign act_s0_tdata_mod[ASIMD-1:SIMD] = '0;
	assign act_s0_tdata_mod[SIMD-1:0] = act_s0_tdata[SIMD-1:0];

// Activation scheduling
// ---------------------------------------------------------------------
	logic [ASIMD-1:0][ACTIVATION_WIDTH-1:0] act_s1_tdata;
	logic [CLEN-1:0] act_s1_tlast;
	logic act_s1_tvalid, act_s1_tready;

	sched_activations #(
        .CU_SIMD(CU_SIMD), .CLEN(CLEN),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .N_BEATS_OP(N_TRS_OP), .N_BEATS_EP(N_TRS_EP)
    ) inst_sched_act (
        .clk(ap_clk), .rst(~ap_rst_n),
        .s_axis_tdata(act_s0_tdata_mod), .s_axis_tvalid(act_s0_tvalid), .s_axis_tready(act_s0_tready), .s_axis_tlast(act_s0_tlast),
        .m_axis_tdata(act_s1_tdata), .m_axis_tvalid(act_s1_tvalid), .m_axis_tready(act_s1_tready), .m_axis_tlast(act_s1_tlast)
    );

// Weight scheduling
// ---------------------------------------------------------------------
	logic [WSIMD-1:0][WEIGHT_WIDTH-1:0] wgt_s1_tdata;
	logic wgt_s1_tvalid, wgt_s1_tready;

if(GEMM_TYPE == "mmau_1d") begin
	sched_weights_1d #(
        .CU_SIMD(CU_SIMD), .PE(PE),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .N_BEATS_OP(N_TRS_OP), .N_BEATS_EP(N_TRS_EP)
    ) inst_sched_wgt (
        .clk(ap_clk), .rst(~ap_rst_n),
        .s_axis_tdata(s_axis_weights_tdata), .s_axis_tvalid(s_axis_weights_tvalid), .s_axis_tready(s_axis_weights_tready),
        .m_axis_tdata(wgt_s1_tdata), .m_axis_tvalid(wgt_s1_tvalid), .m_axis_tready(wgt_s1_tready)
    );
end else begin
	sched_weights_2d #(
        .CU_SIMD(CU_SIMD), .PE(PE),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .N_BEATS_OP(N_TRS_OP), .N_BEATS_EP(N_TRS_EP)
    ) inst_sched_wgt (
        .clk(ap_clk), .rst(~ap_rst_n),
        .s_axis_tdata(s_axis_weights_tdata), .s_axis_tvalid(s_axis_weights_tvalid), .s_axis_tready(s_axis_weights_tready),
        .m_axis_tdata(wgt_s1_tdata), .m_axis_tvalid(wgt_s1_tvalid), .m_axis_tready(wgt_s1_tready)
    );
end


// Global enable
// ---------------------------------------------------------------------
	logic en;
    logic [ASIMD-1:0][ACTIVATION_WIDTH-1:0] act_s2_tdata;
	logic [CLEN-1:0] act_s2_tlast;
    logic [WSIMD-1:0][WEIGHT_WIDTH-1:0] wgt_s2_tdata;
    logic s2_tvalid;

    en_global #(
        .PE(PE), .CLEN(CLEN), .CU_SIMD(CU_SIMD),
        .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) inst_en_global (
        .clk(ap_clk), .rst(~ap_rst_n),
        .en(en),
        .s_act_tvalid(act_s1_tvalid), .s_act_tready(act_s1_tready), .s_act_tdata(act_s1_tdata), .s_act_tlast(act_s1_tlast),
        .m_act_tdata(act_s2_tdata), .m_act_tlast(act_s2_tlast),
        .s_wgt_tvalid(wgt_s1_tvalid), .s_wgt_tready(wgt_s1_tready), .s_wgt_tdata(wgt_s1_tdata), 
        .m_wgt_tdata(wgt_s2_tdata),
        .m_tvalid(s2_tvalid)
    );

// CU
// ---------------------------------------------------------------------
	logic p_tvalid, p_tready;
    logic [PE-1:0][ACCU_WIDTH-1:0] p_tdata;

if(GEMM_TYPE == "mmau_1d") begin
	cu_mmau_1d #(
        .PE(PE), .CLEN(CLEN), .CU_SIMD(CU_SIMD),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
        .FORCE_BEHAVIOURAL(FORCE_BEHAVIOURAL)
    ) inst_cu_mmau (
        .clk(ap_clk), .rst(~ap_rst_n),
		.en(en),
        .ivld(s2_tvalid), .a(act_s2_tdata), .ilast(act_s2_tlast), .w(wgt_s2_tdata),
        .m_axis_tvalid(p_tvalid), .m_axis_tready(p_tready), .m_axis_tdata(p_tdata)
    );
end else begin
	cu_mmau_2d #(
        .PE(PE), .CLEN(CLEN), .CU_SIMD(CU_SIMD),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
        .FORCE_BEHAVIOURAL(FORCE_BEHAVIOURAL)
    ) inst_cu_mmau (
        .clk(ap_clk), .rst(~ap_rst_n),
		.en(en),
        .ivld(s2_tvalid), .a(act_s2_tdata), .ilast(act_s2_tlast), .w(wgt_s2_tdata),
        .m_axis_tvalid(p_tvalid), .m_axis_tready(p_tready), .m_axis_tdata(p_tdata)
    );
end
    

// Reorder
// ---------------------------------------------------------------------
	if(OUT_TILED == 0) begin
		reorder_out #(.W(OUTPUT_STREAM_WIDTH_BA), .XC(NF), .YC(CLEN)) inst_reorder_out (
			.clk(ap_clk), .rst(~ap_rst_n),
			.ivld(p_tvalid), .irdy(p_tready), .idat(p_tdata),
			.ovld(m_axis_output_tvalid), .ordy(m_axis_output_tready), .odat(m_axis_output_tdata)
		);
	end else begin
		assign m_axis_output_tvalid = p_tvalid;
		assign p_tready = m_axis_output_tready;
		assign m_axis_output_tdata = p_tdata;
	end

endmodule
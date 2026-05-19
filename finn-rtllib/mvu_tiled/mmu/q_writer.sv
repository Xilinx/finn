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

module q_writer #(
	int unsigned CU_SIMD,
	int unsigned CLEN,
	int unsigned ACTIVATION_WIDTH
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	input	logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] s_axis_tdata,
    input   logic s_axis_tlast,
	input	logic s_axis_tvalid,
	output	logic s_axis_tready,

	// Output Stream
	output	logic [CU_SIMD-1:0][ACTIVATION_WIDTH-1:0]  m_axis_tdata,
	output  logic m_axis_tlast,
	output	logic m_axis_tvalid,
	input	logic m_axis_tready
);

// Params
// ---------------------------------------------------------------------
    localparam integer CLEN_BITS = (CLEN == 1) ? 1 : $clog2(CLEN);

// Skid
// ---------------------------------------------------------------------
    logic axis_s0_tvalid, axis_s0_tready;
    logic axis_s0_tlast;
    logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] axis_s0_tdata;

    skid #(.DATA_WIDTH(CLEN*CU_SIMD*ACTIVATION_WIDTH + 1), .FEED_STAGES(1)) inst_reg (
		.clk(clk), .rst(rst),
		.idat({s_axis_tlast, s_axis_tdata}), .ivld(s_axis_tvalid), .irdy(s_axis_tready),
		.odat({axis_s0_tlast, axis_s0_tdata}), .ovld(axis_s0_tvalid), .ordy(axis_s0_tready)
	);

// PtoS
// ---------------------------------------------------------------------
    logic [CLEN_BITS-1:0] wr_ptr_C = '0, wr_ptr_N;

    logic axis_s1_tvalid, axis_s1_tready;
    logic axis_s1_tlast;
    logic [CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] axis_s1_tdata;

    // REG
	always_ff @(posedge clk) begin
		if(rst) begin
            wr_ptr_C <= 0;
		end else begin
            wr_ptr_C <= wr_ptr_N;
		end
	end

    // DP
    always_comb begin
        wr_ptr_N = wr_ptr_C;

        axis_s0_tready = 1'b0;
        axis_s1_tvalid = 1'b0;
        axis_s1_tdata = axis_s0_tdata[wr_ptr_C];
        axis_s1_tlast = 1'b0;

        if(axis_s0_tvalid) begin
            axis_s1_tvalid = 1'b1;

            if(axis_s1_tready) begin
                if(wr_ptr_C == CLEN-1) begin
                    wr_ptr_N = 0;
                    axis_s0_tready = 1'b1;
                    axis_s1_tlast = axis_s0_tlast;
                end else begin
                    wr_ptr_N = wr_ptr_C + 1;
                end
            end
        end
    end

// Queue
// ---------------------------------------------------------------------
    Q_srl #(
            .depth(CLEN),
            .width(CU_SIMD*ACTIVATION_WIDTH+1)
        ) inst_queue (
            .clock(clk), .reset(rst),
            .count(), .maxcount(),
            .i_v(axis_s1_tvalid), .i_r(axis_s1_tready), .i_d({axis_s1_tlast, axis_s1_tdata}),
            .o_v(m_axis_tvalid), .o_r(m_axis_tready), .o_d({m_axis_tlast, m_axis_tdata})
        );

endmodule

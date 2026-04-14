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

module sched_weights_1d #(
    int unsigned CU_SIMD,
    int unsigned PE,
    int unsigned WEIGHT_WIDTH,

    int unsigned N_BEATS_OP,
	int unsigned N_BEATS_EP,

	int unsigned N_DCPL_STAGES = 2
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	input	logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0] s_axis_tdata,
	input	logic s_axis_tvalid,
	output	logic s_axis_tready,

	// Output Stream
	output	logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0]  m_axis_tdata,
	output	logic  m_axis_tvalid,
	input	logic  m_axis_tready
);

// Params
// ---------------------------------------------------------------------
	localparam integer CNT_EPLG_BITS = (N_BEATS_OP > N_BEATS_EP) ?
		(N_BEATS_OP == 1) ? 1 : $clog2(N_BEATS_OP) :
		(N_BEATS_EP == 1) ? 1 : $clog2(N_BEATS_EP);

// Queueing
// ---------------------------------------------------------------------
    logic s_out_tready, s_out_tvalid;
    logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0] s_out_tdata;

    skid #(.DATA_WIDTH(PE*CU_SIMD*WEIGHT_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_ireg (
        .clk(clk), .rst(rst),
        .idat(s_axis_tdata), .ivld(s_axis_tvalid), .irdy(s_axis_tready),
        .odat(s_out_tdata), .ovld(s_out_tvalid), .ordy(s_out_tready)
    );

// Shifting
// ---------------------------------------------------------------------
    logic valid_C = '0, valid_N;
    logic eplg_C = '0, eplg_N;
    logic [CNT_EPLG_BITS-1:0] cnt_eplg_C = '0, cnt_eplg_N;

    logic ovld, ordy;
    logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0] odat;

    // REG
    always_ff @(posedge clk) begin
        if(rst) begin
            valid_C <= '1;
            eplg_C <= 1'b0;
            cnt_eplg_C <= '0;
        end else begin
            valid_C <= valid_N;
            eplg_C <= eplg_N;
            cnt_eplg_C <= cnt_eplg_N;
        end
    end

    // DP
    always_comb begin
        valid_N = valid_C;
        eplg_N = eplg_C;
        cnt_eplg_N = cnt_eplg_C;

        // Read
        if (ovld && ordy) begin
            // Shift ctrl
            if(eplg_C) begin
                if(cnt_eplg_C == N_BEATS_EP-1) begin
                    eplg_N = 1'b0;
                    cnt_eplg_N = 0;
                    valid_N = 1'b1;
                end else begin
                    cnt_eplg_N = cnt_eplg_C + 1;
                    valid_N = 1'b0;
                end
            end else begin
                if(cnt_eplg_C == N_BEATS_OP-1) begin
                    eplg_N = 1'b1;
                    cnt_eplg_N = 0;
                    valid_N = 1'b0;
                end else begin
                    cnt_eplg_N = cnt_eplg_C + 1;
                    valid_N = 1'b1;
                end
            end
        end
    end

    // Output valid
    assign ovld = !((s_out_tvalid && valid_C) != valid_C);
    assign s_out_tready = (ovld && ordy) && valid_C;
    assign odat = valid_C ? s_out_tdata : '0;

// Oreg
// ---------------------------------------------------------------------
    skid #(.DATA_WIDTH(PE*CU_SIMD*WEIGHT_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg (
        .clk(clk), .rst(rst),
        .idat(odat), .ivld(ovld), .irdy(ordy),
        .odat(m_axis_tdata), .ovld(m_axis_tvalid), .ordy(m_axis_tready)
    );

endmodule
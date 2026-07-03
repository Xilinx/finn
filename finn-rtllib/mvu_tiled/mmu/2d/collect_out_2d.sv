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

module collect_out_2d #(
    int unsigned PE,
    int unsigned ACCU_WIDTH,

	int unsigned QDEPTH = 2 * PE,
	int unsigned QCNT_BITS = $clog2(QDEPTH),
	int unsigned Q_MAX = PE,

	int unsigned N_DCPL_STAGES = 2
)(
	// Global Control
	input  logic clk,
	input  logic rst,

    output logic en,

	// Input Stream
	input  logic [PE-1:0][ACCU_WIDTH-1:0] p_tdata,
	input  logic [PE-1:0] p_tvalid,

	// Output Stream
	output logic [PE-1:0][ACCU_WIDTH-1:0]  m_axis_tdata,
	output logic m_axis_tvalid,
	input  logic m_axis_tready
);

// Queueing
// ---------------------------------------------------------------------
	logic [PE-1:0] q_in_tready, q_in_tvalid;
	logic [PE-1:0][ACCU_WIDTH-1:0] q_in_tdata;
	logic [PE-1:0] q_out_tready, q_out_tvalid;
	logic [PE-1:0][ACCU_WIDTH-1:0] q_out_tdata;
	logic [PE-1:0][QCNT_BITS-1:0] q_count;
	logic en_int;

	assign q_in_tvalid = p_tvalid;
	assign q_in_tdata = p_tdata;

	for(genvar i = 0; i < PE; i++) begin
		Q_srl #(
			.depth(QDEPTH),
			.width(ACCU_WIDTH)
		) inst_queue (
			.clock(clk), .reset(rst),
			.count(q_count[i]), .maxcount(),
			.i_v(q_in_tvalid[i]), .i_r(q_in_tready[i]), .i_d(q_in_tdata[i]),
			.o_v(q_out_tvalid[i]), .o_r(q_out_tready[i]), .o_d(q_out_tdata[i])
		);
	end

	// Global enable
	always_comb begin
		en_int = 1'b1;

		for(int i = 0; i < PE; i++) begin
			if(q_count[i] > Q_MAX)
				en_int = 1'b0;
		end
	end

	always_ff @( posedge clk ) begin
		if(rst) begin
			en <= 1'b0;
		end
		else begin
			en <= en_int;
		end
	end

// Output
// ---------------------------------------------------------------------
	logic ovld;
	logic ordy;
	logic [PE-1:0][ACCU_WIDTH-1:0] odat;

	assign odat = q_out_tdata;
	assign ovld = &q_out_tvalid;
	for(genvar i = 0; i < PE; i++) begin
		assign q_out_tready[i] = ovld && ordy;
	end

	skid #(.DATA_WIDTH(PE*ACCU_WIDTH), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg (
		.clk(clk), .rst(rst),
		.idat(odat), .ivld(ovld), .irdy(ordy),
		.odat(m_axis_tdata), .ovld(m_axis_tvalid), .ordy(m_axis_tready)
	);

endmodule

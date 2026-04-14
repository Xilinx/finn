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

module sched_activations #(
	int unsigned CU_SIMD,
	int unsigned CLEN,
	int unsigned ACTIVATION_WIDTH,

    int unsigned N_BEATS_OP,
	int unsigned N_BEATS_EP,

	int unsigned N_DCPL_STAGES = 2
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
	output	logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0]  m_axis_tdata,
	output  logic [CLEN-1:0] m_axis_tlast,
	output	logic m_axis_tvalid,
	input	logic m_axis_tready
);

// Params
// ---------------------------------------------------------------------
	localparam integer CLEN_BITS = (CLEN == 1) ? 1 : $clog2(CLEN);
	localparam integer CNT_EPLG_BITS = (N_BEATS_OP > N_BEATS_EP) ?
		(N_BEATS_OP == 1) ? 1 : $clog2(N_BEATS_OP) :
		(N_BEATS_EP == 1) ? 1 : $clog2(N_BEATS_EP);

// Shifting
// ---------------------------------------------------------------------
	logic [CLEN_BITS-1:0] wr_ptr_C = '0, wr_ptr_N;

	logic [CLEN-1:0] valid_C = '0, valid_N;
	logic eplg_C = '0, eplg_N;
	logic [CNT_EPLG_BITS-1:0] cnt_eplg_C = '0, cnt_eplg_N;

    logic [CLEN-1:0] q_in_tvalid, q_in_tready;
    logic [CLEN-1:0] q_in_tlast;
    logic [CLEN-1:0][CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] q_in_tdata;

    logic [CLEN-1:0] q_out_tvalid, q_out_tready;
    logic [CLEN-1:0] q_out_tlast;
    logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] q_out_tdata;

    logic ovld, ordy;
	logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] odat;
	logic [CLEN-1:0] olast;


    for(genvar i = 0; i < CLEN; i++) begin
        q_writer #(
            .CLEN(CLEN), .CU_SIMD(CU_SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
        ) inst_queue_writer (
            .clk(clk), .rst(rst),
            .s_axis_tvalid(q_in_tvalid[i]), .s_axis_tready(q_in_tready[i]), .s_axis_tdata(q_in_tdata[i]), .s_axis_tlast(q_in_tlast[i]),
            .m_axis_tvalid(q_out_tvalid[i]), .m_axis_tready(q_out_tready[i]), .m_axis_tdata(q_out_tdata[i]), .m_axis_tlast(q_out_tlast[i])
        );
    end

	// REG
	always_ff @(posedge clk) begin
		if(rst) begin
			wr_ptr_C <= '0;
			valid_C[0] <= 1'b1;
			valid_C[CLEN-1:1] <= '0;
			eplg_C <= '0;
			cnt_eplg_C <= '0;
		end else begin
			wr_ptr_C <= wr_ptr_N;
			valid_C <= valid_N;
			eplg_C <= eplg_N;
			cnt_eplg_C <= cnt_eplg_N;
		end
	end

	// DP
	always_comb begin
        // Read
		valid_N = valid_C;
		eplg_N = eplg_C;
		cnt_eplg_N = cnt_eplg_C;

        q_out_tready = '0;

		// Read
		if (ovld && ordy) begin
            // Read from queue
            for(int i = 0; i < CLEN; i++) begin
               q_out_tready[i] = valid_C[i]; 
            end

			// Shift ctrl
			valid_N[CLEN-1:1] = valid_C[CLEN-2:0];
			if(eplg_C) begin
				if(cnt_eplg_C == N_BEATS_EP-1) begin
					eplg_N = 1'b0;
					cnt_eplg_N = 0;
					valid_N[0] = 1'b1;
				end else begin
					cnt_eplg_N = cnt_eplg_C + 1;
					valid_N[0] = 1'b0;
				end
			end else begin
				if(cnt_eplg_C == N_BEATS_OP-1) begin
					eplg_N = 1'b1;
					cnt_eplg_N = 0;
					valid_N[0] = 1'b0;
				end else begin
					cnt_eplg_N = cnt_eplg_C + 1;
					valid_N[0] = 1'b1;
				end
			end
		end

		// Write
        wr_ptr_N = wr_ptr_C;

		s_axis_tready = 1'b0;
        q_in_tvalid = '0;
        for(int i = 0; i < CLEN; i++) begin
            q_in_tdata[i] = s_axis_tdata;
            q_in_tlast[i] = s_axis_tlast;
        end

		if(s_axis_tvalid) begin
            q_in_tvalid[wr_ptr_C] = 1'b1;

            if(q_in_tready[wr_ptr_C]) begin
                s_axis_tready = 1'b1;
                wr_ptr_N = (wr_ptr_C == CLEN-1) ? 0 : wr_ptr_C + 1;
            end
		end

	end

	// Output valid
	always_comb begin
		ovld = 1'b1;

		for(int i = 0; i < CLEN; i++) begin
			if((valid_C[i] & q_out_tvalid[i]) != valid_C[i]) begin
				ovld = 1'b0;
			end
		end
	end

	for(genvar i = 0; i < CLEN; i++) begin
		assign odat[i] = valid_C[i] ? q_out_tdata[i] : '0;
		assign olast[i] = valid_C[i] ? q_out_tlast[i] : '0;
	end

// Oreg
// ---------------------------------------------------------------------
	skid #(.DATA_WIDTH(CLEN*(CU_SIMD*ACTIVATION_WIDTH + 1)), .FEED_STAGES(N_DCPL_STAGES)) inst_oreg (
		.clk(clk), .rst(rst),
		.idat({olast, odat}), .ivld(ovld), .irdy(ordy),
		.odat({m_axis_tlast, m_axis_tdata}), .ovld(m_axis_tvalid), .ordy(m_axis_tready)
	);


endmodule
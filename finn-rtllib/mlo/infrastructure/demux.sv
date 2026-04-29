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

module demux #(
    int unsigned                        N_LAYERS,
    int unsigned                        IDX_BITS,
    int unsigned                        FM_SIZE,

    int unsigned                        OLEN_BITS,

    int unsigned                        QDEPTH = 8,
    int unsigned                        N_DCPL_STGS = 1
) (
    input  logic                         aclk,
    input  logic                         aresetn,

    input  logic                         s_idx_tvalid,
    output logic                         s_idx_tready,
    input  logic [IDX_BITS-1:0]          s_idx_tdata,

    output logic                         m_idx_tvalid,
    input  logic                         m_idx_tready,
    output logic [IDX_BITS-1:0]          m_idx_tdata,

    input  logic                         s_axis_tvalid,
    output logic                         s_axis_tready,
    input  logic [OLEN_BITS-1:0]         s_axis_tdata,

    output logic                         m_axis_if_tvalid,
    input  logic                         m_axis_if_tready,
    output logic [OLEN_BITS-1:0]         m_axis_if_tdata,

    output logic                         m_axis_se_tvalid,
    input  logic                         m_axis_se_tready,
    output logic [OLEN_BITS-1:0]         m_axis_se_tdata
);

localparam int unsigned FM_BEATS = FM_SIZE / (OLEN_BITS/8);
localparam int unsigned FM_BEATS_BITS = (FM_BEATS == 1) ? 1 : $clog2(FM_BEATS);

//
// Demux Ctrl
//

typedef enum logic[1:0] {ST_CTRL_IDLE, ST_CTRL_SEND_SE, ST_CTRL_SEND_IF} state_ctrl_t;
state_ctrl_t state_ctrl_C = ST_CTRL_IDLE, state_ctrl_N;

logic val_idx_C = '0, val_idx_N;
logic val_seq_C = '0, val_seq_N;
logic [IDX_BITS-1:0] idx_C = '0, idx_N;
logic seq_C = '0, seq_N;

logic seq_tready;
logic seq_out_tvalid, seq_out_tready;
logic seq_out_tdata;

always_ff @(posedge aclk) begin: REG_CTRL
    if(~aresetn) begin
        state_ctrl_C <= ST_CTRL_IDLE;

        val_idx_C <= '0;
        val_seq_C <= '0;
        idx_C <= 'X;
        seq_C <= 'X;
    end else begin
        state_ctrl_C <= state_ctrl_N;

        val_idx_C <= val_idx_N;
        val_seq_C <= val_seq_N;
        idx_C <= idx_N;
        seq_C <= seq_N;
    end
end

always_comb begin: NSL_CTRL
    state_ctrl_N = state_ctrl_C;

    case (state_ctrl_C)
        ST_CTRL_IDLE:
            state_ctrl_N = s_idx_tvalid ? ((s_idx_tdata == N_LAYERS-1) ? ST_CTRL_SEND_SE : ST_CTRL_SEND_IF) : ST_CTRL_IDLE;

        ST_CTRL_SEND_SE:
            state_ctrl_N = !val_seq_C ? ST_CTRL_IDLE : ST_CTRL_SEND_SE;

        ST_CTRL_SEND_IF:
            state_ctrl_N = (!val_idx_C && !val_seq_C) ? ST_CTRL_IDLE : ST_CTRL_SEND_IF;
    endcase
end

always_comb begin: DP_CTRL
    val_idx_N = val_idx_C;
    val_seq_N = val_seq_C;
    idx_N = idx_C;
    seq_N = seq_C;

    s_idx_tready = 1'b0;

    case (state_ctrl_C)
        ST_CTRL_IDLE: begin
            if(s_idx_tvalid) begin
                s_idx_tready = 1'b1;
                if(s_idx_tdata == N_LAYERS-1) begin
                    val_seq_N = 1'b1;
                    seq_N = 1'b0;
                end
                else begin
                    val_idx_N = 1'b1;
                    val_seq_N = 1'b1;

                    idx_N = s_idx_tdata;
                    seq_N = 1'b1;
                end
            end
        end

        ST_CTRL_SEND_SE: begin
            val_seq_N = seq_tready ? 1'b0 : val_seq_C;
        end

        ST_CTRL_SEND_IF: begin
            val_idx_N = m_idx_tready ? 1'b0 : val_idx_C;
            val_seq_N = seq_tready ? 1'b0 : val_seq_C;
        end
    endcase
end

Q_srl #(
    .depth(QDEPTH), .width(1)
) inst_queue_seq (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(seq_C), .i_v(val_seq_C), .i_r(seq_tready),
    .o_d(seq_out_tdata), .o_v(seq_out_tvalid), .o_r(seq_out_tready)
);

assign m_idx_tvalid = val_idx_C;
assign m_idx_tdata = idx_C;

//
// Data
//

// Regs
typedef enum logic[1:0] {ST_DATA_IDLE, ST_DATA_MUX_SE, ST_DATA_MUX_IF} state_data_t;
state_data_t state_data_C = ST_DATA_IDLE, state_data_N;

logic [FM_BEATS_BITS-1:0] cnt_data_C = '0, cnt_data_N;

logic s_axis_int_tvalid, s_axis_int_tready;
logic [OLEN_BITS-1:0] s_axis_int_tdata;

always_ff @( posedge aclk ) begin : REG_DATA
    if(~aresetn) begin
        state_data_C <= ST_DATA_IDLE;
        cnt_data_C <= 'X;
    end
    else begin
        state_data_C <= state_data_N;
        cnt_data_C <= cnt_data_N;
    end
end

always_comb begin : NSL_DATA
    state_data_N = state_data_C;

    case (state_data_C)
        ST_DATA_IDLE:
            state_data_N = seq_out_tvalid ? (seq_out_tdata ? ST_DATA_MUX_IF : ST_DATA_MUX_SE) : ST_DATA_IDLE;

        ST_DATA_MUX_SE:
            state_data_N = (s_axis_int_tvalid && s_axis_int_tready && (cnt_data_C == FM_BEATS-1)) ? ST_DATA_IDLE : ST_DATA_MUX_SE;

        ST_DATA_MUX_IF:
            state_data_N = (s_axis_int_tvalid && s_axis_int_tready && (cnt_data_C == FM_BEATS-1)) ? ST_DATA_IDLE : ST_DATA_MUX_IF;

    endcase
end

always_comb begin : DP_DATA
    cnt_data_N = cnt_data_C;

    // S
    seq_out_tready = 1'b0;

    s_axis_int_tready = 1'b0;

    m_axis_se_tvalid = 1'b0;
    m_axis_se_tdata = s_axis_int_tdata;
    m_axis_if_tvalid = 1'b0;
    m_axis_if_tdata = s_axis_int_tdata;

    // RD
    case (state_data_C)
        ST_DATA_IDLE: begin
            seq_out_tready = 1'b1;
            cnt_data_N = 0;
        end

        ST_DATA_MUX_SE: begin
            m_axis_se_tvalid = s_axis_int_tvalid;
            s_axis_int_tready = m_axis_se_tready;
            m_axis_se_tdata = s_axis_int_tdata;

            if(s_axis_int_tvalid & s_axis_int_tready) begin
                cnt_data_N = cnt_data_C + 1;
            end
        end

        ST_DATA_MUX_IF: begin
            m_axis_if_tvalid = s_axis_int_tvalid;
            s_axis_int_tready = m_axis_if_tready;
            m_axis_if_tdata = s_axis_int_tdata;

            if(s_axis_int_tvalid & s_axis_int_tready) begin
                cnt_data_N = cnt_data_C + 1;
            end
        end

    endcase
end

// REG
skid #(.FEED_STAGES(N_DCPL_STGS), .DATA_WIDTH(OLEN_BITS)) inst_reg (
    .clk(aclk),
    .rst(~aresetn),
    .ivld(s_axis_tvalid),
    .irdy(s_axis_tready),
    .idat(s_axis_tdata),
    .ovld(s_axis_int_tvalid),
    .ordy(s_axis_int_tready),
    .odat(s_axis_int_tdata)
);

endmodule

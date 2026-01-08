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

module mux #(
    int unsigned              IDX_BITS,
    int unsigned              FM_SIZE,

    int unsigned              ILEN_BITS,

    int unsigned              QDEPTH = 32,
    int unsigned              N_DCPL_STGS = 1
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    // Index Coming From Intermediate Frame Buffer
    input  logic                        s_idx_tvalid,
    output logic                        s_idx_tready,
    input  logic [IDX_BITS-1:0]         s_idx_tdata,

    // Index To StreamTap
    output logic                        m_idx_tvalid,
    input  logic                        m_idx_tready,
    output logic [IDX_BITS-1:0]         m_idx_tdata,

    // Input Activation Data
    input  logic                        s_axis_fs_tvalid,
    output logic                        s_axis_fs_tready,
    input  logic [ILEN_BITS-1:0]        s_axis_fs_tdata,

    // Activation Data From Intermediate Frame Buffer
    input  logic                        s_axis_if_tvalid,
    output logic                        s_axis_if_tready,
    input  logic [ILEN_BITS-1:0]        s_axis_if_tdata,

    // Output Activation Data to Data Path
    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready,
    output logic [ILEN_BITS-1:0]        m_axis_tdata
);

localparam int unsigned FM_BEATS = FM_SIZE / (ILEN_BITS/8);
localparam int unsigned FM_BEATS_BITS = (FM_BEATS == 1) ? 1 : $clog2(FM_BEATS);

//
// Generate idx from data
//

typedef enum logic[0:0] {ST_GEN_IDLE, ST_GEN_DATA} state_gen_t;
state_gen_t state_gen_C = ST_GEN_IDLE, state_gen_N;

logic [FM_BEATS_BITS-1:0] cnt_gen_C = '0, cnt_gen_N;

logic axis_fs_tvalid, axis_fs_tready;
logic [ILEN_BITS-1:0] axis_fs_tdata;
logic axis_fs_tvalid_q, axis_fs_tready_q;
logic [ILEN_BITS-1:0] axis_fs_tdata_q;

logic idx_fs_tvalid, idx_fs_tready;

always_ff @(posedge aclk) begin: REG_GEN
    if(~aresetn) begin
        state_gen_C <= ST_GEN_IDLE;
        cnt_gen_C <= 'X;
    end else begin
        state_gen_C <= state_gen_N;
        cnt_gen_C <= cnt_gen_N;
    end
end

always_comb begin: NSL_GEN
    state_gen_N = state_gen_C;

    case (state_gen_C)
        ST_GEN_IDLE:
            state_gen_N = (s_axis_fs_tvalid && idx_fs_tready) ? ST_GEN_DATA : ST_GEN_IDLE;

        ST_GEN_DATA:
            state_gen_N = (s_axis_fs_tvalid && s_axis_fs_tready && (cnt_gen_C == FM_BEATS-1)) ? ST_GEN_IDLE : ST_GEN_DATA;

    endcase
end

always_comb begin: DP_GEN
    cnt_gen_N = cnt_gen_C;

    axis_fs_tvalid = 1'b0;
    axis_fs_tdata = s_axis_fs_tdata;
    s_axis_fs_tready = 1'b0;

    idx_fs_tvalid = 1'b0;

    case (state_gen_C)
        ST_GEN_IDLE: begin
            if(s_axis_fs_tvalid) begin
                idx_fs_tvalid = 1'b1;
            end
            cnt_gen_N = 0;
        end

        ST_GEN_DATA: begin
            axis_fs_tvalid = s_axis_fs_tvalid;
            s_axis_fs_tready = axis_fs_tready;

            if(s_axis_fs_tvalid && s_axis_fs_tready) begin
                cnt_gen_N = cnt_gen_C + 1;
            end
        end

    endcase
end

Q_srl #(
    .depth(2), .width(ILEN_BITS)
) inst_queue_gend (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(axis_fs_tdata), .i_v(axis_fs_tvalid), .i_r(axis_fs_tready),
    .o_d(axis_fs_tdata_q), .o_v(axis_fs_tvalid_q), .o_r(axis_fs_tready_q)
);

//
// Mux control
//

typedef enum logic[0:0] {ST_CTRL_IDLE, ST_CTRL_SEND} state_ctrl_t;
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
            state_ctrl_N = idx_fs_tvalid ? ST_CTRL_SEND : (s_idx_tvalid ? ST_CTRL_SEND : ST_CTRL_IDLE);

        ST_CTRL_SEND:
            state_ctrl_N = (!val_idx_C && !val_seq_C) ? ST_CTRL_IDLE : ST_CTRL_SEND;
    endcase
end

always_comb begin: DP_CTRL
    val_idx_N = val_idx_C;
    val_seq_N = val_seq_C;
    idx_N = idx_C;
    seq_N = seq_C;

    idx_fs_tready = 1'b0;
    s_idx_tready = 1'b0;

    case (state_ctrl_C)
        ST_CTRL_IDLE: begin
            if(idx_fs_tvalid) begin
                idx_fs_tready = 1'b1;

                val_idx_N = 1'b1;
                val_seq_N = 1'b1;

                idx_N = '0;
                seq_N = 1'b0;
            end
            else if(s_idx_tvalid) begin
                s_idx_tready = 1'b1;

                val_idx_N = 1'b1;
                val_seq_N = 1'b1;

                idx_N = s_idx_tdata;
                seq_N = 1'b1;
            end
        end

        ST_CTRL_SEND: begin
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
// Mux data
//

// Regs
typedef enum logic[1:0] {ST_DATA_IDLE, ST_DATA_MUX_FS, ST_DATA_MUX_IF} state_data_t;
state_data_t state_data_C = ST_DATA_IDLE, state_data_N;

logic [FM_BEATS_BITS-1:0] cnt_data_C = '0, cnt_data_N;

logic m_axis_int_tvalid, m_axis_int_tready;
logic [ILEN_BITS-1:0] m_axis_int_tdata;

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
            state_data_N = seq_out_tvalid ? (seq_out_tdata ? ST_DATA_MUX_IF : ST_DATA_MUX_FS) : ST_DATA_IDLE;

        ST_DATA_MUX_FS:
            state_data_N = (m_axis_int_tvalid && m_axis_int_tready && (cnt_data_C == FM_BEATS-1)) ? ST_DATA_IDLE : ST_DATA_MUX_FS;

        ST_DATA_MUX_IF:
            state_data_N = (m_axis_int_tvalid && m_axis_int_tready && (cnt_data_C == FM_BEATS-1)) ? ST_DATA_IDLE : ST_DATA_MUX_IF;

    endcase
end

always_comb begin : DP_DATA
    cnt_data_N = cnt_data_C;

    // S
    seq_out_tready = 1'b0;

    m_axis_int_tvalid = 1'b0;
    m_axis_int_tdata = '0;

    axis_fs_tready_q = 1'b0;
    s_axis_if_tready = 1'b0;

    // RD
    case (state_data_C)
        ST_DATA_IDLE: begin
            seq_out_tready = 1'b1;
            cnt_data_N = 0;
        end

        ST_DATA_MUX_FS: begin
            m_axis_int_tvalid = axis_fs_tvalid_q;
            axis_fs_tready_q = m_axis_int_tready;
            m_axis_int_tdata = axis_fs_tdata_q;

            if(m_axis_int_tvalid & m_axis_int_tready) begin
                cnt_data_N = cnt_data_C + 1;
            end
        end

        ST_DATA_MUX_IF: begin
            m_axis_int_tvalid = s_axis_if_tvalid;
            s_axis_if_tready = m_axis_int_tready;
            m_axis_int_tdata = s_axis_if_tdata;

            if(m_axis_int_tvalid & m_axis_int_tready) begin
                cnt_data_N = cnt_data_C + 1;
            end
        end

    endcase
end

// REG
skid #(.FEED_STAGES(N_DCPL_STGS), .DATA_WIDTH(ILEN_BITS)) inst_reg (
    .clk(aclk),
    .rst(~aresetn),
    .ivld(m_axis_int_tvalid),
    .irdy(m_axis_int_tready),
    .idat(m_axis_int_tdata),
    .ovld(m_axis_tvalid),
    .ordy(m_axis_tready),
    .odat(m_axis_tdata)
);

endmodule

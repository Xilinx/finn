// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

module mux_out #(
    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              CNT_BITS = 16,

    parameter int unsigned              OLEN_BITS = 32,
    parameter int unsigned              BEAT_SHIFT = $clog2(OLEN_BITS/8),

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              DBG = 0
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    input  logic [CNT_BITS-1:0]         n_layers,

    AXI4S.slave                         s_idx,
    AXI4S.master                        m_idx_se,
    AXI4S.master                        m_idx_if,

    AXI4S.slave                         s_axis,
    AXI4S.master                        m_axis_se,
    AXI4S.master                        m_axis_if
);

//
// Ctrl
//

AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_idx_out ();

queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_idx (.aclk(aclk), .aresetn(aresetn), .s_axis(s_idx), .m_axis(q_idx_out));

AXI4S #(.AXI4S_DATA_BITS(CNT_BITS+LEN_BITS+1)) seq ();
AXI4S #(.AXI4S_DATA_BITS(CNT_BITS+LEN_BITS+1)) seq_out ();

queue #(.QDEPTH(QDEPTH), .QWIDTH(CNT_BITS+LEN_BITS+1)) inst_queue_seq (.aclk(aclk), .aresetn(aresetn), .s_axis(seq), .m_axis(seq_out));

always_comb begin
    q_idx_out.tready = 1'b0;

    m_idx_se.tvalid = 1'b0;
    m_idx_if.tvalid = 1'b0;
    seq.tvalid = 1'b0;

    m_idx_se.tdata = q_idx_out.tdata;
    m_idx_if.tdata = q_idx_out.tdata;
    seq.tdata = '0;

    if(q_idx_out.tvalid) begin
        if(q_idx_out.tdata[0+:CNT_BITS] == n_layers - 1) begin
            if(m_idx_se.tready && seq.tready) begin
                q_idx_out.tready = 1'b1;
                m_idx_se.tvalid = 1'b1;
                seq.tvalid = 1'b1;

                seq.tdata = {1'b0, q_idx_out.tdata[CNT_BITS+:CNT_BITS+LEN_BITS]};
            end
        end
        else begin
            if(m_idx_if.tready && seq.tready) begin
                q_idx_out.tready = 1'b1;
                m_idx_if.tvalid = 1'b1;
                seq.tvalid = 1'b1;

                seq.tdata = {1'b1, q_idx_out.tdata[CNT_BITS+:CNT_BITS+LEN_BITS]};
            end
        end
    end
end

//
// Data
//

// Regs
typedef enum logic[1:0] {ST_IDLE, ST_MUX_SE, ST_MUX_IF} state_t;
state_t state_C = ST_IDLE, state_N;

logic [CNT_BITS-1:0] cnt_frames_C = '0, cnt_frames_N;
logic [CNT_BITS-1:0] n_frames_C = '0, n_frames_N;
logic [LEN_BITS-1:0] len_C = '0, len_N;
logic [LEN_BITS-1:0] cnt_C = '0, cnt_N;

AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) s_axis_int ();

logic [31:0] cnt_out;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        cnt_out <= 0;
    end
    else begin
        cnt_out <= (s_axis.tvalid & s_axis.tready) ? cnt_out + 1 : cnt_out;
    end
end

always_ff @( posedge aclk ) begin : REG
    if(~aresetn) begin
        state_C <= ST_IDLE;

        cnt_frames_C <= 'X;
        n_frames_C <= 'X;
        len_C <= 'X;
        cnt_C <= 'X;
    end
    else begin
        state_C <= state_N;

        cnt_frames_C <= cnt_frames_N;
        n_frames_C <= n_frames_N;
        len_C <= len_N;
        cnt_C <= cnt_N;
    end
end

always_comb begin : NSL
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = seq_out.tvalid ? (seq_out.tdata[CNT_BITS+LEN_BITS+:1] ? ST_MUX_IF : ST_MUX_SE) : ST_IDLE;

        ST_MUX_SE:
            state_N = ((cnt_frames_C == n_frames_C - 1) && (cnt_C == len_C - 1) && (s_axis_int.tvalid & s_axis_int.tready)) ? ST_IDLE : ST_MUX_SE;

        ST_MUX_IF:
            state_N = ((cnt_frames_C == n_frames_C - 1) && (cnt_C == len_C - 1) && (s_axis_int.tvalid & s_axis_int.tready)) ? ST_IDLE : ST_MUX_IF;

    endcase
end

always_comb begin : DP
    // AL
    cnt_frames_N = cnt_frames_C;
    n_frames_N = n_frames_C;
    len_N = len_C;
    cnt_N = cnt_C;

    // S
    seq_out.tready = 1'b0;

    s_axis_int.tready = 1'b0;
    m_axis_se.tvalid = 1'b0;
    m_axis_se.tdata = s_axis_int.tdata;
    m_axis_if.tvalid = 1'b0;
    m_axis_if.tdata = s_axis_int.tdata;

    // RD
    case (state_C)
        ST_IDLE: begin
            seq_out.tready = 1'b1;
            if(seq_out.tvalid) begin
                cnt_frames_N = 0;
                cnt_N = 0;
                n_frames_N = seq_out.tdata[0+:CNT_BITS];
                len_N = seq_out.tdata[CNT_BITS+:LEN_BITS] >> BEAT_SHIFT;
            end
        end

        ST_MUX_SE: begin
            s_axis_int.tready = m_axis_se.tready;
            m_axis_se.tvalid = s_axis_int.tvalid;

            if(s_axis_int.tvalid & s_axis_int.tready) begin
                if(cnt_C == len_C - 1) begin
                    cnt_N = 0;
                    cnt_frames_N = cnt_frames_C + 1;
                end
                else begin
                    cnt_N = cnt_C + 1;
                end
            end
        end

        ST_MUX_IF: begin
            s_axis_int.tready = m_axis_if.tready;
            m_axis_if.tvalid = s_axis_int.tvalid;

            if(s_axis_int.tvalid & s_axis_int.tready) begin
                if(cnt_C == len_C - 1) begin
                    cnt_N = 0;
                    cnt_frames_N = cnt_frames_C + 1;
                end
                else begin
                    cnt_N = cnt_C + 1;
                end
            end
        end

    endcase
end

// REG
axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(OLEN_BITS)) inst_reg (.aclk(aclk), .aresetn(aresetn), .s_axis(s_axis), .m_axis(s_axis_int));

//
// DBG
//

if(DBG == 1) begin
    // ila_mux_out inst_ila_mux_out (
    //     .clk(aclk),
    //     .probe0(q_idx_out.tvalid),
    //     .probe1(q_idx_out.tready),
    //     .probe2(m_idx_se.tvalid),
    //     .probe3(m_idx_se.tready),
    //     .probe4(m_idx_if.tvalid),
    //     .probe5(m_idx_if.tready),
    //     .probe6(seq.tvalid),
    //     .probe7(seq.tready),
    //     .probe8(state_C), // 2
    //     .probe9(cnt_frames_C), // 16
    //     .probe10(n_frames_C), // 16
    //     .probe11(len_C), // 32
    //     .probe12(cnt_C), // 16
    //     .probe13(s_axis.tvalid),
    //     .probe14(s_axis.tready),
    //     .probe15(m_axis_se.tvalid),
    //     .probe16(m_axis_se.tready),
    //     .probe17(m_axis_if.tvalid),
    //     .probe18(m_axis_if.tready),
    //     .probe19(cnt_out) // 32
    // );
end


endmodule

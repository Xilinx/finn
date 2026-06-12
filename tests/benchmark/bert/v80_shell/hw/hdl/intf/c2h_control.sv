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

module c2h_control #(
    parameter integer                   MAX_OUTSTANDING = 16,
    parameter integer                   PCKT_SIZE = (1024 / (QDMA_DATA_BITS/8))
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    AXI4S_USER.slave                    s_axis,

    AXI4S_USER.master                   m_axis,
    AXI4S.master                        m_cmpt
);

// Packetizer
AXI4S_USER #(.AXI4S_DATA_BITS(QDMA_DATA_BITS), .AXI4S_USER_BITS(QDMA_QID_BITS)) axis_pckt ();
AXI4S #(.AXI4S_DATA_BITS(QDMA_DESC_BITS)) q_desc ();

c2h_packetizer #(
    .MAX_OUTSTANDING(MAX_OUTSTANDING),
    .PCKT_SIZE(PCKT_SIZE)
) inst_c2h_packetizer (
    .aclk(aclk), .aresetn(aresetn),
    .s_axis(s_axis),
    .m_axis(axis_pckt),
    .m_desc(q_desc)
);

// Completion
localparam integer BEATS_BITS = $clog2(QDMA_DATA_BITS / 8);

AXI4S #(.AXI4S_DATA_BITS(QDMA_PCK_BITS)) q_cmpt ();

queue #(.QDEPTH(MAX_OUTSTANDING), .QWIDTH(QDMA_PCK_BITS)) inst_cmpt_q (.aclk(aclk), .aresetn(aresetn), .s_axis(q_cmpt), .m_axis(m_cmpt));

typedef enum logic[0:0] {ST_IDLE, ST_SEND} state_t;
state_t state_C = ST_IDLE, state_N;

logic [QDMA_LEN_BITS-1:0] t_size_C = '0, t_size_N;
logic [QDMA_LEN_BITS-1:0] t_beats_C = '0, t_beats_N;
logic [QDMA_QID_BITS-1:0] t_qid_C = '0, t_qid_N;
logic [QDMA_LEN_BITS-1:0] cnt_C = '0, cnt_N;
logic [QDMA_PCK_BITS-1:0] cnt_pck_C = 1, cnt_pck_N;

always_ff @( posedge aclk ) begin: REG
    if(~aresetn) begin
        state_C <= ST_IDLE;
        
        t_size_C <= 'X;
        t_beats_C <= 'X;
        t_qid_C <= 'X;
        cnt_C <= 0;
        cnt_pck_C <= 1;
    end
    else begin
        state_C <= state_N;

        t_size_C <= t_size_N;
        t_beats_C <= t_beats_N;
        t_qid_C <= t_qid_N;
        cnt_C <= cnt_N;
        cnt_pck_C <= cnt_pck_N;
    end
end

always_comb begin: NSL
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = (q_desc.tvalid && q_cmpt.tready) ? ST_SEND : ST_IDLE;

        ST_SEND:
            state_N = (axis_pckt.tvalid && axis_pckt.tready && (cnt_C == t_beats_C-1)) ? ST_IDLE : ST_SEND;
    
    endcase
end

always_comb begin: DP
    t_size_N = t_size_C;
    t_beats_N = t_beats_C;
    t_qid_N = t_qid_C;
    cnt_N = cnt_C;
    cnt_pck_N = cnt_pck_C;

    // Input
    q_desc.tready = 1'b0;
    
    axis_pckt.tready = 1'b0;

    // Output
    m_axis.tvalid = 1'b0;
    m_axis.tdata = axis_pckt.tdata;
    m_axis.tuser = {{QDMA_MTY_BITS{1'b0}}, t_size_C, t_qid_C};
    m_axis.tlast = 1'b0;
    m_axis.tkeep = '1;

    q_cmpt.tvalid = 1'b0;
    q_cmpt.tdata = cnt_pck_C;

    case (state_C)
        ST_IDLE: begin
            if(q_desc.tvalid) begin
                q_cmpt.tvalid = 1'b1;

                if(q_cmpt.tready) begin
                    q_desc.tready = 1'b1;

                    t_qid_N = q_desc.tdata[0+:QDMA_QID_BITS];
                    t_size_N = q_desc.tdata[QDMA_QID_BITS+:QDMA_LEN_BITS];
                    t_beats_N = (q_desc.tdata[QDMA_QID_BITS+:QDMA_LEN_BITS] + (QDMA_LEN_BITS/8-1)) >> BEATS_BITS;
                end
            end
        end

        ST_SEND: begin
            axis_pckt.tready = m_axis.tready;
            m_axis.tvalid = axis_pckt.tvalid;

            if(axis_pckt.tvalid & axis_pckt.tready) begin
                if(cnt_C == t_beats_C-1) begin
                    cnt_N <= 0;
                    cnt_pck_N = cnt_pck_C + 1;
                end
                else begin
                    cnt_N <= cnt_C + 1;
                end
            end
        end 
 
    endcase
end


//
// DEBUG
//

/*
ila_c2h inst_ila_c2h (
    .clk(aclk),
    .probe0(m_axis.tvalid),
    .probe1(m_axis.tready),
    .probe2(m_axis.tdata), // 512
    .probe3(m_axis.tlast),
    .probe4(m_axis.tuser), // 34
    .probe5(s_axis.tvalid), 
    .probe6(s_axis.tready), 
    .probe7(q_desc.tvalid),
    .probe8(q_desc.tready),
    .probe9(q_desc.tdata[QDMA_QID_BITS-1:0]), // 12
    .probe10(q_desc.tdata[QDMA_LEN_BITS+QDMA_QID_BITS-1:QDMA_QID_BITS]), // 16
    .probe11(cnt_pck_C), // 16
    .probe12(state_C),
    .probe13(t_size_C), // 16
    .probe14(t_beats_C), // 16
    .probe15(t_qid_C), // 12
    .probe16(cnt_C), // 16
    .probe17(m_cmpt.tvalid),
    .probe18(m_cmpt.tready),
    .probe19(m_cmpt.tdata) // 16
);
*/

endmodule 
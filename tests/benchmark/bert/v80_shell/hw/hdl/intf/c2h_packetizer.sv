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

module c2h_packetizer #(
    parameter integer                   MAX_OUTSTANDING = 16,
    parameter integer                   PCKT_SIZE = (4096 / (QDMA_DATA_BITS/8))
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    AXI4S_USER.slave                    s_axis,
    AXI4S_USER.master                   m_axis,
    AXI4S.master                        m_desc
);

localparam integer PCKT_BITS = $clog2(PCKT_SIZE);

AXI4S #(.AXI4S_DATA_BITS(QDMA_LEN_BITS+QDMA_QID_BITS)) q_desc ();
queue #(.QDEPTH(MAX_OUTSTANDING), .QWIDTH(QDMA_DESC_BITS)) inst_desc_q (.aclk(aclk), .aresetn(aresetn), .s_axis(q_desc), .m_axis(m_desc));

logic q_axis_tready;
logic q_axis_tvalid;

axis_pckt_fifo inst_pckt_fifo (
    .s_axis_aclk(aclk),
    .s_axis_aresetn(aresetn),
    .s_axis_tvalid(q_axis_tvalid),
    .s_axis_tready(q_axis_tready),
    .s_axis_tdata(s_axis.tdata),
    .s_axis_tlast(s_axis.tlast),
    .s_axis_tkeep(s_axis.tkeep),
    .s_axis_tuser(s_axis.tuser),
    .m_axis_tvalid(m_axis.tvalid),
    .m_axis_tready(m_axis.tready),
    .m_axis_tdata(m_axis.tdata),
    .m_axis_tlast(m_axis.tlast),
    .m_axis_tkeep(m_axis.tkeep),
    .m_axis_tuser(m_axis.tuser)
);

typedef enum logic[0:0] {ST_CNT, ST_SEND} state_t;
state_t state_C = ST_CNT, state_N;

logic [PCKT_BITS:0] cnt_C = '0, cnt_N;
logic [QDMA_QID_BITS-1:0] qid_C = '0, qid_N;

always_ff @( posedge aclk ) begin: REG
    if(~aresetn) begin
        state_C <= ST_CNT;
        
        qid_C <= 'X;
        cnt_C <= 0;
    end
    else begin
        state_C <= state_N;

        qid_C <= qid_N;
        cnt_C <= cnt_N;
    end
end

always_comb begin: NSL
    state_N = state_C;

    case (state_C)
        ST_CNT:
            state_N = (s_axis.tvalid && s_axis.tready) ? (s_axis.tlast ? ST_SEND : ((cnt_C == PCKT_SIZE-1) ? ST_SEND : ST_CNT)) : ST_CNT;

        ST_SEND:
            state_N = (q_desc.tready) ? ST_CNT : ST_SEND;
    
    endcase
end

always_comb begin: DP
    qid_N = qid_C;
    cnt_N = cnt_C;

    // 
    q_desc.tvalid = 1'b0;
    q_desc.tdata = {cnt_C, qid_C};
    q_axis_tready = 1'b0;
    q_axis_tvalid = 1'b0;

    case (state_C)
        ST_CNT: begin
            s_axis.tready = q_axis_tready;
            q_axis_tvalid = s_axis.tvalid;
            
            if(s_axis.tvalid && s_axis.tready) begin
                cnt_N = cnt_C + 1;
            end
        end

        ST_SEND: begin
            q_desc.tvalid = 1'b1;
            cnt_N = q_desc.tready ? 0 : cnt_C;
        end 
 
    endcase
end


//
// DEBUG
//


endmodule 
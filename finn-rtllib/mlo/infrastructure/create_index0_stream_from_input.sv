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

module create_index0_stream_from_input #(
    parameter int unsigned CNT_BITS = 16
) (
    input  logic                    aclk,
    input  logic                    aresetn,

    AXI4S.slave                     s_axis_fs,
    input  logic                    s_axis_fs_done,
    AXI4S.master                    m_idx_fs
);

// State machine for packet detection
typedef enum logic[1:0] {
    ST_IDLE,        // Waiting for packet start
    ST_PACKET,      // In packet, waiting for packet end
    ST_SEND_INDEX   // Index ready to send
} state_t;

state_t state_C, state_N;

// Internal registers
logic index_valid_C, index_valid_N;
logic index_sent;

// State machine
always_comb begin
    state_N = state_C;
    index_valid_N = index_valid_C;

    case (state_C)
        ST_IDLE: begin
            if (s_axis_fs.tvalid) begin
                state_N = ST_SEND_INDEX;
                index_valid_N = 1'b1;
            end
        end

        ST_SEND_INDEX: begin
            if (index_sent) begin
                state_N = ST_PACKET;
                index_valid_N = 1'b0;
            end
        end

        ST_PACKET: begin
            if (s_axis_fs_done) begin
                state_N = ST_IDLE;
            end
        end

        default: begin
            state_N = ST_IDLE;
            index_valid_N = 1'b0;
        end
    endcase
end

// Sequential logic
always_ff @(posedge aclk) begin
    if (~aresetn) begin
        state_C <= ST_IDLE;
        index_valid_C <= 1'b0;
    end else begin
        state_C <= state_N;
        index_valid_C <= index_valid_N;
    end
end

// Output assignments
assign m_idx_fs.tvalid = index_valid_C;
assign m_idx_fs.tdata = '0;
assign index_sent = m_idx_fs.tvalid & m_idx_fs.tready;

// Input ready signal - always ready to accept input
assign s_axis_fs.tready = 1'b1;

endmodule

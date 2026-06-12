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
// indirect, special, incidental, or consequential loss or damage (inclu	ding loss
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

import iwTypes::*;

`include "axi_macros.svh"

module axis_ldet_wr #(
    parameter integer                                   LEN_BITS = HBM_LEN_BITS  
) (
    input  logic                                        aclk,
    input  logic                                        aresetn,

    AXI4S.slave                                         s_meta,

    AXI4S.slave                                         s_axis,
    AXI4S_PCKT.master                                   m_axis                
);

localparam integer BEATS_BITS = $clog2((ILEN_BITS+7)/8);
localparam integer CNT_BITS = LEN_BITS - BEATS_BITS;

typedef enum logic[0:0] {ST_IDLE, ST_WRITE} state_t;
state_t state_C = ST_IDLE, state_N;

logic [CNT_BITS-1:0] cnt_C = '0, cnt_N;

always_ff @( posedge aclk ) begin : REG
    if(~aresetn) begin
        state_C <= ST_IDLE;

        cnt_C <= 0;
    end
    else begin
        state_C <= state_N;

        cnt_C <= cnt_N;
    end
end

always_comb begin : NSL
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = s_meta.tvalid ? ST_WRITE : ST_IDLE;

        ST_WRITE:
            state_N = ((cnt_C == 0) && (m_axis.tvalid && m_axis.tready)) ? ST_IDLE : ST_WRITE;

    endcase
end

always_comb begin : DP 
    cnt_N = cnt_C;

    s_meta.tready = 1'b0;

    s_axis.tready = 1'b0;
    m_axis.tvalid = 1'b0;
    m_axis.tdata = s_axis.tdata;
    m_axis.tkeep = '1;
    m_axis.tlast = 1'b0;

    case (state_C)
        ST_IDLE: begin
            s_meta.tready = 1'b1;
            if(s_meta.tvalid) begin
                cnt_N = (s_meta.tdata - 1) >> BEATS_BITS;
            end
        end 

        ST_WRITE: begin
            m_axis.tvalid = s_axis.tvalid;
            s_axis.tready = m_axis.tready;

            if(cnt_C == 0) begin
                m_axis.tlast = 1'b1;
            end

            if(m_axis.tvalid && m_axis.tready) begin
                cnt_N = cnt_C - 1;
            end                                                                 
        end
    
    endcase

end

endmodule

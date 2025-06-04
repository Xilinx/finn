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

module axis_dwc #(
    parameter integer                   DEPTH = 512,
    parameter integer                   S_DATA_BITS = 32,
    parameter integer                   M_DATA_BITS = 8
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    input  logic                        s_axis_tvalid,
    output logic                        s_axis_tready,
    input  logic [S_DATA_BITS-1:0]      s_axis_tdata,
    input  logic [S_DATA_BITS/8-1:0]    s_axis_tkeep,
    input  logic                        s_axis_tlast,

    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready,
    output logic [S_DATA_BITS-1:0]      m_axis_tdata,
    output logic [S_DATA_BITS/8-1:0]    m_axis_tkeep,
    output logic                        m_axis_tlast
);

axis_fifo_adapter #(
    .DEPTH(DEPTH),
    .S_DATA_WIDTH(S_DATA_BITS),
    .M_DATA_WIDTH(M_DATA_BITS)
) inst_fifo_adapter (
    .clk            (aclk),
    .rst            (~aresetn),

    .s_axis_tdata   (s_axis_tdata),
    .s_axis_tkeep   (s_axis_tkeep),
    .s_axis_tvalid  (s_axis_tvalid),
    .s_axis_tready  (s_axis_tready),
    .s_axis_tlast   (s_axis_tlast),
    .s_axis_tid     ('0),
    .s_axis_tdest   ('0),
    .s_axis_tuser   ('0),


    .m_axis_tdata   (m_axis_tdata),
    .m_axis_tkeep   (m_axis_tkeep),
    .m_axis_tvalid  (m_axis_tvalid),
    .m_axis_tready  (m_axis_tready),
    .m_axis_tlast   (m_axis_tlast),
    .m_axis_tid     (),
    .m_axis_tdest   (),
    .m_axis_tuser   ()
);

endmodule

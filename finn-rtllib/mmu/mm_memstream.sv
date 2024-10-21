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
module mm_memstream #(
    parameter                                               MH,
    parameter                                               MW,              
    parameter                                               PE,
    parameter                                               ACTIVATION_WIDTH,
    parameter                                               INIT_FILE = "",
    parameter                                               RAM_STYLE = "auto",
    parameter                                               PUMPED_MEMORY = 0

) (
    input  logic                                            clk,
    input  logic                                            clk2x,
    input  logic                                            rst,

    output logic [PE-1:0][ACTIVATION_WIDTH-1:0]             odat,
    output logic                                            ovld,
    input  logic                                            ordy
);

memstream_axi #(
    .DEPTH((MH*MW)/PE),
    .WIDTH(PE*ACTIVATION_WIDTH),
    .INIT_FILE(INIT_FILE),
    .RAM_STYLE(RAM_STYLE),
    .PUMPED_MEMORY(PUMPED_MEMORY)
) inst_memstream (
    .clk(clk),
    .clk2x(clk2x),
    .rst(rst),

    .awready(),
    .awvalid(1'b0),
    .awprot(0),
    .awaddr(0),
    .arready(),
    .arvalid(1'b0),
    .arprot(0),
    .araddr(0),
    .wready(),
    .wvalid(1'b0),
    .wdata(0),
    .wstrb(0),
    .bready(1'b1),
    .bvalid(),
    .bresp(),
    .rready(1'b1),
    .rvalid(),
    .rresp(),
    .rdata(),

    .m_axis_0_tready(ordy),
    .m_axis_0_tvalid(ovld),
    .m_axis_0_tdata (odat)
);

endmodule
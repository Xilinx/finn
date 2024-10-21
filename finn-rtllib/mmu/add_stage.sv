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
module add_stage #(
    parameter                                               SIMD,
    parameter                                               PE,
    parameter                                               ACTIVATION_WIDTH,
    parameter                                               ACCU_WIDTH
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [PE-1:0][SIMD-1:0][2*ACTIVATION_WIDTH-1:0] idat_mul,
    input  logic                                            ival,
    input  logic                                            ilast,

    input  logic [PE-1:0][ACCU_WIDTH-1:0]                   i_acc,
    output logic                                            inc_acc,

    output logic [PE-1:0][ACCU_WIDTH-1:0]                   odat,
    output logic                                            oval,
    output logic                                            olast
);

logic [PE-1:0] oval_int;
logic [PE-1:0] olast_int;
logic [PE-1:0] inc_acc_int;

for(genvar i = 0; i < PE; i++) begin
    add_stage_single #(
		.SIMD(SIMD),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH)	
	) inst_add_stage (
		.clk(clk),
		.rst(rst),
		.en(en),
		
		.idat_mul(idat_mul[i]),
		.ival(ival),
		.ilast(ilast),

		.i_acc(i_acc[i]),
		.inc_acc(inc_acc_int[i]),

		.odat(odat[i]),
		.oval(oval_int[i]),
		.olast(olast_int[i])
	);
end

assign oval = oval_int[0];
assign olast = olast_int[0];
assign inc_acc = inc_acc_int[0];

endmodule
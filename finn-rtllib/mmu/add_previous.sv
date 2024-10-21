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
module add_stage_single #(
    parameter                                               SIMD,
    parameter                                               ACTIVATION_WIDTH,
    parameter                                               ACCU_WIDTH
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [SIMD-1:0][2*ACTIVATION_WIDTH-1:0]         idat_mul,
    input  logic                                            ival,
    input  logic                                            ilast,

    input  logic [ACCU_WIDTH-1:0]                           i_acc,
    output logic                                            inc_acc,

    output logic [ACCU_WIDTH-1:0]                           odat,
    output logic                                            oval,
    output logic                                            olast
);

localparam integer T_HEIGHT = SIMD-1;
localparam integer ADD_LAT = T_HEIGHT + 1;

// TODO: Use DSPs for this, if not then do tree reduction instead ...
logic signed [T_HEIGHT:0][SIMD-1:0][ACCU_WIDTH-1:0] add_s;

for(genvar j = 0; j < SIMD; j++) begin
    assign add_s[0][j] = ACCU_WIDTH'(signed'(idat_mul[j]));
end

always_ff @(posedge clk) begin
    if(rst) begin
        for(int j = 1; j <= T_HEIGHT; j++) begin
            add_s[j] <= 0;
        end
    end
    else begin
        if(en) begin
            for(int j = 1; j <= T_HEIGHT; j++) begin
                add_s[j][0] <= $signed(add_s[j-1][0]) + $signed(add_s[j-1][1]);

                for(int k = 0; k < T_HEIGHT-j; k++) begin
                    add_s[j][k+1] <= add_s[j-1][k+2];
                end
            end
        end
    end
end

logic signed [ACCU_WIDTH-1:0] odat_int;

// Add ACC
always_ff @(posedge clk) begin
    if(rst) begin
        odat_int <= 0;
    end
    else begin
        if(en) begin
            odat_int <= $signed(add_s[T_HEIGHT][0]) + $signed(i_acc);
        end
    end
end

assign odat = odat_int;

// REG
logic [ADD_LAT:0] val;
logic [ADD_LAT:0] last;

assign val[0] = ival;
assign last[0] = ilast;

always_ff @(posedge clk) begin
    if(rst) begin
        for(int i = 1; i <= ADD_LAT; i++) begin
            val[i] <= 0;
            last[i] <= 0;
        end
    end
    else begin
        if(en) begin
            for(int i = 1; i <= ADD_LAT; i++) begin
                val[i] <= val[i-1];
                last[i] <= last[i-1];
            end
        end 
    end
end

assign oval = val[ADD_LAT];
assign olast = last[ADD_LAT];

assign inc_acc = en && val[ADD_LAT-1];


endmodule
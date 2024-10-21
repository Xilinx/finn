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
//`define BEHAVIOURAL

module mul_stage #(
    parameter                                               SIMD,
    parameter                                               PE,
    parameter                                               ACTIVATION_WIDTH,
    parameter                                               WEIGHT_WIDTH,
    parameter                                               FORCE_BEHAVIOURAL = 0
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]           a,
    input  logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]       w,
    input  logic                                            ival,
    input  logic                                            ilast,

    output logic [PE-1:0][SIMD-1:0][2*ACTIVATION_WIDTH-1:0] odat,
    output logic                                            oval,
    output logic                                            olast
);

localparam integer MUL_LAT = FORCE_BEHAVIOURAL ? 2 : 5;

// DSPs
logic [PE-1:0][SIMD-1:0][2*ACTIVATION_WIDTH-1:0] odat_int = 0;

for(genvar i = 0; i < PE; i++) begin
    for(genvar j = 0; j < SIMD; j++) begin
        if(FORCE_BEHAVIOURAL) begin
            always_ff @(posedge clk) begin
                if(rst) begin
                    odat_int[i][j] <= 0;
                end
                else begin
                    odat_int[i][j] <= $signed(a[j]) * $signed(w[i][j]);
                end
            end
        end
        else begin
            dsp_macro_0 inst_dsp_mul (
                .CLK(clk),
                .CE(en),

                .A(a[j]),
                .B(w[i][j]),
                .P(odat_int[i][j])
            );
        end
    end
end

// REG
logic [MUL_LAT:0] val;
logic [MUL_LAT:0] last;

assign val[0] = ival;
assign last[0] = ilast;

always_ff @(posedge clk) begin
    if(rst) begin
        for(int i = 1; i <= MUL_LAT; i++) begin
            val[i] <= 0;
            last[i] <= 0;
        end

        odat <= 0;
    end
    else begin
        if(en) begin
            for(int i = 1; i <= MUL_LAT; i++) begin
                val[i] <= val[i-1];
                last[i] <= last[i-1];
            end

            odat <= odat_int;
        end 
    end
end

assign oval = val[MUL_LAT];
assign olast = last[MUL_LAT];


endmodule
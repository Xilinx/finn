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
module acc_stage #(
    parameter                                               PE,
    parameter                                               ACCU_WIDTH,
    parameter                                               TH
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [PE*ACCU_WIDTH-1:0]                        idat,
    input  logic                                            ival,
    input  logic                                            ilast,

    output logic [PE*ACCU_WIDTH-1:0]                        o_acc,
    input  logic                                            inc_acc,

    output logic [PE*ACCU_WIDTH-1:0]                        odat,
    output logic                                            oval
);

localparam integer TH_BITS = $clog2(TH);

logic [TH_BITS-1:0] cnt_prep = 0;
logic prep = 1'b1;

logic fifo_in_tvalid, fifo_in_tready;
logic fifo_out_tvalid, fifo_out_tready;
logic [PE*ACCU_WIDTH-1:0] fifo_in_tdata, fifo_out_tdata;

Q_srl #(
    .depth(TH), .width(PE*ACCU_WIDTH)
) inst_q (
    .clock(clk),
    .reset(rst),
    .count(),
    .maxcount(),
    .i_d(fifo_in_tdata),
    .i_v(fifo_in_tvalid),
    .i_r(fifo_in_tready),
    .o_d(fifo_out_tdata),
    .o_v(fifo_out_tvalid),
    .o_r(fifo_out_tready)
);

always_ff @(posedge clk) begin
    if(rst) begin
        odat <= 0;
        oval <= 1'b0;

        cnt_prep <= 0;
        prep <= 1'b1;
    end
    else begin
        if(cnt_prep == TH-1) begin
            prep <= 1'b0;
            cnt_prep <= 0;
        end
        else begin
            cnt_prep <= cnt_prep + 1;
        end

        if(en) begin
            if(ival) begin
                odat <= idat;
                oval <= ival && ilast;
            end
        end 
    end
end

always_comb begin
    fifo_in_tvalid = 1'b0;
    fifo_in_tdata = 0;

    if(en) begin
        fifo_in_tvalid = prep ? 1'b1 : ival;
        fifo_in_tdata = prep ? 0 : (ilast ? 0 : idat); 
    end
end

assign o_acc = fifo_out_tdata;
assign fifo_out_tready = inc_acc;

endmodule
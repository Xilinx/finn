/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/


module acc_stage #(
    int unsigned                                            CHAINLEN,
    int unsigned                                            PE,
    int unsigned                                            ACCU_WIDTH,
    int unsigned                                            TH,
    int unsigned                                            TH_MAX = 2*TH
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [PE-1:0][CHAINLEN-1:0][ACCU_WIDTH-1:0]     idat,
    input  logic                                            ival,
    input  logic                                            ilast,

    output logic [PE-1:0][ACCU_WIDTH-1:0]                   odat,
    output logic                                            oval
);

//
// Adder tree + accumulator add
//

localparam integer TREE_DEPTH = $clog2(CHAINLEN);
localparam integer ADD_LAT = TREE_DEPTH + 1;

logic [PE-1:0][ACCU_WIDTH-1:0] dat_acc;
logic [PE-1:0][ACCU_WIDTH-1:0] dat_int;

for(genvar i = 0; i < PE; i++) begin : genAdd
    // Tree reduction of CHAINLEN DSP partial products
    logic [ACCU_WIDTH-1:0]  add_arg[CHAINLEN];
    for(genvar k = 0; k < CHAINLEN; k++)
        assign  add_arg[k] = idat[i][k];

    localparam int unsigned  SUM_WIDTH = $clog2(CHAINLEN) + ACCU_WIDTH;
    uwire [SUM_WIDTH-1:0]  tree_sum;
    add_multi #(
        .N(CHAINLEN),
        .DEPTH(TREE_DEPTH),
        .ARG_WIDTH(ACCU_WIDTH)
    ) inst_add (
        .clk(clk), .rst(rst), .en(en),
        .arg(add_arg),
        .sum(tree_sum)
    );

    // Accumulator add (1 registered stage)
    always_ff @(posedge clk) begin
        if(rst)       dat_int[i] <= 'x;
        else if(en)   dat_int[i] <= tree_sum[ACCU_WIDTH-1:0] + dat_acc[i];
    end
end : genAdd

// REG
logic [ADD_LAT:0] val;
logic [ADD_LAT:0] last;

assign val[0] = ival;
assign last[0] = ilast;

always_ff @(posedge clk) begin
    if(rst) begin
        for(int i = 1; i <= ADD_LAT; i++) begin
            val[i] <= 1'b0;
            last[i] <= 'X;
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

logic val_int;
logic last_int;
logic inc_acc;

assign val_int = val[ADD_LAT];
assign last_int = last[ADD_LAT];
assign inc_acc = val[ADD_LAT-1];

//
// Accumulation
//

logic fifo_in_tvalid, fifo_in_tready;
logic fifo_out_tvalid, fifo_out_tready;
logic [PE*ACCU_WIDTH-1:0] fifo_in_tdata, fifo_out_tdata;

Q_srl #(
    .depth(TH_MAX),
    .width(PE*ACCU_WIDTH)
) inst_acc (
    .clock(clk),
    .reset(rst),
    .i_d(fifo_in_tdata),
    .i_v(fifo_in_tvalid),
    .i_r(fifo_in_tready),
    .o_d(fifo_out_tdata),
    .o_v(fifo_out_tvalid),
    .o_r(fifo_out_tready),
    .count(),
    .maxcount()
);

logic signed [$clog2(TH):0] cnt_prep = -TH;
uwire prep = cnt_prep[$left(cnt_prep)];
always_ff @(posedge clk) begin
    if(rst)  cnt_prep <= -TH;
    else     cnt_prep <= cnt_prep + prep;
end

always_ff @(posedge clk) begin
    if(rst) begin
        odat <= 'x;
        oval <= 0;
    end
    else if(en) begin
        odat <= dat_int;
        oval <= val_int && last_int;
    end
end

always_comb begin
    fifo_in_tvalid = prep ? 1'b1 : (en ? val_int : 1'b0);
    fifo_in_tdata = prep ? 0 : (last_int ? 0 : dat_int);
end

assign dat_acc = fifo_out_tdata;
assign fifo_out_tready = en & inc_acc;

endmodule : acc_stage

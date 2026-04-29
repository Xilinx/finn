/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Select one token from a folded token stream.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 *
 * @description
 *	Consumes NUM_TOKENS token vectors fold-by-fold. Folds belonging to
 *	TOKEN_INDEX are forwarded to the output stream; all other folds are
 *	consumed and discarded.
 ***************************************************************************/

module select_token #(
    parameter int unsigned NUM_TOKENS = 197,
    parameter int unsigned NUM_CHANNELS = 192,
    parameter int unsigned SIMD = 1,
    parameter int unsigned ELEM_WIDTH = 8,
    parameter int unsigned TOKEN_INDEX = 0
)(
    input  logic clk,
    input  logic rst,

    output logic irdy,
    input  logic ivld,
    input  logic [SIMD*ELEM_WIDTH-1:0] idat,

    input  logic ordy,
    output logic ovld,
    output logic [SIMD*ELEM_WIDTH-1:0] odat
);

    localparam int unsigned FOLDS_PER_TOKEN = NUM_CHANNELS / SIMD;
    localparam int unsigned TOKEN_CNT_WIDTH = (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS);
    localparam int unsigned FOLD_CNT_WIDTH =
        (FOLDS_PER_TOKEN <= 1) ? 1 : $clog2(FOLDS_PER_TOKEN);

    logic [TOKEN_CNT_WIDTH-1:0] token_cnt;
    logic [FOLD_CNT_WIDTH-1:0] fold_cnt;
    logic is_selected;
    logic in_transfer;
    logic fold_cnt_last;
    logic token_cnt_last;

    assign is_selected = (int'(token_cnt) == TOKEN_INDEX);
    assign in_transfer = irdy & ivld;
    assign fold_cnt_last = (int'(fold_cnt) == FOLDS_PER_TOKEN - 1);
    assign token_cnt_last = (int'(token_cnt) == NUM_TOKENS - 1);

    always_comb begin
        irdy = 1'b1;
        ovld = 1'b0;
        odat = '0;

        if (is_selected) begin
            irdy = ordy;
            ovld = ivld;
            odat = idat;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            token_cnt <= '0;
            fold_cnt <= '0;
        end else if (in_transfer) begin
            if (fold_cnt_last) begin
                fold_cnt <= '0;
                if (token_cnt_last) begin
                    token_cnt <= '0;
                end else begin
                    token_cnt <= token_cnt + 1'b1;
                end
            end else begin
                fold_cnt <= fold_cnt + 1'b1;
            end
        end
    end

endmodule

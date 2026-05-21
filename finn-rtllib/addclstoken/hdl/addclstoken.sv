/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Insert a constant class token into a folded token stream.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 *
 * @description
 *	Prepends a learned class token, supplied through cls_data, to each
 *	input sequence of patch tokens. The class token and patch tokens are
 *	transferred as SIMD-wide folds of ELEM_WIDTH-bit elements.
 *
 *	Per sequence, the output stream is:
 *	  1. NUM_CHANNELS/SIMD folds from cls_data
 *	  2. NUM_TOKENS pass-through input tokens
 *	  3. PAD_TOKENS zero-valued tokens, when padding is enabled
 ***************************************************************************/

module addclstoken #(
    parameter int unsigned NUM_TOKENS = 196,
    parameter int unsigned NUM_CHANNELS = 192,
    parameter int unsigned SIMD = 1,
    parameter int unsigned ELEM_WIDTH = 8,
    parameter int unsigned PAD_TOKENS = 0
)(
    input  logic clk,
    input  logic rst,

    output logic irdy,
    input  logic ivld,
    input  logic [SIMD*ELEM_WIDTH-1:0] idat,

    input  logic ordy,
    output logic ovld,
    output logic [SIMD*ELEM_WIDTH-1:0] odat,

    input  logic [NUM_CHANNELS*ELEM_WIDTH-1:0] cls_data
);

    localparam int unsigned FOLD_WIDTH = SIMD * ELEM_WIDTH;
    localparam int unsigned FOLDS_PER_TOKEN = NUM_CHANNELS / SIMD;
    localparam int unsigned TOTAL_INPUT_FOLDS = NUM_TOKENS * FOLDS_PER_TOKEN;
    localparam int unsigned TOTAL_PAD_FOLDS = PAD_TOKENS * FOLDS_PER_TOKEN;
    localparam int unsigned MAX_PHASE_FOLDS =
        (TOTAL_INPUT_FOLDS > FOLDS_PER_TOKEN) ?
            ((TOTAL_INPUT_FOLDS > TOTAL_PAD_FOLDS) ?
                TOTAL_INPUT_FOLDS : TOTAL_PAD_FOLDS) :
            ((FOLDS_PER_TOKEN > TOTAL_PAD_FOLDS) ?
                FOLDS_PER_TOKEN : TOTAL_PAD_FOLDS);
    localparam int unsigned CNT_WIDTH = (MAX_PHASE_FOLDS <= 1) ? 1 : $clog2(MAX_PHASE_FOLDS);

    typedef enum logic [1:0] {
        EMIT_CLS,
        PASSTHROUGH,
        EMIT_PAD
    } state_t;

    state_t state;
    state_t next_state;
    logic [CNT_WIDTH-1:0] fold_cnt;
    logic fold_cnt_last;
    logic out_transfer;

    logic [CNT_WIDTH-1:0] cls_fold_cnt;
    logic [FOLD_WIDTH-1:0] cls_fold;

    assign cls_fold_cnt = (int'(fold_cnt) < FOLDS_PER_TOKEN) ? fold_cnt : '0;
    assign cls_fold = cls_data[cls_fold_cnt * FOLD_WIDTH +: FOLD_WIDTH];
    assign out_transfer = ovld & ordy;

    always_comb begin
        unique case (state)
            EMIT_CLS:    fold_cnt_last = (int'(fold_cnt) == FOLDS_PER_TOKEN - 1);
            PASSTHROUGH: fold_cnt_last = (int'(fold_cnt) == TOTAL_INPUT_FOLDS - 1);
            EMIT_PAD:    fold_cnt_last = (int'(fold_cnt) == TOTAL_PAD_FOLDS - 1);
            default:     fold_cnt_last = 1'b1;
        endcase
    end

    always_comb begin
        irdy = 1'b0;
        ovld = 1'b0;
        odat = '0;

        unique case (state)
            EMIT_CLS: begin
                ovld = 1'b1;
                odat = cls_fold;
            end
            PASSTHROUGH: begin
                irdy = ordy;
                ovld = ivld;
                odat = idat;
            end
            EMIT_PAD: begin
                ovld = 1'b1;
            end
            default: begin
            end
        endcase
    end

    always_comb begin
        next_state = state;
        if (out_transfer && fold_cnt_last) begin
            unique case (state)
                EMIT_CLS: begin
                    next_state = PASSTHROUGH;
                end
                PASSTHROUGH: begin
                    next_state = (PAD_TOKENS == 0) ? EMIT_CLS : EMIT_PAD;
                end
                EMIT_PAD: begin
                    next_state = EMIT_CLS;
                end
                default: begin
                    next_state = EMIT_CLS;
                end
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= EMIT_CLS;
            fold_cnt <= '0;
        end else if (out_transfer) begin
            if (fold_cnt_last) begin
                state <= next_state;
                fold_cnt <= '0;
            end else begin
                fold_cnt <= fold_cnt + 1'b1;
            end
        end
    end

endmodule

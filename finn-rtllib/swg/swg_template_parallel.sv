/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 *****************************************************************************/

module $TOP_MODULE_NAME$_wb
#(
    int unsigned IN_WIDTH          = 1, // bit-width*C*MMV_in
    int unsigned OUT_ELEM_WIDTH    = 1, // bit-width*C
    int unsigned OUT_WIDTH         = 1, // bit-width*C*MMV_out
    int unsigned BUFFER_ELEM_TOTAL = 1
)
(
    input logic CLK,
    input logic RST,
    input logic shift_enable,
    input logic [IN_WIDTH-1:0] data_in,
    output logic [OUT_WIDTH-1:0] data_out
);

$GENERATE_REG_FIFOS$

$GENERATE_BRAM_FIFOS$

// fixed interconnect between linear buffers
$GENERATE_BUFFER_CONNECTION$

// fixed REG FIFO -> output mapping
$GENERATE_OUTPUT_MAPPING$

endmodule : $TOP_MODULE_NAME$_wb

module $TOP_MODULE_NAME$_impl #(
    int unsigned BIT_WIDTH,
    int unsigned SIMD,
    int unsigned MMV_IN,
    int unsigned MMV_OUT,
    int unsigned LAST_READ_ELEM = $LAST_READ_ELEM$,
    int unsigned FIRST_WRITE_ELEM = $FIRST_WRITE_ELEM$,
    int unsigned LAST_WRITE_ELEM = $LAST_WRITE_ELEM$,
    int unsigned BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$,
    int unsigned INCR_BITWIDTH = $INCR_BITWIDTH$
)(
    input   logic  ap_clk,
    input   logic  ap_rst_n,

    input   logic  in0_V_V_TVALID,
    output  logic  in0_V_V_TREADY,
    input   logic [BIT_WIDTH * SIMD * MMV_IN-1:0]  in0_V_V_TDATA,

    output  logic  out_V_V_TVALID,
    input   logic  out_V_V_TREADY,
    output  logic [BIT_WIDTH * SIMD * MMV_OUT-1:0]  out_V_V_TDATA
);
    // derived constants
    localparam int unsigned  BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
    localparam int unsigned  BUF_OUT_ELEM_WIDTH = BIT_WIDTH * SIMD;
    localparam int unsigned  BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;

    // main buffer instantiation
    uwire [BUF_IN_WIDTH -1:0] window_buffer_in;
    uwire [BUF_OUT_WIDTH-1:0] window_buffer_out;
    uwire window_buffer_shift_enable;
    $TOP_MODULE_NAME$_wb
    #(
        .IN_WIDTH(BUF_IN_WIDTH),
        .OUT_ELEM_WIDTH(BUF_OUT_ELEM_WIDTH),
        .OUT_WIDTH(BUF_OUT_WIDTH),
        .BUFFER_ELEM_TOTAL(BUF_ELEM_TOTAL)
    )
    window_buffer_inst
    (
        .CLK(ap_clk),
        .RST(ap_rst_n),
        .data_in(window_buffer_in),
        .shift_enable(window_buffer_shift_enable),
        .data_out(window_buffer_out)
    );

    // controller instantiation
    uwire  advance_controller;
    uwire signed [INCR_BITWIDTH-1:0]  addr_incr;
    uwire        [INCR_BITWIDTH-1:0]  tail_incr;
    swg_controller #(
        .LOOP_H_ITERATIONS($LOOP_H_ITERATIONS$),
        .LOOP_W_ITERATIONS($LOOP_W_ITERATIONS$),
        .LOOP_KH_ITERATIONS($LOOP_KH_ITERATIONS$),
        .LOOP_KW_ITERATIONS($LOOP_KW_ITERATIONS$),
        .LOOP_SIMD_ITERATIONS($LOOP_SIMD_ITERATIONS$),
        .HEAD_INCR_SIMD($HEAD_INCR_SIMD$),
        .HEAD_INCR_KW($HEAD_INCR_KW$),
        .HEAD_INCR_KH($HEAD_INCR_KH$),
        .HEAD_INCR_W($HEAD_INCR_W$),
        .HEAD_INCR_H($HEAD_INCR_H$),
        .TAIL_INCR_W($TAIL_INCR_W$),
        .TAIL_INCR_H($TAIL_INCR_H$),
        .TAIL_INCR_LAST($TAIL_INCR_LAST$),
        .INCR_BITWIDTH($INCR_BITWIDTH$),
        .IS_DEPTHWISE($IS_DEPTHWISE$),
        .INNERMOST_STATE($INNERMOST_STATE$)
    )
    controller_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .advance(advance_controller),
        .addr_incr(addr_incr),
        .tail_incr(tail_incr)
    );

    // counters/address registers
    logic signed [$clog2(LAST_READ_ELEM+1)+1-1:0]  Newest_buffered_elem = -1;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  Current_elem = FIRST_WRITE_ELEM;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  First_elem_next_window = 0;

    // control registers/signals
    logic  Writing_done  = 0;
    logic  Write_done    = 0;
    uwire  write_cmd     = !($signed(Current_elem) > Newest_buffered_elem) && !Writing_done;;
    uwire  write_ok      = write_cmd && (out_V_V_TREADY || Write_done);
    uwire  write_blocked = write_cmd && !out_V_V_TREADY && !Write_done;

    uwire  reading_done = Newest_buffered_elem == LAST_READ_ELEM;
    uwire  read_cmd     =
        !reading_done && ( // if there is still an input element left to read
            Writing_done || ( // if writing is done (e.g. for skipped rows at FM end due to stride)
                $signed(((Newest_buffered_elem - ($signed(BUF_ELEM_TOTAL) - 1)))) < $signed(First_elem_next_window) &&
                $signed(((Newest_buffered_elem - ($signed(BUF_ELEM_TOTAL) - 1)))) < $signed(Current_elem)
            ) // (over-)write to buffer if oldest buffered element will no longer be needed
        );
    uwire  read_ok      = read_cmd && in0_V_V_TVALID && !write_blocked;

    //            includes waiting on W    if W-only cycle: wait only on W     no R/W to wait for
    uwire advance       = read_ok        ||   (!read_cmd && write_ok)    || (!read_cmd && !write_cmd);

    // assign buffer control
    assign window_buffer_shift_enable = advance;
    assign  advance_controller = write_ok;

    // assign I/O ports
    assign  window_buffer_in = in0_V_V_TDATA;
    assign  out_V_V_TDATA = window_buffer_out;
    assign  in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
    assign  out_V_V_TVALID = ap_rst_n && write_cmd && !Write_done; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

    // write done logic
    always_ff @(posedge ap_clk) begin
        if(!ap_rst_n) begin
            Write_done <= 1'b0;
        end
        else begin
            if (advance) begin
                Write_done <= 1'b0; //reset flag
            end else if (write_ok)  //successful W in this cycle, but R still outstanding
                Write_done <= 1'b1; //write can happen even if read is blocked, but only for the current cycle!
        end
    end

    // main process for advancing counters
    always_ff @(posedge ap_clk) begin
        if(!ap_rst_n) begin
            Newest_buffered_elem <= -1;
            Current_elem <= FIRST_WRITE_ELEM;
            First_elem_next_window <= 0;
            Writing_done <= 0;
        end
        else begin
            if (read_ok) begin
                Newest_buffered_elem <= Newest_buffered_elem+1;

                // check if this is the last read cycle (reading_done will be true afterwards)
                if ((Newest_buffered_elem == LAST_READ_ELEM-1) && Writing_done) begin
                    // start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                    // todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                    Newest_buffered_elem <= -1;
                    Current_elem <= FIRST_WRITE_ELEM;
                    First_elem_next_window <= 0;
                    Writing_done <= 0;
                end
            end

            if (write_ok) begin
                First_elem_next_window <= First_elem_next_window + tail_incr;

                // check if this is the last write cycle (Writing_done will be true afterwards)
                if (Current_elem == LAST_WRITE_ELEM) begin
                    Writing_done <= 1;

                    if (reading_done || (read_ok && (Newest_buffered_elem == LAST_READ_ELEM - 1))) begin
                        // start processing of next FM if reading is done already, or completes in the same cycle
                        Newest_buffered_elem <= -1;
                        Current_elem <= FIRST_WRITE_ELEM;
                        First_elem_next_window <= 0;
                        Writing_done <= 0;
                    end
                end
                else
                    Current_elem <= $signed(Current_elem) + addr_incr;
            end
        end
    end

endmodule : $TOP_MODULE_NAME$_impl

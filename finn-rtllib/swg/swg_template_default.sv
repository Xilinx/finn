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
module $TOP_MODULE_NAME$_impl #(
    int  BIT_WIDTH,
    int  SIMD,
    int  MMV_IN,
    int  MMV_OUT,
    int  LAST_READ_ELEM = $LAST_READ_ELEM$,
    int  LAST_WRITE_ELEM = $LAST_WRITE_ELEM$,
    int  BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$,
    int  ELEM_PER_WINDOW = $ELEM_PER_WINDOW$,
    int  INCR_BITWIDTH = $INCR_BITWIDTH$
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
    uwire [BUF_IN_WIDTH -1:0]  window_buffer_in;
    uwire [BUF_OUT_WIDTH-1:0]  window_buffer_out;
    uwire  window_buffer_write_enable;
    uwire  window_buffer_read_enable;
    uwire [$clog2(BUF_ELEM_TOTAL)-1:0]  window_buffer_write_addr;
    uwire [$clog2(BUF_ELEM_TOTAL)-1:0]  window_buffer_read_addr;
    swg_cyclic_buffer_addressable #(
        .WIDTH(BUF_IN_WIDTH),
        .DEPTH(BUF_ELEM_TOTAL),
        .RAM_STYLE($RAM_STYLE$)
    ) window_buffer_inst (
        .clk(ap_clk),

        .write_enable(window_buffer_write_enable),
        .write_addr(window_buffer_write_addr),
        .data_in(window_buffer_in),

        .read_enable(window_buffer_read_enable),
        .read_addr(window_buffer_read_addr),
        .data_out(window_buffer_out)
    );

    //controller instantiation
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
        .INNERMOST_STATE(swg::$INNERMOST_STATE$)
    )
    controller_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .advance(advance_controller),
        .addr_incr(addr_incr),
        .tail_incr(tail_incr)
    );

    // Counters/address registers
    // Add a sign bit even to (most) unsigned counters and Window_buffer_read_addr_reg,
    // so we can use automatic sign extension and simplify calculations w/ signed increment.
    // Alternatively, we could manually sign-extend and shave off a bit here or there.

    //                                                                                          v ReadingLast     v ReadingDone
    logic signed [$clog2(LAST_READ_ELEM+1)+1-1:0]  Newest_buffered_elem = -1; // -1, 0, 1, ..., LAST_READ_ELEM-1, LAST_READ_ELEM
    logic  ReadingLast = LAST_READ_ELEM == 0;
    logic  ReadingDone = 0;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  Current_elem = 0;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  First_elem_next_window = 0;

    //                                                               v PositionZero
    logic [$clog2(ELEM_PER_WINDOW)-1:0]  Position_in_window = 0;  // 0, 1, ..., ELEM_PER_WINDOW-1
    logic  PositionZero = 1;

    logic [$clog2(BUF_ELEM_TOTAL)+1-1:0]  Window_buffer_read_addr_reg = 0;
    logic [$clog2(BUF_ELEM_TOTAL)  -1:0]  Window_buffer_write_addr_reg = 0;

    // Control signals/registers
    logic  Write_cmd    = 0;
    logic  Writing_done = 0;
    uwire  write_ok      = Write_cmd &&  out_V_V_TREADY;
    uwire  write_blocked = Write_cmd && !out_V_V_TREADY;

    logic  Fetching_done = 0;
    uwire  fetch_cmd = !($signed(Current_elem) > Newest_buffered_elem) && !write_blocked && !Fetching_done;

    uwire  read_cmd =
        !ReadingDone && ( // if there is still an input element left to read
            Fetching_done || ( // if fetching is done (e.g. for skipped rows at FM end due to stride)
                $signed(((Newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(First_elem_next_window) &&
                $signed(((Newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(Current_elem)
            ) // (over-)write to buffer if oldest buffered element will no longer be needed
        );
    uwire  read_ok      = read_cmd && in0_V_V_TVALID;

    //assign buffer control
    assign  window_buffer_write_addr = Window_buffer_write_addr_reg;
    assign  window_buffer_read_addr = Window_buffer_read_addr_reg;
    assign  window_buffer_write_enable = read_ok;
    assign  window_buffer_read_enable = fetch_cmd;
    assign  advance_controller = fetch_cmd;

    //assign I/O ports
    assign  window_buffer_in = in0_V_V_TDATA;
    assign  out_V_V_TDATA = window_buffer_out;
    assign  in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
    assign  out_V_V_TVALID = ap_rst_n && Write_cmd; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

    //main process for advancing counters
    always_ff @(posedge ap_clk) begin
        if(!ap_rst_n) begin
            Newest_buffered_elem <= -1;
            ReadingLast <= LAST_READ_ELEM == 0;
            ReadingDone <= 0;

            Current_elem <= 0;
            First_elem_next_window <= 0;

            Position_in_window <= 0;
            PositionZero <= 1;

            Window_buffer_read_addr_reg <= 0;
            Window_buffer_write_addr_reg <= 0;
            Fetching_done <= 0;
            Write_cmd <= 0;
            Writing_done <= 0;
        end
        else begin
            automatic logic  nbe_rst = 0;
            automatic logic  nbe_inc = 0;

            if (read_ok) begin
                automatic logic  wa_rst = ReadingLast ||
                    ((~Window_buffer_write_addr_reg & (BUF_ELEM_TOTAL-1)) == 0); // (Window_buffer_write_addr_reg == BUF_ELEM_TOTAL-1)
                Window_buffer_write_addr_reg <= Window_buffer_write_addr_reg - (wa_rst? Window_buffer_write_addr_reg : -1);

                nbe_inc = 1;

                //check if this is the last read cycle (ReadingDone will be true afterwards)
                if(ReadingLast && Writing_done) begin
                    //start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                    //todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                    nbe_rst = 1;

                    Current_elem <= 0;
                    Window_buffer_read_addr_reg <= 0;
                    First_elem_next_window <= 0;
                    Writing_done <= 0;
                    Fetching_done <= 0;
                end
            end

            if (fetch_cmd) begin
                automatic logic  wrap_position;

                //count up to track which element index is about to be read from the buffer, and where it is located within the buffer
                //use increment value calculated by controller

                // absolute buffer address wrap-around
                automatic logic signed [$clog2(BUF_ELEM_TOTAL)+1:0]  ra = $signed(Window_buffer_read_addr_reg) + $signed(addr_incr);
                automatic logic signed [$clog2(BUF_ELEM_TOTAL+1):0]  ra_correct =
                    (ra >= BUF_ELEM_TOTAL)? -BUF_ELEM_TOTAL :
                    (ra <               0)?  BUF_ELEM_TOTAL : 0;
                Window_buffer_read_addr_reg <= ra + ra_correct;

                //keep track where we are within a window
                wrap_position = (~Position_in_window & (ELEM_PER_WINDOW - 1)) == 0;
                Position_in_window <= Position_in_window + (wrap_position? -ELEM_PER_WINDOW+1 : 1);
                PositionZero <= wrap_position;

                //update first element of next window to allow buffer overwrite up until that point
                First_elem_next_window <= First_elem_next_window + (PositionZero? tail_incr : 0);

                //check if this is the last write cycle (Writing_done will be true afterwards)
                if (Current_elem == LAST_WRITE_ELEM)
                    Fetching_done <= 1;
                else
                    Current_elem <= $signed(Current_elem) + addr_incr;

                // determine if prefetched data will be outstanding in the next cycle
                // if we fetch in this cycle -> yes
                // if we do not fetch nor write -> do not change
                // if we do not fetch but write successfully-> clear outstanding data
                Write_cmd <= fetch_cmd;
            end

            if(write_ok) begin
                Write_cmd <= fetch_cmd;

                if(Fetching_done) begin
                    //check if this is the last write cycle (Writing_done will be true afterwards)
                    if(ReadingDone || (read_ok && ReadingLast)) begin
                        //start processing of next FM if reading is done already, or completes in the same cycle
                        nbe_rst = 1;

                        Current_elem <= 0;
                        Window_buffer_read_addr_reg <= 0;
                        First_elem_next_window <= 0;
                        Fetching_done <= 0;
                    end
                    else  Writing_done <= 1;
                end
            end

            Newest_buffered_elem <= Newest_buffered_elem + (nbe_rst? ~Newest_buffered_elem : nbe_inc);
            ReadingLast <= nbe_rst? LAST_READ_ELEM == 0 : !nbe_inc? ReadingLast : ((~Newest_buffered_elem & (LAST_READ_ELEM-2)) == 0) && !ReadingLast;
            ReadingDone <= nbe_rst?                   0 : !nbe_inc? ReadingDone : ReadingLast;
        end
    end

endmodule : $TOP_MODULE_NAME$_impl

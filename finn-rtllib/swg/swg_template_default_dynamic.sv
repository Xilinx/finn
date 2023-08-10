/******************************************************************************
 * Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
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

module $TOP_MODULE_NAME$_controller #(
    int unsigned  CNTR_BITWIDTH,
    int unsigned  INCR_BITWIDTH,

    bit IS_DEPTHWISE = $IS_DEPTHWISE$
)(
    input   logic  clk,
    input   logic  rst_n,

    input   logic  advance,
    output  logic [INCR_BITWIDTH-1:0]  addr_incr,
    output  logic [INCR_BITWIDTH-1:0]  tail_incr,

    input logic                     cfg_valid,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_simd,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_kw,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_kh,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_w,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_simd,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_kw,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_kh,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_w,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_w,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_last
);

    import  swg::*;

    // (dynamic) configuration registers
    logic [CNTR_BITWIDTH-1:0] Cfg_cntr_simd      = $LOOP_SIMD_ITERATIONS$;
    logic [CNTR_BITWIDTH-1:0] Cfg_cntr_kw        = $LOOP_KW_ITERATIONS$;
    logic [CNTR_BITWIDTH-1:0] Cfg_cntr_kh        = $LOOP_KH_ITERATIONS$;
    logic [CNTR_BITWIDTH-1:0] Cfg_cntr_w         = $LOOP_W_ITERATIONS$;
    logic [CNTR_BITWIDTH-1:0] Cfg_cntr_h         = $LOOP_H_ITERATIONS$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_head_simd = $HEAD_INCR_SIMD$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_head_kw   = $HEAD_INCR_KW$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_head_kh   = $HEAD_INCR_KH$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_head_w    = $HEAD_INCR_W$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_head_h    = $HEAD_INCR_H$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_tail_w    = $TAIL_INCR_W$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_tail_h    = $TAIL_INCR_H$;
    logic [INCR_BITWIDTH-1:0] Cfg_incr_tail_last = $TAIL_INCR_LAST$;

    // configuration reset/set logic
    always_ff @ (posedge clk) begin
        if(cfg_valid) begin
            Cfg_cntr_simd      <= cfg_cntr_simd;
            Cfg_cntr_kw        <= cfg_cntr_kw;
            Cfg_cntr_kh        <= cfg_cntr_kh;
            Cfg_cntr_w         <= cfg_cntr_w;
            Cfg_cntr_h         <= cfg_cntr_h;
            Cfg_incr_head_simd <= cfg_incr_head_simd;
            Cfg_incr_head_kw   <= cfg_incr_head_kw;
            Cfg_incr_head_kh   <= cfg_incr_head_kh;
            Cfg_incr_head_w    <= cfg_incr_head_w;
            Cfg_incr_head_h    <= cfg_incr_head_h;
            Cfg_incr_tail_w    <= cfg_incr_tail_w;
            Cfg_incr_tail_h    <= cfg_incr_tail_h;
            Cfg_incr_tail_last <= cfg_incr_tail_last;
        end
    end

    // state and counters
    state_e  State = $INNERMOST_STATE$;
    state_e  state_next;

    logic signed [$clog2($LOOP_H_ITERATIONS$   +2)+1-1:0]  Counter_loop_h    = $LOOP_H_ITERATIONS$;
    logic signed [$clog2($LOOP_W_ITERATIONS$   +2)+1-1:0]  Counter_loop_w    = $LOOP_W_ITERATIONS$;
    logic signed [$clog2($LOOP_KH_ITERATIONS$  +2)+1-1:0]  Counter_loop_kh   = $LOOP_KH_ITERATIONS$;
    logic signed [$clog2($LOOP_KW_ITERATIONS$  +2)+1-1:0]  Counter_loop_kw   = $LOOP_KW_ITERATIONS$;
    logic signed [$clog2($LOOP_SIMD_ITERATIONS$+2)+1-1:0]  Counter_loop_simd = $LOOP_SIMD_ITERATIONS$;

    // combinational logic for addr_incr generation
    always_comb begin : blkHead
        unique case (State)
            0 : addr_incr = 0;
            1 : addr_incr = Cfg_incr_head_simd;
            2 : addr_incr = Cfg_incr_head_kw;
            3 : addr_incr = Cfg_incr_head_kh;
            4 : addr_incr = Cfg_incr_head_w;
            5 : addr_incr = Cfg_incr_head_h;
        endcase
    end

    // combinational logic for tail_incr generation
    uwire  tail_incr_inner_condition = IS_DEPTHWISE? (Counter_loop_kh >= 0) : 0;
    assign tail_incr =
        tail_incr_inner_condition? 1 :
        Counter_loop_w >= 0?       Cfg_incr_tail_w :
        Counter_loop_h >= 0?       Cfg_incr_tail_h :
        /* else */                 Cfg_incr_tail_last;

    // combinational next state logic
    always_comb begin : blkState
        state_next = State;
        if(State != $INNERMOST_STATE$)  state_next = $INNERMOST_STATE$;
        else begin
            if(Counter_loop_simd < 0) begin
                state_next =
                    (Counter_loop_kw >= 0)? STATE_LOOP_KW :
                    (Counter_loop_kh >= 0)? STATE_LOOP_KH :
                    (Counter_loop_w  >= 0)? STATE_LOOP_W :
                    (Counter_loop_h  >= 0)? STATE_LOOP_H :
                    /* else */              STATE_START;
            end
        end
    end : blkState

    // sequential logic
    always_ff @ (posedge clk) begin
        if(!rst_n) begin
            State <= $INNERMOST_STATE$;
            Counter_loop_h    <= Cfg_cntr_h;
            Counter_loop_w    <= Cfg_cntr_w;
            Counter_loop_kh   <= Cfg_cntr_kh;
            Counter_loop_kw   <= Cfg_cntr_kw;
            Counter_loop_simd <= Cfg_cntr_simd;
        end
        else if(advance) begin
            State <= state_next;
            if (State == $INNERMOST_STATE$) begin
                if(Counter_loop_simd >= 0)  Counter_loop_simd <= Counter_loop_simd-1;
                else begin
                    Counter_loop_simd <= Cfg_cntr_simd;
                    if(Counter_loop_kw >= 0)  Counter_loop_kw <= Counter_loop_kw-1;
                    else begin
                        Counter_loop_kw <= Cfg_cntr_kw;
                        if(Counter_loop_kh >= 0)  Counter_loop_kh <= Counter_loop_kh-1;
                        else begin
                            Counter_loop_kh <= Cfg_cntr_kh;
                            if(Counter_loop_w >= 0)  Counter_loop_w <= Counter_loop_w-1;
                            else begin
                                Counter_loop_w <= Cfg_cntr_w;
                                if(Counter_loop_h >= 0)  Counter_loop_h <= Counter_loop_h-1;
                                else  Counter_loop_h <= Cfg_cntr_h;
                            end
                        end
                    end
                end
            end
        end
    end

endmodule :  $TOP_MODULE_NAME$_controller

module $TOP_MODULE_NAME$_impl #(
    int  BIT_WIDTH,
    int  SIMD,
    int  MMV_IN,
    int  MMV_OUT,
    int unsigned  CNTR_BITWIDTH,
    int unsigned  INCR_BITWIDTH,

    int  LAST_READ_ELEM = $LAST_READ_ELEM$,
    int  LAST_WRITE_ELEM = $LAST_WRITE_ELEM$,
    int  BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$,
    int  ELEM_PER_WINDOW = $ELEM_PER_WINDOW$
)(
    input   logic  ap_clk,
    input   logic  ap_rst_n,

    input   logic  in0_V_V_TVALID,
    output  logic  in0_V_V_TREADY,
    input   logic [BIT_WIDTH * SIMD * MMV_IN-1:0]  in0_V_V_TDATA,

    output  logic  out_V_V_TVALID,
    input   logic  out_V_V_TREADY,
    output  logic [BIT_WIDTH * SIMD * MMV_OUT-1:0]  out_V_V_TDATA,

    input logic                     cfg_valid,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_simd,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_kw,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_kh,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_w,
    input logic [CNTR_BITWIDTH-1:0] cfg_cntr_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_simd,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_kw,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_kh,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_w,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_head_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_w,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_h,
    input logic [INCR_BITWIDTH-1:0] cfg_incr_tail_last,
    input logic [31:0]              cfg_last_read,
    input logic [31:0]              cfg_last_write
);
    // derived constants
    localparam int unsigned  BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
    localparam int unsigned  BUF_OUT_ELEM_WIDTH = BIT_WIDTH * SIMD;
    localparam int unsigned  BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;

    // (dynamic) configuration registers
    logic [31:0] Cfg_last_read  = LAST_READ_ELEM;
    logic [31:0] Cfg_last_write = LAST_WRITE_ELEM;

    // configuration reset/set logic
    always_ff @ (posedge ap_clk) begin
        if(cfg_valid) begin
            Cfg_last_read  <= cfg_last_read;
            Cfg_last_write <= cfg_last_write;
        end
    end

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
    $TOP_MODULE_NAME$_controller #(
        .CNTR_BITWIDTH(CNTR_BITWIDTH),
        .INCR_BITWIDTH(INCR_BITWIDTH)
    ) controller_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .advance(advance_controller),
        .addr_incr(addr_incr),
        .tail_incr(tail_incr),

        .cfg_valid(cfg_valid),
        .cfg_cntr_simd(cfg_cntr_simd),
        .cfg_cntr_kw(cfg_cntr_kw),
        .cfg_cntr_kh(cfg_cntr_kh),
        .cfg_cntr_w(cfg_cntr_w),
        .cfg_cntr_h(cfg_cntr_h),
        .cfg_incr_head_simd(cfg_incr_head_simd),
        .cfg_incr_head_kw(cfg_incr_head_kw),
        .cfg_incr_head_kh(cfg_incr_head_kh),
        .cfg_incr_head_w(cfg_incr_head_w),
        .cfg_incr_head_h(cfg_incr_head_h),
        .cfg_incr_tail_w(cfg_incr_tail_w),
        .cfg_incr_tail_h(cfg_incr_tail_h),
        .cfg_incr_tail_last(cfg_incr_tail_last)
    );

    // Counters/address registers
    // Add a sign bit even to (most) unsigned counters and Window_buffer_read_addr_reg,
    // so we can use automatic sign extension and simplify calculations w/ signed increment.
    // Alternatively, we could manually sign-extend and shave off a bit here or there.
    logic signed [$clog2(LAST_READ_ELEM+1)+1-1:0]  Newest_buffered_elem = -1;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  Current_elem = 0;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  First_elem_next_window = 0;
    logic        [$clog2(ELEM_PER_WINDOW)   -1:0]  Position_in_window = 0;
    logic        [$clog2(BUF_ELEM_TOTAL)+1  -1:0]  Window_buffer_read_addr_reg = 0;
    logic        [$clog2(BUF_ELEM_TOTAL)-1:0]      Window_buffer_write_addr_reg = 0;

    // Control signals/registers
    logic  Write_cmd    = 0;
    logic  Writing_done = 0;
    uwire  write_ok      = Write_cmd &&  out_V_V_TREADY;
    uwire  write_blocked = Write_cmd && !out_V_V_TREADY;

    logic  Fetching_done = 0;
    uwire  fetch_cmd = !($signed(Current_elem) > Newest_buffered_elem) && !write_blocked && !Fetching_done;

    uwire  reading_done = Newest_buffered_elem == Cfg_last_read;
    uwire  read_cmd =
        !reading_done && ( // if there is still an input element left to read
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
            Current_elem <= 0;
            First_elem_next_window <= 0;
            Position_in_window <= 0;
            Window_buffer_read_addr_reg <= 0;
            Window_buffer_write_addr_reg <= 0;
            Fetching_done <= 0;
            Write_cmd <= 0;
            Writing_done <= 0;
        end
        else begin
            if (read_ok) begin
                Window_buffer_write_addr_reg <= (Window_buffer_write_addr_reg == BUF_ELEM_TOTAL-1)? 0 : Window_buffer_write_addr_reg + 1;
                Newest_buffered_elem <= Newest_buffered_elem+1;

                if (Newest_buffered_elem == Cfg_last_read-1) begin
                    Window_buffer_write_addr_reg <= 0;
                end
                //check if this is the last read cycle (reading_done will be true afterwards)
                if ((Newest_buffered_elem == Cfg_last_read-1) && Writing_done) begin
                    //start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                    //todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                    Newest_buffered_elem <= -1;
                    Current_elem <= 0;
                    Window_buffer_read_addr_reg <= 0;
                    First_elem_next_window <= 0;
                    Writing_done <= 0;
                    Fetching_done <= 0;
                end
            end

            if (fetch_cmd) begin
                //count up to track which element index is about to be read from the buffer, and where it is located within the buffer
                //use increment value calculated by controller

                // absolute buffer address wrap-around
                automatic logic signed [$clog2(BUF_ELEM_TOTAL)+1:0]  ra = $signed(Window_buffer_read_addr_reg) + $signed(addr_incr);
                automatic logic signed [$clog2(BUF_ELEM_TOTAL+1):0]  ra_correct =
                    (ra >= BUF_ELEM_TOTAL)? -BUF_ELEM_TOTAL :
                    (ra <               0)?  BUF_ELEM_TOTAL : 0;
                Window_buffer_read_addr_reg <= ra + ra_correct;

                //keep track where we are within a window
                Position_in_window <= (Position_in_window != ELEM_PER_WINDOW - 1)? Position_in_window+1 : 0;

                //update first element of next window to allow buffer overwrite up until that point
                if (Position_in_window == 0)
                    First_elem_next_window <= First_elem_next_window + tail_incr;

                //check if this is the last write cycle (Writing_done will be true afterwards)
                if (Current_elem == Cfg_last_write)
                    Fetching_done <= 1;
                else
                    Current_elem <= $signed(Current_elem) + addr_incr;

                // determine if prefetched data will be outstanding in the next cycle
                // if we fetch in this cycle -> yes
                // if we do not fetch nor write -> do not change
                // if we do not fetch but write successfully-> clear outstanding data
                Write_cmd <= fetch_cmd;
            end

            if (write_ok)
                Write_cmd <= fetch_cmd;

            if (write_ok && Fetching_done) begin
                //check if this is the last write cycle (Writing_done will be true afterwards)
                if (reading_done || (read_ok && (Newest_buffered_elem == Cfg_last_read - 1))) begin
                    //start processing of next FM if reading is done already, or completes in the same cycle
                    Newest_buffered_elem <= -1;
                    Current_elem <= 0;
                    Window_buffer_read_addr_reg <= 0;
                    First_elem_next_window <= 0;
                    Fetching_done <= 0;
                end else
                    Writing_done <= 1;
            end
        end
    end

endmodule : $TOP_MODULE_NAME$_impl

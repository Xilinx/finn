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
module $TOP_MODULE_NAME$_controller #(
    int unsigned  LOOP_H_ITERATIONS    = $LOOP_H_ITERATIONS$,
    int unsigned  LOOP_W_ITERATIONS    = $LOOP_W_ITERATIONS$,
    int unsigned  LOOP_KH_ITERATIONS   = $LOOP_KH_ITERATIONS$,
    int unsigned  LOOP_KW_ITERATIONS   = $LOOP_KW_ITERATIONS$,
    int unsigned  LOOP_SIMD_ITERATIONS = $LOOP_SIMD_ITERATIONS$,

    int unsigned  INCR_BITWIDTH = $INCR_BITWIDTH$,

    bit IS_DEPTHWISE = $IS_DEPTHWISE$
)(
    input   logic  clk,
    input   logic  rst_n,

    input   logic  advance,
    output  logic [INCR_BITWIDTH-1:0]  addr_incr,
    output  logic [INCR_BITWIDTH-1:0]  tail_incr
);

    // state and counters
    typedef enum logic [2:0] {
        STATE_START,
        STATE_LOOP_SIMD,
        STATE_LOOP_KW,
        STATE_LOOP_KH,
        STATE_LOOP_W,
        STATE_LOOP_H
    }  state_e;
    state_e  State = $INNERMOST_STATE$;
    state_e  state_next;

    logic signed [$clog2(LOOP_H_ITERATIONS   +2)+1-1:0]  Counter_loop_h    = LOOP_H_ITERATIONS;
    logic signed [$clog2(LOOP_W_ITERATIONS   +2)+1-1:0]  Counter_loop_w    = LOOP_W_ITERATIONS;
    logic signed [$clog2(LOOP_KH_ITERATIONS  +2)+1-1:0]  Counter_loop_kh   = LOOP_KH_ITERATIONS;
    logic signed [$clog2(LOOP_KW_ITERATIONS  +2)+1-1:0]  Counter_loop_kw   = LOOP_KW_ITERATIONS;
    logic signed [$clog2(LOOP_SIMD_ITERATIONS+2)+1-1:0]  Counter_loop_simd = LOOP_SIMD_ITERATIONS;

    // combinational logic for addr_incr generation
    always_comb begin : blkHead
        unique case (State)
            0 : addr_incr = 0;
            1 : addr_incr = $HEAD_INCR_SIMD$;
            2 : addr_incr = $HEAD_INCR_KW$;
            3 : addr_incr = $HEAD_INCR_KH$;
            4 : addr_incr = $HEAD_INCR_W$;
            5 : addr_incr = $HEAD_INCR_H$;
        endcase
    end

    // combinational logic for tail_incr generation
    uwire  tail_incr_inner_condition = IS_DEPTHWISE? (Counter_loop_kh >= 0) : 0;
    assign tail_incr =
        tail_incr_inner_condition? 1 :
        Counter_loop_w >= 0?       $TAIL_INCR_W$ :
        Counter_loop_h >= 0?       $TAIL_INCR_H$ :
        /* else */                 $TAIL_INCR_LAST$;

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
            Counter_loop_h    <= LOOP_H_ITERATIONS;
            Counter_loop_w    <= LOOP_W_ITERATIONS;
            Counter_loop_kh   <= LOOP_KH_ITERATIONS;
            Counter_loop_kw   <= LOOP_KW_ITERATIONS;
            Counter_loop_simd <= LOOP_SIMD_ITERATIONS;
        end
        else if(advance) begin
            State <= state_next;
            if (State == $INNERMOST_STATE$) begin
                if(Counter_loop_simd >= 0)  Counter_loop_simd <= Counter_loop_simd-1;
                else begin
                    Counter_loop_simd <= LOOP_SIMD_ITERATIONS;
                    if(Counter_loop_kw >= 0)  Counter_loop_kw <= Counter_loop_kw-1;
                    else begin
                        Counter_loop_kw <= LOOP_KW_ITERATIONS;
                        if(Counter_loop_kh >= 0)  Counter_loop_kh <= Counter_loop_kh-1;
                        else begin
                            Counter_loop_kh <= LOOP_KH_ITERATIONS;
                            if(Counter_loop_w >= 0)  Counter_loop_w <= Counter_loop_w-1;
                            else begin
                                Counter_loop_w <= LOOP_W_ITERATIONS;
                                if(Counter_loop_h >= 0)  Counter_loop_h <= Counter_loop_h-1;
                                else  Counter_loop_h <= LOOP_H_ITERATIONS;
                            end
                        end
                    end
                end
            end
        end
    end

endmodule :  $TOP_MODULE_NAME$_controller

module $TOP_MODULE_NAME$_reg_buffer
#(
    parameter WIDTH = 1,
    parameter DEPTH = 1
)
(
    CLK,
    shift_enable,
    shift_in,
    shift_out,
    data_out
);

input CLK, shift_enable;
input [WIDTH-1:0] shift_in;
output [WIDTH-1:0] shift_out;
output [WIDTH*DEPTH-1:0] data_out;

reg [WIDTH-1:0] data [DEPTH-1:0];

assign shift_out = data[DEPTH-1];

for (genvar e=0; e<DEPTH; e=e+1)
    assign data_out[e*WIDTH +: WIDTH] = data[e];

always @ (posedge CLK) begin
    if (shift_enable) begin
        for (integer i=DEPTH-1; i>0; i=i-1)
            data[i] <= data[i-1];
        data[0] <= shift_in;
    end
end
endmodule : $TOP_MODULE_NAME$_reg_buffer

module $TOP_MODULE_NAME$_ram_buffer
#(
    parameter WIDTH = 1,
    parameter DEPTH = 1
)
(
    CLK,
    RST,
    shift_enable,
    shift_in,
    shift_out
);

input CLK, RST, shift_enable;
input [WIDTH-1:0] shift_in;
output [WIDTH-1:0] shift_out;

reg [WIDTH-1:0] out_reg;
assign shift_out = out_reg;

integer addr_w, addr_r; //TODO: minimize width + simplify

$RAM_STYLE$ reg [WIDTH-1:0] ram [DEPTH-1:0];

always @(posedge CLK) begin
    if (RST == 1'b0) begin
        addr_w <= 0;
        addr_r <= 1;
    end else begin
        if (shift_enable) begin
            ram[addr_w] <= shift_in;
            out_reg <= ram[addr_r];

            if (addr_w == DEPTH-1)
                addr_w <= 0;
            else
                addr_w <= addr_w + 1;

            if (addr_r == DEPTH-1)
                addr_r <= 0;
            else
                addr_r <= addr_r + 1;
        end
    end
end
endmodule : $TOP_MODULE_NAME$_ram_buffer

module $TOP_MODULE_NAME$_wb
#(
    parameter IN_WIDTH = 1, //bit-width*C*MMV_in
    parameter OUT_ELEM_WIDTH = 1, //bit-width*C
    parameter OUT_WIDTH = 1, //bit-width*C*MMV_out
    parameter BUFFER_ELEM_TOTAL = 1
)
(
    CLK,
    RST,
    data_in,
    shift_enable,
    data_out
);

input CLK, RST;
input [IN_WIDTH-1:0] data_in;
input shift_enable;
output [OUT_WIDTH-1:0] data_out;

$GENERATE_REG_FIFOS$

$GENERATE_BRAM_FIFOS$

//Fixed interconnect between linear buffers
$GENERATE_BUFFER_CONNECTION$

//Fixed REG FIFO <-> output mapping
$GENERATE_OUTPUT_MAPPING$


endmodule : $TOP_MODULE_NAME$_wb

module $TOP_MODULE_NAME$_impl #(
    int  BIT_WIDTH,
    int  SIMD,
    int  MMV_IN,
    int  MMV_OUT,
    int  LAST_READ_ELEM = $LAST_READ_ELEM$,
    int  FIRST_WRITE_ELEM = $FIRST_WRITE_ELEM$,
    int  LAST_WRITE_ELEM = $LAST_WRITE_ELEM$,
    int  BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$,
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

    //main buffer instantiation
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

    //controller instantiation
    uwire  advance_controller;
    uwire signed [INCR_BITWIDTH-1:0]  addr_incr;
    uwire        [INCR_BITWIDTH-1:0]  tail_incr;
    $TOP_MODULE_NAME$_controller controller_inst (
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
    logic signed [$clog2(LAST_READ_ELEM+1)+1-1:0]  Newest_buffered_elem = -1;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  Current_elem = FIRST_WRITE_ELEM;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  First_elem_next_window = 0;

    // Control signals/registers
    logic  Writing_done = 0;
    logic  write_done   = 0;

    uwire  write_ok          = write_cmd && (out_V_V_TREADY || write_done);
    uwire  write_blocked     = write_cmd && !out_V_V_TREADY && !write_done;

    uwire  write_cmd =   !($signed(Current_elem) > Newest_buffered_elem)                   && !Writing_done;;

    uwire  reading_done = Newest_buffered_elem == LAST_READ_ELEM;
    uwire  read_cmd =
        !reading_done && ( // if there is still an input element left to read
            Writing_done || ( // if fetching is done (e.g. for skipped rows at FM end due to stride)
                $signed(((Newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(First_elem_next_window) &&
                $signed(((Newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(Current_elem)
            ) // (over-)write to buffer if oldest buffered element will no longer be needed
        );
    uwire  read_ok      = read_cmd && in0_V_V_TVALID && !write_blocked;

    //            includes waiting on W    if W-only cycle: wait only on W     no R/W to wait for
    uwire advance =      read_ok        ||   (!read_cmd && write_ok)    || (!read_cmd && !write_cmd);

    //assign buffer control
    assign window_buffer_shift_enable = advance;
    assign  advance_controller = write_ok;

    //assign I/O ports
    assign  window_buffer_in = in0_V_V_TDATA;
    assign  out_V_V_TDATA = window_buffer_out;
    assign  in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
    assign  out_V_V_TVALID = ap_rst_n && write_cmd && !write_done; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

    //write done logic
    always_ff @(posedge ap_clk) begin
        if (advance) begin
            write_done <= 1'b0; //reset flag
        end else if (write_ok) // successful W in this cycle, but R still outstanding
            write_done <= 1'b1; //write can happen even if read is blocked, but only for the current cycle!
    end

    //main process for advancing counters
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

                //check if this is the last read cycle (reading_done will be true afterwards)
                if ((Newest_buffered_elem == LAST_READ_ELEM-1) && Writing_done) begin
                    //start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                    //todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                    Newest_buffered_elem <= -1;
                    Current_elem <= FIRST_WRITE_ELEM;
                    First_elem_next_window <= 0;
                    Writing_done <= 0;
                end
            end

            if (write_ok) begin
                First_elem_next_window <= First_elem_next_window + tail_incr;

                //check if this is the last write cycle (Writing_done will be true afterwards)
                if (Current_elem == LAST_WRITE_ELEM) begin
                    Writing_done <= 1;

                    if (reading_done || (read_ok && (Newest_buffered_elem == LAST_READ_ELEM - 1))) begin
                        //start processing of next FM if reading is done already, or completes in the same cycle
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

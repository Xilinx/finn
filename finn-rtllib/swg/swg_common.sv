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


// loop controller used for both, "default" and "parallel", implementation styles
module swg_controller
import swg::*; #(
    int unsigned  LOOP_H_ITERATIONS,
    int unsigned  LOOP_W_ITERATIONS,
    int unsigned  LOOP_KH_ITERATIONS,
    int unsigned  LOOP_KW_ITERATIONS,
    int unsigned  LOOP_SIMD_ITERATIONS,

    int unsigned  INCR_BITWIDTH,

    bit IS_DEPTHWISE,

    int HEAD_INCR_SIMD,
    int HEAD_INCR_KW,
    int HEAD_INCR_KH,
    int HEAD_INCR_W,
    int HEAD_INCR_H,
    int TAIL_INCR_W,
    int TAIL_INCR_H,
    int TAIL_INCR_LAST,

    state_e INNERMOST_STATE
)(
    input   logic  clk,
    input   logic  rst_n,

    input   logic  advance,
    output  logic [INCR_BITWIDTH-1:0]  addr_incr,
    output  logic [INCR_BITWIDTH-1:0]  tail_incr
);

    // state and counters
    state_e  State = INNERMOST_STATE;
    state_e  state_next;

    logic signed [$clog2(LOOP_H_ITERATIONS   +2)+1-1:0]  Counter_loop_h    = LOOP_H_ITERATIONS;
    logic signed [$clog2(LOOP_W_ITERATIONS   +2)+1-1:0]  Counter_loop_w    = LOOP_W_ITERATIONS;
    logic signed [$clog2(LOOP_KH_ITERATIONS  +2)+1-1:0]  Counter_loop_kh   = LOOP_KH_ITERATIONS;
    logic signed [$clog2(LOOP_KW_ITERATIONS  +2)+1-1:0]  Counter_loop_kw   = LOOP_KW_ITERATIONS;
    logic signed [$clog2(LOOP_SIMD_ITERATIONS+2)+1-1:0]  Counter_loop_simd = LOOP_SIMD_ITERATIONS;

    // combinational logic for addr_incr generation
    always_comb begin : blkHead
        unique case (State)
            STATE_START     : addr_incr = 0;
            STATE_LOOP_SIMD : addr_incr = HEAD_INCR_SIMD;
            STATE_LOOP_KW   : addr_incr = HEAD_INCR_KW;
            STATE_LOOP_KH   : addr_incr = HEAD_INCR_KH;
            STATE_LOOP_W    : addr_incr = HEAD_INCR_W;
            STATE_LOOP_H    : addr_incr = HEAD_INCR_H;
        endcase
    end

    // combinational logic for tail_incr generation
    uwire  tail_incr_inner_condition = IS_DEPTHWISE? (Counter_loop_kh >= 0) : 0;
    assign tail_incr =
        tail_incr_inner_condition? 1 :
        Counter_loop_w >= 0?       TAIL_INCR_W :
        Counter_loop_h >= 0?       TAIL_INCR_H :
        /* else */                 TAIL_INCR_LAST;

    // combinational next state logic
    always_comb begin : blkState
        state_next = State;
        if(State != INNERMOST_STATE)  state_next = INNERMOST_STATE;
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
            State <= INNERMOST_STATE;
            Counter_loop_h    <= LOOP_H_ITERATIONS;
            Counter_loop_w    <= LOOP_W_ITERATIONS;
            Counter_loop_kh   <= LOOP_KH_ITERATIONS;
            Counter_loop_kw   <= LOOP_KW_ITERATIONS;
            Counter_loop_simd <= LOOP_SIMD_ITERATIONS;
        end
        else if(advance) begin
            State <= state_next;
            if (State == INNERMOST_STATE) begin
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

endmodule :  swg_controller

// buffer used in "default" implementation style
module swg_cyclic_buffer_addressable #(
    int unsigned  WIDTH,
    int unsigned  DEPTH,
    parameter RAM_STYLE = "auto"
)(
    input   logic  clk,

    input   logic  write_enable,
    input   logic [$clog2(DEPTH)-1:0] write_addr,
    input   logic [WIDTH-1:0]  data_in,

    input   logic  read_enable,
    input   logic [$clog2(DEPTH)-1:0]  read_addr, // absolute (!) read address of cyclic buffer
    output  logic [WIDTH-1:0]  data_out
);

    (*ram_style=RAM_STYLE*) logic [WIDTH-1:0] Ram[DEPTH];
    logic [WIDTH-1:0]  Out = 'x;
    always_ff @(posedge clk) begin
        if (read_enable)  Out <= Ram[read_addr];
        if (write_enable) Ram[write_addr] <= data_in;
    end
    assign  data_out = Out;

endmodule : swg_cyclic_buffer_addressable

// buffer used in "parallel" implementation style
module swg_reg_buffer
#(
    int unsigned WIDTH = 1,
    int unsigned DEPTH = 1
)
(
    input logic clk,
    input logic shift_enable,
    input logic [WIDTH-1:0] shift_in,
    output logic [WIDTH-1:0] shift_out,
    output logic [WIDTH*DEPTH-1:0] data_out
);

logic [WIDTH-1:0] Data [DEPTH-1:0];

assign shift_out = Data[DEPTH-1];

for (genvar e=0; e<DEPTH; e++)
    assign data_out[e*WIDTH +: WIDTH] = Data[e];

always @ (posedge clk) begin
    if (shift_enable) begin
        if (DEPTH > 1) Data[DEPTH-1:1] <= Data[DEPTH-2:0];
        Data[0] <= shift_in;
    end
end
endmodule : swg_reg_buffer

// buffer used in "parallel" implementation style
module swg_ram_buffer
#(
    int unsigned WIDTH,
    int unsigned DEPTH,
    parameter RAM_STYLE = "auto"
)
(
    input logic clk,
    input logic rst_n,
    input logic shift_enable,
    input logic [WIDTH-1:0] shift_in,
    output logic [WIDTH-1:0] shift_out
);

logic [WIDTH-1:0] Out_reg;
assign shift_out = Out_reg;

logic [$clog2(DEPTH)-1:0] Addr_w = 0;
logic [$clog2(DEPTH)-1:0] Addr_r = 0;

(*ram_style=RAM_STYLE*) logic [WIDTH-1:0] Ram [DEPTH-1:0];

always_ff @(posedge clk) begin
    if (rst_n == 1'b0) begin
        Addr_w <= 0;
        Addr_r <= 1;
    end else begin
        if (shift_enable) begin
            Ram[Addr_w] <= shift_in;
            Out_reg <= Ram[Addr_r];

            if (Addr_w == DEPTH-1)
                Addr_w <= 0;
            else
                Addr_w <= Addr_w + 1;

            if (Addr_r == DEPTH-1)
                Addr_r <= 0;
            else
                Addr_r <= Addr_r + 1;
        end
    end
end
endmodule : swg_ram_buffer

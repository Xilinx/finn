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

module krnl_counter  #(
  int unsigned C_WIDTH,
  logic [C_WIDTH-1:0] C_INIT = '0
)
(
  input  logic               aclk,
  input  logic               clken,
  input  logic               aresetn,
  input  logic               load,
  input  logic               incr,
  input  logic               decr,
  input  logic [C_WIDTH-1:0] load_value,
  output logic [C_WIDTH-1:0] count,
  output logic               is_zero
);

  localparam [C_WIDTH-1:0] LP_ZERO = {C_WIDTH{1'b0}};
  localparam [C_WIDTH-1:0] LP_ONE = {{C_WIDTH-1{1'b0}},1'b1};
  localparam [C_WIDTH-1:0] LP_MAX = {C_WIDTH{1'b1}};

  logic [C_WIDTH-1:0] count_r = C_INIT;
  logic   is_zero_r = (C_INIT == LP_ZERO);

  assign count = count_r;

  always_ff @(posedge aclk) begin
    if (~aresetn) begin
      count_r <= C_INIT;
    end
    else if (clken) begin
      if (load) begin
        count_r <= load_value;
      end
      else if (incr & ~decr) begin
        count_r <= count_r + 1'b1;
      end
      else if (~incr & decr) begin
        count_r <= count_r - 1'b1;
      end
      else
        count_r <= count_r;
    end
  end

  assign is_zero = is_zero_r;

   always_ff @(posedge aclk) begin
    if (~aresetn) begin
      is_zero_r <= (C_INIT == LP_ZERO);
    end
    else if (clken) begin
      if (load) begin
        is_zero_r <= (load_value == LP_ZERO);
      end
      else begin
        is_zero_r <= incr ^ decr ? (decr && (count_r == LP_ONE)) || (incr && (count_r == LP_MAX)) : is_zero_r;
      end
    end
    else begin
      is_zero_r <= is_zero_r;
    end
  end


endmodule : krnl_counter

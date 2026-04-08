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

module ram_p_c #(
    int unsigned ADDR_BITS,
    int unsigned DATA_BITS,
    parameter  RAM_STYLE = "block"
) (
    input  logic                          clk,
    input  logic                          a_en,
    input  logic [(DATA_BITS/8)-1:0]      a_we,
    input  logic [ADDR_BITS-1:0]          a_addr,
    input  logic                          b_en,
    input  logic [ADDR_BITS-1:0]          b_addr,
    input  logic [DATA_BITS-1:0]          a_data_in,
    output logic [DATA_BITS-1:0]          a_data_out,
    output logic [DATA_BITS-1:0]          b_data_out
);

  localparam int unsigned DEPTH = 2**ADDR_BITS;

  (* ram_style = RAM_STYLE *) logic [DATA_BITS-1:0] ram[DEPTH];

  logic [DATA_BITS-1:0] a_data_reg = 0;
  logic [DATA_BITS-1:0] b_data_reg = 0;

  always_ff @(posedge clk) begin
    if(a_en) begin
      for (int i = 0; i < (DATA_BITS/8); i++) begin
        if(a_we[i]) begin
          ram[a_addr][(i*8)+:8] <= a_data_in[(i*8)+:8];
        end
      end
      a_data_reg <= ram[a_addr];
      a_data_out <= a_data_reg;
    end
    if(b_en) begin
      b_data_reg <= ram[b_addr];
      b_data_out <= b_data_reg;
    end
   //end
   end

endmodule : ram_p_c

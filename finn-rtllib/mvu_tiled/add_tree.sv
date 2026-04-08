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

module add_tree #(
    parameter                                               CHAINLEN,
    parameter                                               ACCU_WIDTH,
    parameter                                               TREE_HEIGHT
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [CHAINLEN-1:0][ACCU_WIDTH-1:0]             idat,

    input  logic [ACCU_WIDTH-1:0]                           iacc,

    output logic [ACCU_WIDTH-1:0]                           odat
);

//-------------------- Adder tree

function automatic int level_len (int lvl);
    return (CHAINLEN + (1 << lvl) - 1) >> lvl; // ceil(CHAINLEN / 2^lvl)
endfunction

logic signed [ACCU_WIDTH-1:0] add_sf;

if(CHAINLEN == 1) begin
    assign add_sf = idat[0];
end
else begin
    logic signed [TREE_HEIGHT:0][CHAINLEN-1:0][ACCU_WIDTH-1:0] add_s;

    for(genvar i = 0; i < CHAINLEN; i++) begin
        assign add_s[0][i] = signed'(idat[i]);
    end

    /*
    always_ff @(posedge clk) begin
        if(rst) begin
            for(int i = 1; i <= TREE_HEIGHT; i++) begin
                add_s[i] <= '0;
            end
        end
        else begin
            if(en) begin
                for(int i = 0; i < TREE_HEIGHT; i++) begin
                    for(int j = 0; j < (CHAINLEN/2 + (2**i-1))/(2**i); j++) begin
                        add_s[i+1][j] <= $signed(add_s[i][2*j+0]) + $signed(add_s[i][2*j+1]);
                    end
                end
            end
        end
    end
    */

    always_ff @(posedge clk) begin
        if (rst) begin
            // Clear all levels (safe for unused slots too)
            for (int i = 1; i <= TREE_HEIGHT; i++) begin
                for (int j = 0; j < CHAINLEN; j++) begin
                    add_s[i][j] <= '0;
                end
            end
        end else if (en) begin
            // For each level i, produce next level i+1
            for (int i = 0; i < TREE_HEIGHT; i++) begin
                int src_len = level_len(i);       // live elems at level i
                int dst_len = level_len(i + 1);   // live elems at level i+1 (ceil(src_len/2))

                // Compute valid outputs only (0..dst_len-1). Leave rest zero.
                for (int j = 0; j < dst_len; j++) begin
                    int a_idx = 2*j;
                    int b_idx = 2*j + 1;

                    // Cases:
                    //  - both indices in range  -> sum
                    //  - only a_idx in range    -> pass-through
                    //  - neither in range       -> zero (shouldn't happen for j<dst_len)
                    if (b_idx < src_len) begin
                        add_s[i+1][j] <= $signed(add_s[i][a_idx]) + $signed(add_s[i][b_idx]);
                    end else if (a_idx < src_len) begin
                        add_s[i+1][j] <= $signed(add_s[i][a_idx]); // pass through the odd one
                    end else begin
                        add_s[i+1][j] <= '0;
                    end
                end

                // Optional: clear any unused slots above dst_len (keeps nets tidy)
                for (int j = dst_len; j < CHAINLEN; j++) begin
                    add_s[i+1][j] <= '0;
                end
            end
        end
    end

    assign add_sf = add_s[TREE_HEIGHT][0];
end 

logic signed [ACCU_WIDTH-1:0] odat_int = '0;

// Add ACC
always_ff @(posedge clk) begin
    if(rst) begin
        odat_int <= 'X;
    end
    else begin
        if(en) begin
            odat_int <= $signed(add_sf) + $signed(iacc);
        end
    end
end

assign odat = odat_int;

endmodule : add_tree
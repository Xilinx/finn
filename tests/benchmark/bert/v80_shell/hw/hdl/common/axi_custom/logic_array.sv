/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF    SUCH DAMAGE.
  */

module logic_array #(
    parameter integer                   N_STAGES = 1,
    parameter integer                   DATA_BITS = 32
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    input  wire [DATA_BITS-1:0]         s_tdata,
    output wire [DATA_BITS-1:0]         m_tdata
);

// ----------------------------------------------------------------------------------------------------------------------- 
// Register slices
// ----------------------------------------------------------------------------------------------------------------------- 
logic [N_STAGES:0][DATA_BITS-1:0] int_tdata;

assign int_tdata[0] = s_tdata;
assign m_tdata = int_tdata[N_STAGES];

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        for(int i = 1; i <= N_STAGES; i++) begin
            int_tdata[i] <= '0;
        end
    end
    else begin
        for(int i = 1; i <= N_STAGES; i++) begin
            int_tdata[i] <= int_tdata[i-1];
        end
    end
end

endmodule
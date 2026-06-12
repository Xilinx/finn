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

import iwTypes::*;

`include "axi_macros.svh"

module axi_hbm_reg_array #(
    parameter integer                       N_STAGES = 1,
    parameter integer                       ID_BITS = HBM_ID_BITS,
    parameter integer                       ADDR_BITS = HBM_ADDR_BITS,
    parameter integer                       DATA_BITS = HBM_DATA_BITS
) (
    input  logic                            aclk,
    input  logic                            aresetn,

    AXI4.slave                              s_axi,
    AXI4.master                             m_axi
);

// ----------------------------------------------------------------------------------------------------------------------- 
// Register slices
// ----------------------------------------------------------------------------------------------------------------------- 
AXI4 #(.AXI4_ID_BITS(ID_BITS), .AXI4_ADDR_BITS(ADDR_BITS), .AXI4_DATA_BITS(DATA_BITS)) axi_s [N_STAGES+1] ();

`AXI_ASSIGN(s_axi, axi_s[0])
`AXI_ASSIGN(axi_s[N_STAGES], m_axi)

for(genvar i = 0; i < N_STAGES; i++) begin
    axi_hbm_reg inst_reg (.aclk(aclk), .aresetn(aresetn), .s_axi(axi_s[i]), .m_axi(axi_s[i+1]));  
end

endmodule
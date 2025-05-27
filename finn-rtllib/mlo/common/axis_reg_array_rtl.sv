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

`include "axi_macros.svh"

module axis_reg_array_rtl #(
    parameter integer                   N_STAGES = 4,
    parameter integer                   DATA_BITS = 32
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    AXI4S.slave                         s_axis,
    AXI4S.master                        m_axis
);

// -----------------------------------------------------------------------------------------------------------------------
// Register slices
// -----------------------------------------------------------------------------------------------------------------------
AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) axis_s [N_STAGES+1] ();

`AXIS_ASSIGN(s_axis, axis_s[0])
`AXIS_ASSIGN(axis_s[N_STAGES], m_axis)

for(genvar i = 0; i < N_STAGES; i++) begin

    axis_reg_rtl #(
        .DATA_WIDTH(DATA_BITS),
        .LAST_ENABLE(0),
        .USER_ENABLE(0)
    ) inst_reg (
        .aclk(aclk),
        .aresetn(aresetn),

        .s_axis_tdata (axis_s[i].tdata),
        .s_axis_tkeep ('1),
        .s_axis_tvalid(axis_s[i].tvalid),
        .s_axis_tready(axis_s[i].tready),
        .s_axis_tlast ('0),
        .s_axis_tid   ('0),
        .s_axis_tdest ('0),
        .s_axis_tuser ('0),

        .m_axis_tdata (axis_s[i+1].tdata),
        .m_axis_tkeep (),
        .m_axis_tvalid(axis_s[i+1].tvalid),
        .m_axis_tready(axis_s[i+1].tready),
        .m_axis_tlast (),
        .m_axis_tid   (),
        .m_axis_tdest (),
        .m_axis_tuser ()
    );

end

endmodule

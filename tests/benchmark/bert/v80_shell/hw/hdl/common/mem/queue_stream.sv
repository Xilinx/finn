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

module queue_stream #(
    parameter type QTYPE = logic[63:0],
    parameter QDEPTH = 8
) (
    input  logic        aclk,
    input  logic        aresetn,

    input  logic        val_snk,
    output logic        rdy_snk,
    input  QTYPE        data_snk,

    output logic        val_src,
    input  logic        rdy_src,
    output QTYPE        data_src
);

logic val_rd;
logic rdy_rd;

fifo #(
    .DATA_BITS($bits(QTYPE)),
    .FIFO_SIZE(QDEPTH)
) inst_fifo (
    .aclk       (aclk),
    .aresetn    (aresetn),
    .rd         (val_rd),
    .wr         (val_snk),
    .ready_rd   (rdy_rd),
    .ready_wr   (rdy_snk),
    .data_in    (data_snk),
    .data_out   (data_src)
);

assign val_src = rdy_rd;
assign val_rd = val_src & rdy_src;

endmodule
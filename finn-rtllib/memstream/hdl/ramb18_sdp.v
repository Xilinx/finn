/*
 Copyright (c) 2020, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of FINN nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

module ramb18_sdp
#(
    parameter ID = 0,
    parameter DWIDTH = 18,
    parameter AWIDTH = 10,
    parameter DEPTH = 2**AWIDTH,
    parameter MEM_INIT = "",
    parameter RAM_STYLE = "auto"
)
(
	input clk,

	input ena,
	input wea,
	input [AWIDTH-1:0] addra,
	input [DWIDTH-1:0] wdataa,

    input enb,
    input enqb,
	input [AWIDTH-1:0] addrb,
	output reg [DWIDTH-1:0] rdqb
);

(* ram_style = RAM_STYLE *) reg [DWIDTH-1:0] mem[0:DEPTH-1];
reg [DWIDTH-1:0] rdatab;

`ifdef SYNTHESIS
reg [7:0] idx = ID;
`else
reg [15:0] idx;
`endif

//initialize memory
initial begin
  //note the hacky way of adding a filename memblock_ID.dat to the path provided in MEM_INIT
  //ID can go up to 99
  if (ID < 0 && ID > 99) begin
    $display("ID out of range [0-99]");
    $finish();
  end
	//MEM_INIT path must be terminated by /
  `ifdef SYNTHESIS
  if (ID < 10)
    $readmemh({MEM_INIT,"memblock_synth_",idx+8'd48,".dat"}, mem, 0, DEPTH-1);
  else
    $readmemh({MEM_INIT,"memblock_synth_",(idx/10)+8'd48,(idx%10)+8'd48,".dat"}, mem, 0, DEPTH-1);
  `else
  $sformat(idx,"%0d",ID);
  if (ID < 10)
    $readmemh({MEM_INIT,"memblock_sim_",idx[7:0],".dat"}, mem, 0, DEPTH-1);
  else
    $readmemh({MEM_INIT,"memblock_sim_",idx,".dat"}, mem, 0, DEPTH-1);
  `endif
end

//memory ports, with output pipeline register
always @(posedge clk) begin
    if(wea)
        mem[addra] <= wdataa;
    if(enb)
        rdatab <= mem[addrb];
    if(enqb)
        rdqb <= rdatab;
end

endmodule

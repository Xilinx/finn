// Copyright (c) 2022 Xilinx, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of Xilinx nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

`timescale 1 ns / 1 ps
`define HEXFILE "data.hex"

parameter MAX_FL =4000;


module tb ();

logic [28*8-1:0] data [MAX_FL];
logic [28*8-1:0] data_row;
logic [28*28*8-1:0] img_data;
logic [7:0] fifo [16];
logic [3:0] rd_ptr=0;
logic [3:0] wr_ptr=0;
int err_count=0;
int data_count=0;
int i,j;
logic [31:0] file_lines;

logic ap_clk = 0;
logic ap_rst_n = 0;

logic [7:0]dout_tdata;
logic dout_tlast;
logic dout_tready;
logic dout_tvalid;

logic [392-1:0]din_tdata;
logic din_tready;
logic din_tvalid;



finn_design_wrapper finn_design_wrapper (
  .ap_clk                (ap_clk               ),//i 
  .ap_rst_n              (ap_rst_n             ),//i 

  .m_axis_0_tdata        (dout_tdata           ),//o
  .m_axis_0_tready       (dout_tready          ),//i 
  .m_axis_0_tvalid       (dout_tvalid          ),//o 

  .s_axis_0_tdata        (din_tdata           ),//i
  .s_axis_0_tready       (din_tready          ),//o 
  .s_axis_0_tvalid       (din_tvalid          ) //i 
);

initial begin: AP_CLK
  forever begin
    ap_clk = #5 ~ap_clk;
  end
end


initial begin
  // Hex file formated for Upper N bits as input data, and lower N bits as expected output data
  
  $readmemh(`HEXFILE, data);
  // Determine how large file actuall is
  for (i=0; i<MAX_FL; i+=1)  if (data[i][0] !== 1'bx) file_lines = i;
  if (file_lines[0] === {1'bx}) begin
    $display("ERROR:  Unable to read hex file: %s",`HEXFILE);
    $finish;
  end
  

  din_tvalid = 0;
  din_tdata = 0;
  dout_tready = 1;

  repeat (100)  @(negedge ap_clk);
  ap_rst_n = 1;
  repeat (100)  @(negedge ap_clk);
  dout_tready = 1;

  repeat (10)  @(negedge ap_clk);
  //while (~din_tready) @(negedge ap_clk);
  @(negedge ap_clk);
  @(negedge ap_clk);

  // The hex file is formated in 29 row blocks
  //    The first 28 rows are the image data
  //    The 29th row is the goundtruth expected result stored in the lowest byte.
  // Note that each row's byte-order is saved such that the high-byte is in the upper
  // most bits, and the first byte in the lower-most bits.
  for (j=0; j<=file_lines; j+=1) begin
    if ((j%29) < 28) begin
      img_data[(j%29)*28*8+:28*8] = data[j];
    end else begin
      // Grab the verifcation result on the 29th row
      data_row = data[j];
      //$display("wr_ptr %h, data:%h,  j=%d",wr_ptr,data[j],j);
      fifo[wr_ptr] = data_row[7:0];
      wr_ptr++;
       
      // Due to folding factors, the 784 bytes of each image gets fed 49-bytes at a time 
      // over 16 cycles
      for (i=0; i<16; i+=1) begin
        din_tvalid = 1;
        din_tdata = img_data[392*i+:392];
        @(negedge ap_clk);
        while (~din_tready)  @(negedge ap_clk);
        din_tvalid = 0;
        //repeat (200) @(negedge ap_clk);
      end
    end
  end
  din_tdata = 0;
  din_tvalid = 0;

  repeat (1000)  @(negedge ap_clk);
  din_tdata = 0;
  if (wr_ptr != rd_ptr) begin
    $display("ERR: End-sim check: rd_ptr %h != %h wr_ptr",rd_ptr, wr_ptr);
    err_count++;
  end
    
  $display("\n************************************************************ ");
  $display("  SIM COMPLETE");
  $display("  Validated %0d data points ",data_count);
  $display("  Total error count: ====>  %0d  <====\n",err_count);
  $finish;
end


// Check the result at each valid output from the model
always @(posedge ap_clk) begin
  if (dout_tvalid && ap_rst_n) begin
    if (dout_tdata !== fifo[rd_ptr]) begin
      $display("ERR: Data mismatch %h != %h ",dout_tdata, fifo[rd_ptr]);
      err_count++;
    end else begin
      $display("CHK: Data    match %h == %h   --> %0d",dout_tdata, fifo[rd_ptr], data_count);
    end
    rd_ptr++;
    data_count++;
  end
end

endmodule





// Copyright (c) 2022 Advanced Micro Devices, Inc.
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
// * Neither the name of AMD nor the names of its
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
`define INPUT_HEXFILE "input.hex"
`define EXPECTED_OUTPUT_HEXFILE "expected_output.hex"

parameter N_SAMPLES = $N_SAMPLES$;
parameter IN_STREAM_BITWIDTH = $IN_STREAM_BITWIDTH$;
parameter OUT_STREAM_BITWIDTH = $OUT_STREAM_BITWIDTH$;
parameter IN_BEATS_PER_SAMPLE = $IN_BEATS_PER_SAMPLE$;
parameter OUT_BEATS_PER_SAMPLE = $OUT_BEATS_PER_SAMPLE$;
parameter TIMEOUT_CYCLES = $TIMEOUT_CYCLES$;

parameter IN_SAMPLE_BITWIDTH = IN_STREAM_BITWIDTH * IN_BEATS_PER_SAMPLE;
parameter OUT_SAMPLE_BITWIDTH = OUT_STREAM_BITWIDTH * OUT_BEATS_PER_SAMPLE;

module tb ();


logic [IN_STREAM_BITWIDTH-1:0] input_data [N_SAMPLES*IN_BEATS_PER_SAMPLE];
logic [OUT_STREAM_BITWIDTH-1:0] exp_output_data [N_SAMPLES*OUT_BEATS_PER_SAMPLE];
logic [IN_STREAM_BITWIDTH-1:0] current_input [IN_BEATS_PER_SAMPLE];
logic [OUT_STREAM_BITWIDTH-1:0] current_exp_output [OUT_BEATS_PER_SAMPLE];
logic [OUT_SAMPLE_BITWIDTH-1:0] exp_output_queue [N_SAMPLES];
logic [$clog2(N_SAMPLES):0] rd_ptr=0;
logic [$clog2(N_SAMPLES):0] wr_ptr=0;
int err_count=0;
int data_count=0;
int i,j;
logic [31:0] input_file_lines;
logic [31:0] exp_output_file_lines;

logic ap_clk = 0;
logic ap_rst_n = 0;

logic [OUT_STREAM_BITWIDTH-1:0] dout_tdata;
logic dout_tlast;
logic dout_tready;
logic dout_tvalid;

logic [IN_STREAM_BITWIDTH-1:0] din_tdata;
logic din_tready;
logic din_tvalid;



finn_design_wrapper finn_design_wrapper (
  .ap_clk                (ap_clk               ),
  .ap_rst_n              (ap_rst_n             ),

  .m_axis_0_tdata        (dout_tdata           ),
  .m_axis_0_tready       (dout_tready          ),
  .m_axis_0_tvalid       (dout_tvalid          ),

  .s_axis_0_tdata        (din_tdata           ),
  .s_axis_0_tready       (din_tready          ),
  .s_axis_0_tvalid       (din_tvalid          )
);

initial begin: AP_CLK
  forever begin
    ap_clk = #5 ~ap_clk;
  end
end


initial begin
    // read input hexfile
    $readmemh(`INPUT_HEXFILE, input_data);
    for (i=0; i<N_SAMPLES*IN_BEATS_PER_SAMPLE; i+=1)  if (input_data[i][0] !== 1'bx) input_file_lines = i;
    if (input_file_lines[0] === {1'bx}) begin
        $display("ERROR:  Unable to read hex file: %s",`INPUT_HEXFILE);
        $finish;
    end
    // read expected output hexfile
    $readmemh(`EXPECTED_OUTPUT_HEXFILE, exp_output_data);
    for (i=0; i<N_SAMPLES*OUT_BEATS_PER_SAMPLE; i+=1)  if (exp_output_data[i][0] !== 1'bx) exp_output_file_lines = i;
    if (exp_output_file_lines[0] === {1'bx}) begin
        $display("ERROR:  Unable to read hex file: %s",`EXPECTED_OUTPUT_HEXFILE);
        $finish;
    end

    din_tvalid = 0;
    din_tdata = 0;
    dout_tready = 1;

    // perform reset
    repeat (100)  @(negedge ap_clk);
    ap_rst_n = 1;
    repeat (100)  @(negedge ap_clk);
    dout_tready = 1;

    repeat (10)  @(negedge ap_clk);
    @(negedge ap_clk);
    @(negedge ap_clk);


    // feed all inputs
    for (j=0; j<N_SAMPLES; j+=1) begin
        // get current input and expected output samples from batch data
        for (i=0; i<IN_BEATS_PER_SAMPLE; i+=1) begin
            current_input[i] = input_data[j*IN_BEATS_PER_SAMPLE+i];
        end
        // put corresponding expected output into queue
        for (i=0; i<OUT_BEATS_PER_SAMPLE; i+=1) begin
            exp_output_queue[wr_ptr] = exp_output_data[j*OUT_BEATS_PER_SAMPLE+i];
            wr_ptr++;
        end
        // feed current input
        for (i=0; i<IN_BEATS_PER_SAMPLE; i+=1) begin
            din_tvalid = 1;
            din_tdata = current_input[i];
            @(negedge ap_clk);
            // TODO add timeout on input backpressure
            while (~din_tready)  @(negedge ap_clk);
            din_tvalid = 0;
        end
    end

    din_tdata = 0;
    din_tvalid = 0;

    repeat (TIMEOUT_CYCLES)  @(negedge ap_clk);
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
    // TODO implement output folding - current code assumes OUT_BEATS_PER_SAMPLE=1
    if (dout_tdata !== exp_output_queue[rd_ptr]) begin
      $display("ERR: Data mismatch %h != %h ",dout_tdata, exp_output_queue[rd_ptr]);
      err_count++;
    end else begin
      $display("CHK: Data    match %h == %h   --> %0d",dout_tdata, exp_output_queue[rd_ptr], data_count);
    end
    rd_ptr++;
    data_count++;
  end
end

endmodule

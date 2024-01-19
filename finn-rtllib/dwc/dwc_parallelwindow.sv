/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	StreamingDataWidthConverter
 *****************************************************************************/

module dwc_parallelwindow #(
    int unsigned    IN_WIDTH,
    int unsigned    OUT_WIDTH,
    int unsigned    SIMD,
    int unsigned    PE,
    int unsigned    CHANNELS,
    int unsigned    KERNEL_PROD,
    int unsigned    ACTIVATION_WIDTH,

    // Safely deducible parameters
    localparam int unsigned   NF = CHANNELS/PE,
    localparam int unsigned   SF = KERNEL_PROD/SIMD,
    localparam int unsigned   OUTPUT_STREAM_WIDTH = PE*SIMD*ACTIVATION_WIDTH,
    localparam int unsigned   TOTAL_ITERS = NF*SF
)
(
    // Global control
    input   logic  ap_clk,
    input   logic  ap_rst_n,

    // Input stream
    input   logic [IN_WIDTH-1:0]  s_axis_input_tdata,
    input   logic  s_axis_input_tvalid,
    output  logic  s_axis_input_tready,

    // Output stream
    output  logic  [OUT_WIDTH-1:0]  m_axis_output_tdata,
    output  logic  m_axis_output_tvalid,
    input   logic  m_axis_output_tready
);
    uwire rst = !ap_rst_n;
    
    typedef logic [$clog2(TOTAL_ITERS+1)-1:0]  count_t;
    count_t  Count = 0;
    logic [$clog2(NF)-1:0] Nf_count;
    localparam int unsigned SF_WIDTH = SF==1 ? 1 : $clog2(SF);
    logic [SF_WIDTH-1:0] Sf_count = '{ default : 0};
    if (SF==1) begin
        always_ff @(posedge ap_clk) begin
            if (rst) begin
                Count <= 0;
                Nf_count <= 0;
            end
            else if(en) begin
                if (buffer_load)  Count <= TOTAL_ITERS;
                else if (!buffer_empty) begin
                    Count <= Count - 1;
                    Nf_count <= (Nf_count == NF-1) ? 0 : Nf_count + 1;
                end
            end
        end
    end
    else begin
        always_ff @(posedge ap_clk) begin
            if (rst) begin
                Count <= 0;
                Nf_count <= 0;
                Sf_count <= 0;
            end
            else if(en) begin
                if (buffer_load)  Count <= TOTAL_ITERS;
                else if (!buffer_empty) begin
                    Count <= Count - 1;
                    Sf_count <= (Sf_count == SF-1) ? 0 : Sf_count + 1;
                    if (Sf_count == SF-1)  Nf_count <= (Nf_count == NF-1) ? 0 : Nf_count + 1;
                end
            end
        end
    end
   
    uwire  buffer_empty = Count==0; // flag to indicate no data loaded into Mem
    uwire  buffer_load = buffer_empty && s_axis_input_tvalid;
    uwire  en = m_axis_output_tready; // Only advance counter/output when output_ready=1
    assign  s_axis_input_tready = en && buffer_empty;
    
    typedef logic [ACTIVATION_WIDTH-1:0] input_t;
    input_t Mem [KERNEL_PROD-1:0][CHANNELS-1:0];
    for (genvar i = 0; i<KERNEL_PROD; i++) begin : genMemH
        for (genvar j = 0; j<CHANNELS; j++) begin : genMemW
            always_ff @(posedge ap_clk) begin
                if (en && buffer_load) begin
                    Mem[i][j] <= s_axis_input_tdata[(i*CHANNELS+j)*ACTIVATION_WIDTH +: ACTIVATION_WIDTH];
                end
            end
        end : genMemW
    end : genMemH

    typedef logic [ACTIVATION_WIDTH-1:0] output_t;
    output_t mem_i [SIMD-1:0][PE-1:0];
    // Count = TOTAL_ITERS ---> Count = 0, SIMD=0, PE=0
    // Count = TOTAL_ITERS-1 ---> Count = 1, SIMD=1, PE=0
    // Count = TOTAL_ITERS-2 ---> Count = 2, SIMD=2, PE=0
    // Count = TOTAL_ITERS-3 ---> Count = 3, SIMD=0, PE=1
    // Count = TOTAL_ITERS-4 ---> Count = 4, SIMD=1, PE=1
    // TODO: Add extra counters for simd and pe offset instead of deriving it from Count
    for (genvar i = 0; i<SIMD; i++) begin : genWirePE
        for (genvar j = 0; j<PE; j++) begin : genWireSIMD
            assign  mem_i[i][j] = Mem[i+SIMD*Sf_count][j+PE*Nf_count];
        end : genWireSIMD
    end : genWirePE

    for (genvar i = 0; i<SIMD; i++) begin : genOutSIMD
      for (genvar j = 0; j<PE; j++) begin : genOutPE
        assign  m_axis_output_tdata[(j+i*PE)*ACTIVATION_WIDTH +: ACTIVATION_WIDTH] = mem_i[i][j];      
      end
    end

    assign  m_axis_output_tvalid = !buffer_empty;

endmodule : dwc_parallelwindow
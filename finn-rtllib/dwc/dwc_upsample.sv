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

module dwc_upsample #(
    int unsigned    IN_WIDTH, // = IN_FOLD * ACTIVATION_WIDTH
    int unsigned    OUT_WIDTH, // = OUT_FOLD * ACTIVATION_WIDTH
    int unsigned    ACTIVATION_WIDTH,

    // Safely deducible parameters
    localparam int unsigned   TOTAL_ITERS = OUT_WIDTH / IN_WIDTH
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
    always_ff @(posedge ap_clk) begin
        if (rst) begin
            Count <= 0;
        end
        else if(en) begin
            if (buffer_empty && buffer_advance)  Count <= TOTAL_ITERS;
            else if (!buffer_empty && buffer_advance)  Count <= Count - 1;
        end
    end
   
    uwire  buffer_empty = Count==0; // flag to indicate no data loaded into Mem
    uwire  buffer_advance = s_axis_input_tvalid;
    uwire  en = m_axis_output_tready; // Only advance counter/output when output_ready=1
    assign  s_axis_input_tready = en && buffer_empty;
    
    typedef logic [ACTIVATION_WIDTH-1:0] input_t;
    input_t  Mem [TOTAL_ITERS-1:0];
    always_ff @(posedge ap_clk) begin
        if (en)  Mem[TOTAL_ITERS-Count] <= s_axis_input_tdata;
    end

    for (genvar i = 0; i < TOTAL_ITERS; i++) begin : genWire
        assign  m_axis_output_tdata[i*ACTIVATION_WIDTH +: ACTIVATION_WIDTH] = Mem[i];
    end : genWire

    assign  m_axis_output_tvalid = !buffer_empty;

endmodule : dwc_upsample
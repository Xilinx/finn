/*

Copyright (c) 2019 Alex Forencich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

// Language: Verilog 2001

/*
 * AXI4-Stream broadcaster
 */
module broadcast #(
    // Number of AXI stream outputs
    parameter M_COUNT = 2,
    // Width of AXI stream interfaces in bits
    parameter DATA_WIDTH = 8
) (
    input  wire                          ap_clk,
    input  wire                          ap_rst_n,

    /*
     * AXI input
     */
    input  wire [DATA_WIDTH-1:0]         s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,

    /*
     * AXI outputs
     */
    output wire [M_COUNT*DATA_WIDTH-1:0] m_axis_tdata,
    output wire [M_COUNT-1:0]            m_axis_tvalid,
    input  wire [M_COUNT-1:0]            m_axis_tready
);

parameter CL_M_COUNT = $clog2(M_COUNT);

// datapath registers
reg s_axis_tready_reg = 1'b0, s_axis_tready_next;

reg [DATA_WIDTH-1:0] m_axis_tdata_reg  = {DATA_WIDTH{1'b0}};
reg [M_COUNT-1:0]    m_axis_tvalid_reg = {M_COUNT{1'b0}}, m_axis_tvalid_next;

reg [DATA_WIDTH-1:0] temp_m_axis_tdata_reg  = {DATA_WIDTH{1'b0}};
reg                  temp_m_axis_tvalid_reg = 1'b0, temp_m_axis_tvalid_next;

// // datapath control
reg store_axis_input_to_output;
reg store_axis_input_to_temp;
reg store_axis_temp_to_output;

assign s_axis_tready = s_axis_tready_reg;

assign m_axis_tdata  = {M_COUNT{m_axis_tdata_reg}};
assign m_axis_tvalid = m_axis_tvalid_reg;

// enable ready input next cycle if output is ready or the temp reg will not be filled on the next cycle (output reg empty or no input)
wire s_axis_tready_early = ((m_axis_tready & m_axis_tvalid) == m_axis_tvalid) || (!temp_m_axis_tvalid_reg && (!m_axis_tvalid || !s_axis_tvalid));

always @* begin
    // transfer sink ready state to source
    m_axis_tvalid_next = m_axis_tvalid_reg & ~m_axis_tready;
    temp_m_axis_tvalid_next = temp_m_axis_tvalid_reg;

    store_axis_input_to_output = 1'b0;
    store_axis_input_to_temp = 1'b0;
    store_axis_temp_to_output = 1'b0;

    if (s_axis_tready_reg) begin
        // input is ready
        if (((m_axis_tready & m_axis_tvalid) == m_axis_tvalid) || !m_axis_tvalid) begin
            // output is ready or currently not valid, transfer data to output
            m_axis_tvalid_next = {M_COUNT{s_axis_tvalid}};
            store_axis_input_to_output = 1'b1;
        end else begin
            // output is not ready, store input in temp
            temp_m_axis_tvalid_next = s_axis_tvalid;
            store_axis_input_to_temp = 1'b1;
        end
    end else if ((m_axis_tready & m_axis_tvalid) == m_axis_tvalid) begin
        // input is not ready, but output is ready
        m_axis_tvalid_next = {M_COUNT{temp_m_axis_tvalid_reg}};
        temp_m_axis_tvalid_next = 1'b0;
        store_axis_temp_to_output = 1'b1;
    end
end

always @(posedge ap_clk) begin
    s_axis_tready_reg <= s_axis_tready_early;
    m_axis_tvalid_reg <= m_axis_tvalid_next;
    temp_m_axis_tvalid_reg <= temp_m_axis_tvalid_next;

    // datapath
    if (store_axis_input_to_output) begin
        m_axis_tdata_reg <= s_axis_tdata;
    end else if (store_axis_temp_to_output) begin
        m_axis_tdata_reg <= temp_m_axis_tdata_reg;
    end

    if (store_axis_input_to_temp) begin
        temp_m_axis_tdata_reg <= s_axis_tdata;
    end

    if (~ap_rst_n) begin
        s_axis_tready_reg <= 1'b0;
        m_axis_tvalid_reg <= {M_COUNT{1'b0}};
        temp_m_axis_tvalid_reg <= {M_COUNT{1'b0}};
    end
end

endmodule
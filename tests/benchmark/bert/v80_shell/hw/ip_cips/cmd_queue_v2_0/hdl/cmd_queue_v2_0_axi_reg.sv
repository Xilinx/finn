// (c) Copyright 2022, Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a 
// copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation 
// the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.
////////////////////////////////////////////////////////////

module cmd_queue_v2_0_0_axi_reg #(
  parameter int C_DATA_WIDTH       = 32,  // Data width
  parameter int C_ADDR_WIDTH       = 32   // Address width
) (
  // AXI4-Lite Subordinate Interface
  cmd_queue_v2_0_0_axi_if.sub             axi_if,

  // Clock/Reset
  input  logic                          aclk,
  input  logic                          aresetn,

  // Register Read Interface
  output logic                          reg_rd_valid_o,
  output logic [C_ADDR_WIDTH-1:0]       reg_rd_addr_o,
  input  logic                          reg_rd_done_i,
  input  logic [1:0]                    reg_rd_resp_i,
  input  logic [C_DATA_WIDTH-1:0]       reg_rd_data_i,

  // Register Write Interface
  output logic                          reg_wr_valid_o,
  output logic [C_ADDR_WIDTH-1:0]       reg_wr_addr_o,
  output logic [(C_DATA_WIDTH/8)-1:0]   reg_wr_be_o,
  output logic [C_DATA_WIDTH-1:0]       reg_wr_data_o,
  input  logic                          reg_wr_done_i,
  input  logic [1:0]                    reg_wr_resp_i
);

// --------------------------------------------------------
// Time Units/Precision
// --------------------------------------------------------
// synthesis translate_off
timeunit 1ns/1ps;
// synthesis translate_on

// --------------------------------------------------------
// Parameters
// --------------------------------------------------------

// --------------------------------------------------------
// Types
// --------------------------------------------------------

// --------------------------------------------------------
// Functions
// --------------------------------------------------------

// --------------------------------------------------------
// Variables/Nets
// --------------------------------------------------------
logic wr_rdy;
logic wr_wait;
logic rd_wait;

// ========================================================

// AXI Write
always_ff @(posedge aclk) begin
  if (!aresetn) begin
    wr_rdy         <= '0;
    wr_wait        <= '0;
    reg_wr_valid_o <= '0;
    reg_wr_addr_o  <= '0;
    reg_wr_be_o    <= '0;
    reg_wr_data_o  <= '0;
    axi_if.bvalid  <= '0;
    axi_if.bresp   <= '0;
  end else begin
    // Defaults
    wr_rdy         <= '0;
    wr_wait        <= wr_wait;
    reg_wr_valid_o <= '0;
    reg_wr_addr_o  <= reg_wr_addr_o;
    reg_wr_be_o    <= reg_wr_be_o;
    reg_wr_data_o  <= reg_wr_data_o;
    axi_if.bvalid  <= '0;
    axi_if.bresp   <= '0;
    // coverage off -item c 1 -feccondrow 4
    if (!wr_wait && !wr_rdy && axi_if.awvalid && axi_if.wvalid) begin
    // coverage on
      wr_rdy         <= 1'b1;
      wr_wait        <= 1'b1;
      reg_wr_valid_o <= 1'b1;
      reg_wr_addr_o  <= axi_if.awaddr;
      reg_wr_be_o    <= axi_if.wstrb;
      reg_wr_data_o  <= axi_if.wdata;
    end else if (reg_wr_done_i) begin
      axi_if.bvalid  <= 1'b1;
      axi_if.bresp   <= reg_wr_resp_i;
      reg_wr_addr_o  <= '0;
      reg_wr_be_o    <= '0;
      reg_wr_data_o  <= '0;
    end else if (axi_if.bvalid) begin
      if (!axi_if.bready) begin
        axi_if.bvalid <= axi_if.bvalid;
        axi_if.bresp  <= axi_if.bresp;
      end else begin
        wr_wait       <= 1'b0;
      end
    end
  end
end
assign axi_if.awready = wr_rdy;
assign axi_if.wready  = wr_rdy;

// AXI Read
always_ff @(posedge aclk) begin
  if (!aresetn) begin
    axi_if.arready  <= '0;
    rd_wait         <= '0;
    reg_rd_valid_o  <= '0;
    reg_rd_addr_o   <= '0;
    axi_if.rvalid   <= '0;
    axi_if.rresp    <= '0;
    axi_if.rdata    <= '0;
  end else begin
    // Defaults
    axi_if.arready  <= '0;
    rd_wait         <= rd_wait;
    reg_rd_valid_o  <= '0;
    reg_rd_addr_o   <= '0;
    axi_if.rvalid   <= '0;
    axi_if.rresp    <= '0;
    axi_if.rdata    <= '0;
    // coverage off -item c 1 -feccondrow 4
    if (!rd_wait && !axi_if.arready && axi_if.arvalid) begin
    // coverage on
      axi_if.arready  <= 1'b1;
      rd_wait         <= 1'b1;
      reg_rd_valid_o  <= 1'b1;
      reg_rd_addr_o   <= axi_if.araddr;
    end else if (reg_rd_done_i) begin
      axi_if.rvalid   <= 1'b1;
      axi_if.rresp    <= reg_rd_resp_i;
      axi_if.rdata    <= reg_rd_data_i;
    end else if (axi_if.rvalid) begin
      if (!axi_if.rready) begin
        axi_if.rvalid  <= axi_if.rvalid;
        axi_if.rresp   <= axi_if.rresp;
        axi_if.rdata   <= axi_if.rdata;
      end else begin
        rd_wait        <= 1'b0;
      end
    end
  end
end

endmodule : cmd_queue_v2_0_0_axi_reg

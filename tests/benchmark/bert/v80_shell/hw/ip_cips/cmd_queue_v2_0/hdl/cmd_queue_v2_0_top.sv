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

module cmd_queue_v2_0_0_top #(
  parameter int C_S00_ADDR_WIDTH          = 12,
  parameter int C_S01_ADDR_WIDTH          = 12
) (
  // Clock Ports
  input  logic                         aclk,

  // Reset Ports
  input  logic                         aresetn,

  // S00_AXI Interface Ports
  input  logic [C_S00_ADDR_WIDTH-1:0]  s00_axi_awaddr,
  input  logic                         s00_axi_awvalid,
  output logic                         s00_axi_awready,
  input  logic [32-1:0]                s00_axi_wdata,
  input  logic [4-1:0]                 s00_axi_wstrb,
  input  logic                         s00_axi_wvalid,
  output logic                         s00_axi_wready,
  output logic [2-1:0]                 s00_axi_bresp,
  output logic                         s00_axi_bvalid,
  input  logic                         s00_axi_bready,
  input  logic [C_S00_ADDR_WIDTH-1:0]  s00_axi_araddr,
  input  logic                         s00_axi_arvalid,
  output logic                         s00_axi_arready,
  output logic [32-1:0]                s00_axi_rdata,
  output logic [2-1:0]                 s00_axi_rresp,
  output logic                         s00_axi_rvalid,
  input  logic                         s00_axi_rready,

  // S01_AXI Interface Ports
  input  logic [C_S01_ADDR_WIDTH-1:0]  s01_axi_awaddr,
  input  logic                         s01_axi_awvalid,
  output logic                         s01_axi_awready,
  input  logic [32-1:0]                s01_axi_wdata,
  input  logic [4-1:0]                 s01_axi_wstrb,
  input  logic                         s01_axi_wvalid,
  output logic                         s01_axi_wready,
  output logic [2-1:0]                 s01_axi_bresp,
  output logic                         s01_axi_bvalid,
  input  logic                         s01_axi_bready,
  input  logic [C_S01_ADDR_WIDTH-1:0]  s01_axi_araddr,
  input  logic                         s01_axi_arvalid,
  output logic                         s01_axi_arready,
  output logic [32-1:0]                s01_axi_rdata,
  output logic [2-1:0]                 s01_axi_rresp,
  output logic                         s01_axi_rvalid,
  input  logic                         s01_axi_rready,

  // Interrupt Ports
  output logic                         irq_sq,
  output logic                         irq_cq
);

// --------------------------------------------------------
// Time Units/Precision
// --------------------------------------------------------
// synthesis translate_off
timeunit 1ns/1ps;
// synthesis translate_on

// --------------------------------------------------------
// Package Import
// --------------------------------------------------------

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

// ========================================================

// AXI4-Lite Interface Instantiations
cmd_queue_v2_0_0_axi_if #(.C_ADDR_WIDTH(C_S00_ADDR_WIDTH)) sq_axi_if();
cmd_queue_v2_0_0_axi_if #(.C_ADDR_WIDTH(C_S01_ADDR_WIDTH)) cq_axi_if();

// Producer (SQ) AXI-Lite Interface/port connections
assign sq_axi_if.awaddr   = s00_axi_awaddr;
assign sq_axi_if.awvalid  = s00_axi_awvalid;
assign s00_axi_awready    = sq_axi_if.awready;
assign sq_axi_if.wdata    = s00_axi_wdata;
assign sq_axi_if.wstrb    = s00_axi_wstrb;
assign sq_axi_if.wvalid   = s00_axi_wvalid;
assign s00_axi_wready     = sq_axi_if.wready;
assign s00_axi_bresp      = sq_axi_if.bresp;
assign s00_axi_bvalid     = sq_axi_if.bvalid;
assign sq_axi_if.bready   = s00_axi_bready;
assign sq_axi_if.araddr   = s00_axi_araddr;
assign sq_axi_if.arvalid  = s00_axi_arvalid;
assign s00_axi_arready    = sq_axi_if.arready;
assign s00_axi_rdata      = sq_axi_if.rdata;
assign s00_axi_rresp      = sq_axi_if.rresp;
assign s00_axi_rvalid     = sq_axi_if.rvalid;
assign sq_axi_if.rready   = s00_axi_rready;

// Consumer (CQ) AXI-Lite Interface/port connections
assign cq_axi_if.awaddr   = s01_axi_awaddr;
assign cq_axi_if.awvalid  = s01_axi_awvalid;
assign s01_axi_awready    = cq_axi_if.awready;
assign cq_axi_if.wdata    = s01_axi_wdata;
assign cq_axi_if.wstrb    = s01_axi_wstrb;
assign cq_axi_if.wvalid   = s01_axi_wvalid;
assign s01_axi_wready     = cq_axi_if.wready;
assign s01_axi_bresp      = cq_axi_if.bresp;
assign s01_axi_bvalid     = cq_axi_if.bvalid;
assign cq_axi_if.bready   = s01_axi_bready;
assign cq_axi_if.araddr   = s01_axi_araddr;
assign cq_axi_if.arvalid  = s01_axi_arvalid;
assign s01_axi_arready    = cq_axi_if.arready;
assign s01_axi_rdata      = cq_axi_if.rdata;
assign s01_axi_rresp      = cq_axi_if.rresp;
assign s01_axi_rvalid     = cq_axi_if.rvalid;
assign cq_axi_if.rready   = s01_axi_rready;

// ========================================================

// Register Interface Instantiations
cmd_queue_v2_0_0_reg_if #(.C_ADDR_WIDTH(C_S00_ADDR_WIDTH)) sq_reg_if();
cmd_queue_v2_0_0_reg_if #(.C_ADDR_WIDTH(C_S01_ADDR_WIDTH)) cq_reg_if();

// ========================================================

// AXI Module Instantiation
cmd_queue_v2_0_0_axi #(
  .C_S00_ADDR_WIDTH(C_S00_ADDR_WIDTH),
  .C_S01_ADDR_WIDTH(C_S01_ADDR_WIDTH)
) axi_inst (
  .sq_axi_if,
  .cq_axi_if,
  .sq_reg_if,
  .cq_reg_if,
  .aclk,
  .aresetn
);

// ========================================================

// Command Queue Registers Module Instantiation
cmd_queue_v2_0_0_regs top_reg_inst (
  .sq_reg_if,
  .cq_reg_if,
  .aclk,
  .aresetn,
  .irq_sq,
  .irq_cq
);

// ========================================================

endmodule : cmd_queue_v2_0_0_top
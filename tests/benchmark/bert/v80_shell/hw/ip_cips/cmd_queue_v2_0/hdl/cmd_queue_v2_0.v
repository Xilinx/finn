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

`timescale 1ns/1ps

module cmd_queue_v2_0_0 #(
	parameter integer C_S00_ADDR_WIDTH          = 12,
 	parameter integer C_S01_ADDR_WIDTH          = 12,
  parameter         C_S00_AXI_BASEADDR        = 32'hFFFFFFFF,
  parameter         C_S00_AXI_HIGHADDR        = 32'h00000000,
  parameter         C_S01_AXI_BASEADDR        = 32'hFFFFFFFF,
  parameter         C_S01_AXI_HIGHADDR        = 32'h00000000
)	(
	// Clock Ports
  input  wire                         aclk,

  // Reset Ports
  input  wire                         aresetn,

  // S00_AXI Interface Ports
  input  wire [C_S00_ADDR_WIDTH-1:0]  s00_axi_awaddr,
  input  wire                         s00_axi_awvalid,
  output wire                         s00_axi_awready,
  input  wire [32-1:0]                s00_axi_wdata,
  input  wire [4-1:0]                 s00_axi_wstrb,
  input  wire                         s00_axi_wvalid,
  output wire                         s00_axi_wready,
  output wire [2-1:0]                 s00_axi_bresp,
  output wire                         s00_axi_bvalid,
  input  wire                         s00_axi_bready,
  input  wire [C_S00_ADDR_WIDTH-1:0]  s00_axi_araddr,
  input  wire                         s00_axi_arvalid,
  output wire                         s00_axi_arready,
  output wire [32-1:0]                s00_axi_rdata,
  output wire [2-1:0]                 s00_axi_rresp,
  output wire                         s00_axi_rvalid,
  input  wire                         s00_axi_rready,

  // S01_AXI Interface Ports
  input  wire [C_S01_ADDR_WIDTH-1:0]  s01_axi_awaddr,
  input  wire                         s01_axi_awvalid,
  output wire                         s01_axi_awready,
  input  wire [32-1:0]                s01_axi_wdata,
  input  wire [4-1:0]                 s01_axi_wstrb,
  input  wire                         s01_axi_wvalid,
  output wire                         s01_axi_wready,
  output wire [2-1:0]                 s01_axi_bresp,
  output wire                         s01_axi_bvalid,
  input  wire                         s01_axi_bready,
  input  wire [C_S01_ADDR_WIDTH-1:0]  s01_axi_araddr,
  input  wire                         s01_axi_arvalid,
  output wire                         s01_axi_arready,
  output wire [32-1:0]                s01_axi_rdata,
  output wire [2-1:0]                 s01_axi_rresp,
  output wire                         s01_axi_rvalid,
  input  wire                         s01_axi_rready,

  // Interrupt Ports
  output wire                         irq_sq,
  output wire                         irq_cq
);

// --------------------------------------------------------
// GCQ Top Level Instantiation
// --------------------------------------------------------
cmd_queue_v2_0_0_top #(
  .C_S00_ADDR_WIDTH(C_S00_ADDR_WIDTH),
	.C_S01_ADDR_WIDTH(C_S01_ADDR_WIDTH)
) cmd_queue_top_inst (
  // Clocks
  .aclk(aclk),

  // Resets
  .aresetn(aresetn),

  // S00_AXI Interface
  .s00_axi_awaddr(s00_axi_awaddr),
  .s00_axi_awvalid(s00_axi_awvalid),
  .s00_axi_awready(s00_axi_awready),
  .s00_axi_wdata(s00_axi_wdata),
  .s00_axi_wstrb(s00_axi_wstrb),
  .s00_axi_wvalid(s00_axi_wvalid),
  .s00_axi_wready(s00_axi_wready),
  .s00_axi_bresp(s00_axi_bresp),
  .s00_axi_bvalid(s00_axi_bvalid),
  .s00_axi_bready(s00_axi_bready),
  .s00_axi_araddr(s00_axi_araddr),
  .s00_axi_arvalid(s00_axi_arvalid),
  .s00_axi_arready(s00_axi_arready),
  .s00_axi_rdata(s00_axi_rdata),
  .s00_axi_rresp(s00_axi_rresp),
  .s00_axi_rvalid(s00_axi_rvalid),
  .s00_axi_rready(s00_axi_rready),

	// S01_AXI Interface
  .s01_axi_awaddr(s01_axi_awaddr),
  .s01_axi_awvalid(s01_axi_awvalid),
  .s01_axi_awready(s01_axi_awready),
  .s01_axi_wdata(s01_axi_wdata),
  .s01_axi_wstrb(s01_axi_wstrb),
  .s01_axi_wvalid(s01_axi_wvalid),
  .s01_axi_wready(s01_axi_wready),
  .s01_axi_bresp(s01_axi_bresp),
  .s01_axi_bvalid(s01_axi_bvalid),
  .s01_axi_bready(s01_axi_bready),
  .s01_axi_araddr(s01_axi_araddr),
  .s01_axi_arvalid(s01_axi_arvalid),
  .s01_axi_arready(s01_axi_arready),
  .s01_axi_rdata(s01_axi_rdata),
  .s01_axi_rresp(s01_axi_rresp),
  .s01_axi_rvalid(s01_axi_rvalid),
  .s01_axi_rready(s01_axi_rready),

  // Interrupts
  .irq_sq(irq_sq),
	.irq_cq(irq_cq)

);

endmodule

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

interface cmd_queue_v2_0_0_axi_if #(
  parameter int C_DATA_WIDTH = 32,
  parameter int C_ADDR_WIDTH = 32
);

// --------------------------------------------------------
// Time Units/Precision
// --------------------------------------------------------
// synthesis translate_off
timeunit 1ns/1ps;
// synthesis translate_on

// --------------------------------------------------------
// AXI4-Lite Interface Signals
// --------------------------------------------------------

// Write Address Channel
logic [C_ADDR_WIDTH-1:0]      awaddr;
logic                         awvalid;
logic                         awready;

// Write Data Channel
logic [C_DATA_WIDTH-1:0]      wdata;
logic [(C_DATA_WIDTH/8)-1:0]  wstrb;
logic                         wvalid;
logic                         wready;

// Write Response Channel
logic                         bvalid;
logic                         bready;
logic [1:0]                   bresp;

// Read Address Channel
logic [C_ADDR_WIDTH-1:0]      araddr;
logic                         arvalid;
logic                         arready;

// Read Data Channel
logic [C_DATA_WIDTH-1:0]      rdata;
logic [1:0]                   rresp;
logic                         rvalid;
logic                         rready;

// --------------------------------------------------------
// AXI4-Lite Manager Interface
// --------------------------------------------------------
modport man (
  output awaddr,
  output awvalid,
  input  awready,
  output wdata,
  output wstrb,
  output wvalid,
  input  wready,
  input  bvalid,
  input  bresp,
  output bready,
  output araddr,
  output arvalid,
  input  arready,
  input  rdata,
  input  rresp,
  input  rvalid,
  output rready
);

// --------------------------------------------------------
// AXI4-Lite Subordinate Interface
// --------------------------------------------------------
modport sub (
  input  awaddr,
  input  awvalid,
  output awready,
  input  wdata,
  input  wstrb,
  input  wvalid,
  output wready,
  output bvalid,
  output bresp,
  input  bready,
  input  araddr,
  input  arvalid,
  output arready,
  output rdata,
  output rresp,
  output rvalid,
  input  rready
);

// --------------------------------------------------------
// AXI4-Lite Monitor Interface
// --------------------------------------------------------
modport mon (
  input awaddr,
  input awvalid,
  input awready,
  input wdata,
  input wstrb,
  input wvalid,
  input wready,
  input bvalid,
  input bresp,
  input bready,
  input araddr,
  input arvalid,
  input arready,
  input rdata,
  input rresp,
  input rvalid,
  input rready
);

endinterface : cmd_queue_v2_0_0_axi_if

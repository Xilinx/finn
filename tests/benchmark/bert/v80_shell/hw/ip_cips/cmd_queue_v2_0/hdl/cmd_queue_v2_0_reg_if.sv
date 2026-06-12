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

interface cmd_queue_v2_0_0_reg_if #(
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
// Register Read Interface
// --------------------------------------------------------

logic                         reg_rd_valid;
logic [C_ADDR_WIDTH-1:0]      reg_rd_addr;
logic                         reg_rd_done;
logic [1:0]                   reg_rd_resp;
logic [C_DATA_WIDTH-1:0]      reg_rd_data;

// --------------------------------------------------------
// Register Write Interface
// --------------------------------------------------------

logic                         reg_wr_valid;
logic [C_ADDR_WIDTH-1:0]      reg_wr_addr;
logic [(C_DATA_WIDTH/8)-1:0]  reg_wr_be;
logic [C_DATA_WIDTH-1:0]      reg_wr_data;
logic                         reg_wr_done;
logic [1:0]                   reg_wr_resp;

// --------------------------------------------------------
// Register Manager Interface
// --------------------------------------------------------
modport man (
  output reg_rd_valid,
  output reg_rd_addr,
  input  reg_rd_done,
  input  reg_rd_resp,
  input  reg_rd_data,
  output reg_wr_valid,
  output reg_wr_addr,
  output reg_wr_be,
  output reg_wr_data,
  input  reg_wr_done,
  input  reg_wr_resp
);

// --------------------------------------------------------
// Register Subordinate Interface
// --------------------------------------------------------
modport sub (
  input  reg_rd_valid,
  input  reg_rd_addr,
  output reg_rd_done,
  output reg_rd_resp,
  output reg_rd_data,
  input  reg_wr_valid,
  input  reg_wr_addr,
  input  reg_wr_be,
  input  reg_wr_data,
  output reg_wr_done,
  output reg_wr_resp
);

endinterface : cmd_queue_v2_0_0_reg_if

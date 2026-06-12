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

module cmd_queue_v2_0_0_axi #(
  parameter int C_S00_ADDR_WIDTH   = 12,  // Address width of SQ AXI and Reg interfaces
  parameter int C_S01_ADDR_WIDTH   = 12,  // Address Width of CQ AXI and Reg interfaces
  parameter int C_S00_DATA_WIDTH   = 32,  // Data width of SQ AXI and Reg interfaces
  parameter int C_S01_DATA_WIDTH   = 32   // Data width of CQ AXI and Reg interfaces
) (
  // AXI4-Lite Subordinate Interface
  cmd_queue_v2_0_0_axi_if.sub         sq_axi_if,
  cmd_queue_v2_0_0_axi_if.sub         cq_axi_if,

  // Manager Register Interfaces
  cmd_queue_v2_0_0_reg_if.man         sq_reg_if,
  cmd_queue_v2_0_0_reg_if.man         cq_reg_if,

  // Clock/Reset
  input  logic                      aclk,
  input  logic                      aresetn
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

logic                             sq_reg_rd_valid;
logic [C_S00_ADDR_WIDTH-1:0]      sq_reg_rd_addr;
logic                             sq_reg_rd_done;
logic [1:0]                       sq_reg_rd_resp;
logic [C_S00_DATA_WIDTH-1:0]      sq_reg_rd_data;
logic                             sq_reg_wr_valid;
logic [C_S00_ADDR_WIDTH-1:0]      sq_reg_wr_addr;
logic [(C_S00_DATA_WIDTH/8)-1:0]  sq_reg_wr_be;
logic [C_S00_DATA_WIDTH-1:0]      sq_reg_wr_data;
logic                             sq_reg_wr_done;
logic [1:0]                       sq_reg_wr_resp;

logic                             cq_reg_rd_valid;
logic [C_S01_ADDR_WIDTH-1:0]      cq_reg_rd_addr;
logic                             cq_reg_rd_done;
logic [1:0]                       cq_reg_rd_resp;
logic [C_S01_DATA_WIDTH-1:0]      cq_reg_rd_data;
logic                             cq_reg_wr_valid;
logic [C_S01_ADDR_WIDTH-1:0]      cq_reg_wr_addr;
logic [(C_S01_DATA_WIDTH/8)-1:0]  cq_reg_wr_be;
logic [C_S01_DATA_WIDTH-1:0]      cq_reg_wr_data;
logic                             cq_reg_wr_done;
logic [1:0]                       cq_reg_wr_resp;

// ========================================================

// --------------------------------------------------------
// SQ AXI Register Interface Module Instantiation

cmd_queue_v2_0_0_axi_reg #(
  .C_ADDR_WIDTH(C_S00_ADDR_WIDTH),
  .C_DATA_WIDTH(C_S00_DATA_WIDTH)
) sq_axi_reg_inst (
  .axi_if(sq_axi_if),
  .aclk,
  .aresetn,
  .reg_rd_valid_o(sq_reg_rd_valid),
  .reg_rd_addr_o(sq_reg_rd_addr),
  .reg_rd_done_i(sq_reg_rd_done),
  .reg_rd_resp_i(sq_reg_rd_resp),
  .reg_rd_data_i(sq_reg_rd_data),
  .reg_wr_valid_o(sq_reg_wr_valid),
  .reg_wr_addr_o(sq_reg_wr_addr),
  .reg_wr_be_o(sq_reg_wr_be),
  .reg_wr_data_o(sq_reg_wr_data),
  .reg_wr_done_i(sq_reg_wr_done),
  .reg_wr_resp_i(sq_reg_wr_resp)
);

// --------------------------------------------------------
// CQ AXI Register Interface Module Instantiation

cmd_queue_v2_0_0_axi_reg #(
  .C_ADDR_WIDTH(C_S01_ADDR_WIDTH),
  .C_DATA_WIDTH(C_S01_DATA_WIDTH)
) cq_axi_reg_inst (
  .axi_if(cq_axi_if),
  .aclk,
  .aresetn,
  .reg_rd_valid_o(cq_reg_rd_valid),
  .reg_rd_addr_o(cq_reg_rd_addr),
  .reg_rd_done_i(cq_reg_rd_done),
  .reg_rd_resp_i(cq_reg_rd_resp),
  .reg_rd_data_i(cq_reg_rd_data),
  .reg_wr_valid_o(cq_reg_wr_valid),
  .reg_wr_addr_o(cq_reg_wr_addr),
  .reg_wr_be_o(cq_reg_wr_be),
  .reg_wr_data_o(cq_reg_wr_data),
  .reg_wr_done_i(cq_reg_wr_done),
  .reg_wr_resp_i(cq_reg_wr_resp)
);

// --------------------------------------------------------
// Register Interface assignments

// SQ Register Interface
assign sq_reg_if.reg_rd_valid   = sq_reg_rd_valid;
assign sq_reg_if.reg_rd_addr    = sq_reg_rd_addr;
assign sq_reg_rd_done           = sq_reg_if.reg_rd_done;
assign sq_reg_rd_resp           = sq_reg_if.reg_rd_resp;
assign sq_reg_rd_data           = sq_reg_if.reg_rd_data;

assign sq_reg_if.reg_wr_valid   = sq_reg_wr_valid;
assign sq_reg_if.reg_wr_addr    = sq_reg_wr_addr;
assign sq_reg_if.reg_wr_be      = sq_reg_wr_be;
assign sq_reg_if.reg_wr_data    = sq_reg_wr_data;
assign sq_reg_wr_done           = sq_reg_if.reg_wr_done;
assign sq_reg_wr_resp           = sq_reg_if.reg_wr_resp;

// CQ Register Interface
assign cq_reg_if.reg_rd_valid   = cq_reg_rd_valid;
assign cq_reg_if.reg_rd_addr    = cq_reg_rd_addr;
assign cq_reg_rd_done           = cq_reg_if.reg_rd_done;
assign cq_reg_rd_resp           = cq_reg_if.reg_rd_resp;
assign cq_reg_rd_data           = cq_reg_if.reg_rd_data;

assign cq_reg_if.reg_wr_valid   = cq_reg_wr_valid;
assign cq_reg_if.reg_wr_addr    = cq_reg_wr_addr;
assign cq_reg_if.reg_wr_be      = cq_reg_wr_be;
assign cq_reg_if.reg_wr_data    = cq_reg_wr_data;
assign cq_reg_wr_done           = cq_reg_if.reg_wr_done;
assign cq_reg_wr_resp           = cq_reg_if.reg_wr_resp;

endmodule : cmd_queue_v2_0_0_axi

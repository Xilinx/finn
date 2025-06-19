/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

import iwTypes::*;

/**
 * @brief   Aligned CDMA AXI write engine
 *
 * The aligned CDMA write engine, AXI to stream. Supports outstanding transactions (N_OUTSTANDING).
 * Low resource overhead. Used in striping.
 *
 *  @param BURST_LEN    Maximum burst length size
 *  @param DATA_BITS    Size of the data bus (both AXI and stream)
 *  @param ADDR_BITS    Size of the address bits
 *  @param ID_BITS      Size of the ID bits
 */
module axi_dma_wr_a #(
  parameter integer                     BURST_LEN = 16,
  parameter integer                     DATA_BITS = AXI_DATA_BITS,
  parameter integer                     ADDR_BITS = AXI_ADDR_BITS,
  parameter integer                     ID_BITS = AXI_ID_BITS,
  parameter integer                     MAX_OUTSTANDING = 8
) (
  // AXI Interface 
  input  wire                           aclk,
  input  wire                           aresetn,

  // Control interface
  input  wire                           ctrl_valid,
  output wire                           stat_ready,
  input  wire [ADDR_BITS-1:0]           ctrl_addr,
  input  wire [LEN_BITS-1:0]            ctrl_len,
  input  wire                           ctrl_ctl,
  output wire                           stat_done,

  output wire                           awvalid,
  input  wire                           awready,
  output wire [ADDR_BITS-1:0]           awaddr,
  output wire [ID_BITS-1:0]             awid,
  output wire [7:0]                     awlen,
  output wire [2:0]                     awsize,
  output wire [1:0]                     awburst,
  output wire [0:0]                     awlock,
  output wire [3:0]                     awcache,
  output wire [DATA_BITS-1:0]           wdata,
  output wire [DATA_BITS/8-1:0]         wstrb,
  output wire                           wlast,
  output wire                           wvalid,
  input  wire                           wready,
  input  wire [ID_BITS-1:0]             bid,
  input  wire [1:0]                     bresp,
  input  wire                           bvalid,
  output wire                           bready,

  // AXI4-Stream slave interface
  input  wire                           axis_in_tvalid,
  output wire                           axis_in_tready,
  input  wire [DATA_BITS-1:0]           axis_in_tdata,
  input  wire [DATA_BITS/8-1:0]         axis_in_tkeep,
  input  wire                           axis_in_tlast
);

///////////////////////////////////////////////////////////////////////////////
// Local Parameters
///////////////////////////////////////////////////////////////////////////////
localparam integer AXI_MAX_BURST_LEN = BURST_LEN;
localparam integer AXI_DATA_BYTES = DATA_BITS / 8;
localparam integer LOG_DATA_LEN = $clog2(AXI_DATA_BYTES);
localparam integer LOG_BURST_LEN = $clog2(AXI_MAX_BURST_LEN);
localparam integer LP_MAX_OUTSTANDING_CNTR_WIDTH = $clog2(MAX_OUTSTANDING+1); 
localparam integer LP_TRANSACTION_CNTR_WIDTH = LEN_BITS-LOG_BURST_LEN-LOG_DATA_LEN;

logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_full_bursts;
logic num_partial_bursts;

logic start;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_transactions;
logic has_partial_burst;
logic [LOG_BURST_LEN-1:0] final_burst_len;
logic single_transaction;

// AW
logic awvalid_r;
logic [ADDR_BITS-1:0] addr_r;
logic ctl_r;
logic aw_done;
logic aw_idle;

logic awxfer;
logic aw_final_transaction;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] aw_transactions_to_go;

// W
logic wxfer;
logic [LOG_BURST_LEN-1:0] wxfers_to_go;

logic burst_load;
logic burst_active;
logic burst_ready_snk;
logic burst_ready_src;
logic [LOG_BURST_LEN-1:0] burst_len;

// B
logic bxfer;
logic b_final_transaction;

logic b_ready_snk;

/////////////////////////////////////////////////////////////////////////////
// Control logic
/////////////////////////////////////////////////////////////////////////////
assign stat_done = bxfer & b_final_transaction;
assign stat_ready = aw_idle;

// Count the number of transfers and assert done when the last bvalid is received.
assign num_full_bursts = ctrl_len[LOG_DATA_LEN+LOG_BURST_LEN+:LEN_BITS-LOG_DATA_LEN-LOG_BURST_LEN];
assign num_partial_bursts = ctrl_len[LOG_DATA_LEN+:LOG_BURST_LEN] ? 1'b1 : 1'b0; 

always_ff @(posedge aclk) begin
  if(~aresetn) begin
    start <= 0;
    num_transactions <= 'X;
    has_partial_burst <= 'X;
    final_burst_len <= 'X;
  end
  else begin
    start <= ctrl_valid & stat_ready;
    if(ctrl_valid & stat_ready) begin
      num_transactions <= (num_partial_bursts == 1'b0) ? num_full_bursts - 1'b1 : num_full_bursts;
      has_partial_burst <= num_partial_bursts;
      final_burst_len <=  ctrl_len[LOG_DATA_LEN+:LOG_BURST_LEN] - 1'b1;
    end
  end
end

// Special case if there is only 1 AXI transaction. 
assign single_transaction = (num_transactions == {LP_TRANSACTION_CNTR_WIDTH{1'b0}}) ? 1'b1 : 1'b0;

///////////////////////////////////////////////////////////////////////////////
// AXI Write Address Channel
///////////////////////////////////////////////////////////////////////////////
assign awvalid = awvalid_r;
assign awaddr = addr_r;
assign awlen = aw_final_transaction ? final_burst_len : AXI_MAX_BURST_LEN - 1;
assign awsize = LOG_DATA_LEN;
assign awid = 0;

assign awburst = 2'b01;
assign awlock = 1'b0;
assign awcache = 4'b0011;

assign awxfer = awvalid & awready;

// Send aw_valid
always_ff @(posedge aclk) begin
  if (~aresetn) begin 
    awvalid_r <= 1'b0;
  end
  else begin
    awvalid_r <= ~aw_idle & ~awvalid_r & b_ready_snk ? 1'b1 : 
                 awready ? 1'b0 : awvalid_r;
  end
end

// When aw_idle, there are no transactions to issue.
always_ff @(posedge aclk) begin
  if (~aresetn) begin 
    aw_idle <= 1'b1; 
  end
  else begin 
    aw_idle <= (ctrl_valid & stat_ready) ? 1'b0 :
               aw_done    ? 1'b1 : aw_idle;
  end
end

// Increment to next address after each transaction is issued. Ctl latching.
always_ff @(posedge aclk) begin
  if (~aresetn) begin
    ctl_r <= 1'b0;
    addr_r <= 'X;
  end
  else begin
    addr_r <= (ctrl_valid & stat_ready) ? ctrl_addr :
               awxfer  ? addr_r + AXI_MAX_BURST_LEN*AXI_DATA_BYTES : addr_r;
    ctl_r <= (ctrl_valid & stat_ready) ? ctrl_ctl : ctl_r;
  end
end

// Counts down the number of transactions to send.
krnl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_aw_transaction_cntr ( 
  .aclk       ( aclk                   ) ,
  .clken      ( 1'b1                   ) ,
  .aresetn    ( aresetn                ) ,
  .load       ( start                  ) ,
  .incr       ( 1'b0                   ) ,
  .decr       ( awxfer                 ) ,
  .load_value ( num_transactions       ) ,
  .count      ( aw_transactions_to_go  ) ,
  .is_zero    ( aw_final_transaction   ) 
);

assign aw_done = aw_final_transaction && awxfer;

/////////////////////////////////////////////////////////////////////////////
// AXI Write Data Channel
/////////////////////////////////////////////////////////////////////////////
assign wvalid = axis_in_tvalid & burst_active;
assign wdata = axis_in_tdata;
assign wstrb = axis_in_tkeep;
assign axis_in_tready = wready & burst_active;

assign wxfer = wvalid & wready;

assign burst_load = burst_ready_src && ((wlast & wxfer) || ~burst_active);

always_ff @(posedge aclk) begin
  if (~aresetn) begin 
    burst_active <= 1'b0;
  end
  else begin
    burst_active <=  burst_load ? 1'b1 : 
                        (wlast & wxfer) ? 1'b0 : burst_active;
  end
end

krnl_counter #(
  .C_WIDTH ( LOG_BURST_LEN         ) ,
  .C_INIT  ( {LOG_BURST_LEN{1'b1}} ) 
)
inst_burst_cntr ( 
  .aclk       ( aclk            ) ,
  .clken      ( 1'b1            ) ,
  .aresetn    ( aresetn         ) ,
  .load       ( burst_load      ) ,
  .incr       ( 1'b0            ) ,
  .decr       ( wxfer           ) ,
  .load_value ( burst_len       ) ,
  .count      ( wxfers_to_go    ) ,
  .is_zero    ( wlast           ) 
);

Q_srl #(
    .depth(MAX_OUTSTANDING), 
    .width(LOG_BURST_LEN)
) inst_q_wr_req (
    .clock(aclk),
    .reset(!aresetn),
    .count(),
    .maxcount(),
    .i_d(awlen[0+:LOG_BURST_LEN]),
    .i_v(awxfer),
    .i_r(burst_ready_snk),
    .o_d(burst_len),
    .o_v(burst_ready_src),
    .o_r(burst_load)
);

/////////////////////////////////////////////////////////////////////////////
// AXI Write Response Channel
/////////////////////////////////////////////////////////////////////////////
assign bready = 1'b1;
assign bxfer = bready & bvalid;

Q_srl #(
    .depth(MAX_OUTSTANDING), 
    .width(1)
) inst_q_wr_rsp (
    .clock(aclk),
    .reset(!aresetn),
    .count(),
    .maxcount(),
    .i_d(ctl_r & aw_final_transaction),
    .i_v(awxfer),
    .i_r(b_ready_snk),
    .o_d(b_final_transaction),
    .o_v(),
    .o_r(bxfer)
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_WR_A

`endif

endmodule
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
 * @brief   Aligned CDMA AXI read engine
 *
 * The aligned CDMA read engine, AXI to stream. Supports outstanding transactions (N_OUTSTANDING).
 * Low resource overhead. Used in striping.
 *
 *  @param BURST_LEN    Maximum burst length size
 *  @param DATA_BITS    Size of the data bus (both AXI and stream)
 *  @param ADDR_BITS    Size of the address bits
 *  @param ID_BITS      Size of the ID bits
 */
module axi_dma_rd_a #(
  parameter integer                 BURST_LEN = 16,
  parameter integer                 DATA_BITS = HBM_DATA_BITS,
  parameter integer                 ADDR_BITS = HBM_ADDR_BITS,
  parameter integer                 LEN_BITS = HBM_LEN_BITS,
  parameter integer                 ID_BITS = HBM_ID_BITS,
  parameter integer                 MAX_OUTSTANDING = 8
) (
  // Clock and reset
  input  wire                       aclk,
  input  wire                       aresetn,

  // Control and status
  input  wire                       ctrl_valid,
  output wire                       stat_ready,
  input  wire [ADDR_BITS-1:0]       ctrl_addr,
  input  wire [LEN_BITS-1:0]        ctrl_len,
  input  wire                       ctrl_ctl,
  output wire                       stat_done,

  // AXI4 master interface
  output wire                       arvalid,
  input  wire                       arready,
  output wire [ADDR_BITS-1:0]       araddr,
  output wire [ID_BITS-1:0]         arid,
  output wire [7:0]                 arlen,
  output wire [2:0]                 arsize,
  output wire [1:0]                 arburst,
  output wire [0:0]                 arlock,
  output wire [3:0]                 arcache,
  output wire [2:0]                 arprot,
  input  wire                       rvalid,
  output wire                       rready,
  input  wire [DATA_BITS-1:0]       rdata,
  input  wire                       rlast,
  input  wire [ID_BITS-1:0]         rid,
  input  wire [1:0]                 rresp,

  // AXI4-Stream master interface
  output wire                       axis_out_tvalid,
  input  wire                       axis_out_tready,
  output wire [DATA_BITS-1:0]       axis_out_tdata,
  output wire [DATA_BITS/8-1:0]     axis_out_tkeep,
  output wire                       axis_out_tlast
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

// AR
logic arvalid_r;
logic [ADDR_BITS-1:0] addr_r;
logic ctl_r;
logic ar_done;
logic ar_idle;

logic arxfer;
logic ar_final_transaction;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] ar_transactions_to_go;

// R
logic rxfer;
logic r_final_transaction;

logic burst_ready_snk;

///////////////////////////////////////////////////////////////////////////////
// Ctrl
///////////////////////////////////////////////////////////////////////////////
assign  stat_done = rxfer & rlast & r_final_transaction;
assign  stat_ready = ar_idle;

// Determine how many full burst to issue and if there are any partial bursts.
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
// AXI Read Address Channel
///////////////////////////////////////////////////////////////////////////////
assign arvalid = arvalid_r;
assign araddr = addr_r;
assign arlen = ar_final_transaction ? final_burst_len : AXI_MAX_BURST_LEN - 1;
assign arsize = LOG_DATA_LEN;
assign arid = 0;

assign arburst = 2'b01;
assign arlock = 1'b0;
assign arcache = 4'b0011;
assign arprot = 3'b010;

assign arxfer = arvalid & arready;

// Send ar_valid
 always_ff @(posedge aclk) begin
  if (~aresetn) begin
    arvalid_r <= 1'b0;
  end
  else begin
    arvalid_r <= ~ar_idle & ~arvalid_r & burst_ready_snk ? 1'b1 :
                 arready ? 1'b0 : arvalid_r;
  end
end

// When ar_idle, there are no transactions to issue.
 always_ff @(posedge aclk) begin
  if (~aresetn) begin
    ar_idle <= 1'b1;
  end
  else begin
    ar_idle <= (ctrl_valid & stat_ready) ? 1'b0 :
               ar_done    ? 1'b1 : ar_idle;
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
               arxfer  ? addr_r + AXI_MAX_BURST_LEN*AXI_DATA_BYTES : addr_r;
    ctl_r <= (ctrl_valid & stat_ready) ? ctrl_ctl : ctl_r;
  end
end

// Counts down the number of transactions to send.
krnl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} )
)
inst_ar_transaction_cntr (
  .aclk       ( aclk                   ) ,
  .clken      ( 1'b1                   ) ,
  .aresetn    ( aresetn                ) ,
  .load       ( start                  ) ,
  .incr       ( 1'b0                   ) ,
  .decr       ( arxfer                 ) ,
  .load_value ( num_transactions       ) ,
  .count      ( ar_transactions_to_go  ) ,
  .is_zero    ( ar_final_transaction   )
);

assign ar_done = ar_final_transaction && arxfer;

///////////////////////////////////////////////////////////////////////////////
// AXI Read Channel
///////////////////////////////////////////////////////////////////////////////
assign axis_out_tvalid = rvalid;
assign axis_out_tdata = rdata;
assign axis_out_tkeep = ~0;
assign axis_out_tlast = rlast & r_final_transaction;
assign rready = axis_out_tready;

assign rxfer = rready & rvalid;

Q_srl #(
    .depth(MAX_OUTSTANDING),
    .width(1)
) inst_q_rd (
    .clock(aclk),
    .reset(!aresetn),
    .count(),
    .maxcount(),
    .i_d(ctl_r & ar_final_transaction),
    .i_v(arxfer),
    .i_r(burst_ready_snk),
    .o_d(r_final_transaction),
    .o_v(),
    .o_r(rlast & rxfer)
);

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_RD_A

`endif

endmodule

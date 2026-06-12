// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

import iwTypes::*;

module axil_iw_slv_mlo #(
    parameter                                           LEN_BITS = 32,
    parameter                                           ADDR_BITS = CSR_ADDR_BITS,
    parameter                                           CNT_BITS = 16,

    parameter int unsigned                              N_FW_CORES,
    parameter int unsigned                              MAX_INTERVALS = 1024
) (
    input  logic                                        aclk,
    input  logic                                        aresetn,

    AXI4L.slave                                         axi_ctrl,

    output logic [CNT_BITS-1:0]                         n_layers,

    AXI4S.master                                        f_ctrl_fs,
    AXI4S.master                                        f_ctrl_se,

    input  logic                                        s_done,
    input  logic [1:0]                                  s_done_if,
    input  logic [N_FW_CORES-1:0]                       s_done_w
);

// -- Decl ----------------------------------------------------------
// ------------------------------------------------------------------
// Constants
localparam integer N_REGS = 32;
localparam integer AXIL_DATA_BITS = 64;
localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer AXIL_ADDR_BITS = ADDR_LSB + ADDR_MSB;
localparam integer N_INT_MS = 1 + 2 + N_FW_CORES; // End + If + Weights

// Internal registers
logic [AXIL_ADDR_BITS-1:0] axi_awaddr;
logic axi_awready;
logic [AXIL_ADDR_BITS-1:0] axi_araddr;
logic axi_arready;
logic [1:0] axi_bresp;
logic axi_bvalid;
logic axi_wready;
logic [AXIL_DATA_BITS-1:0] axi_rdata;
logic [1:0] axi_rresp;
logic axi_rvalid;

// Registers
logic [N_REGS-1:0][AXIL_DATA_BITS-1:0] slv_reg;
logic slv_reg_rden;
logic slv_reg_wren;
logic aw_en;

// Internal
logic post;
logic [15:0] cnt_done;
logic cnt_up;

logic [N_INT_MS-1:0][63:0] cnt_int;
logic [N_INT_MS-1:0] valid_int;
logic [N_INT_MS-1:0][63:0] data_int;
logic [N_INT_MS-1:0][63:0] data_int_out;
logic [N_INT_MS-1:0] done = {s_done_w, s_done_if[1], s_done_if[0], s_done};

logic ready_int;
logic ready_int_if;
logic ready_int_w;
logic [N_INT_MS-1:0] ready_int_out = {{N_FW_CORES{ready_int_w}}, ready_int_if, ready_int_if, ready_int};

// -- Def -----------------------------------------------------------
// ------------------------------------------------------------------
localparam integer PROBE_ID = 1044942;

// -- Register map ----------------------------------------------------------------------- 
localparam integer CTRL_REG = 0;
localparam integer STAT_REG = 1;
localparam integer SRC_OFFS_REG = 2;
localparam integer DST_OFFS_REG = 3;
localparam integer N_FRAMES_REG = 4;
localparam integer LEN_FRAME_REG = 5;
localparam integer N_LAYERS_REG = 6;
localparam integer PERF_LAT_REG = 7;
localparam integer PERF_INT_REG = 8;

localparam integer PROBE_REG = 31;

// Write process
assign slv_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 ) begin
    slv_reg <= 0;
    
    post <= 1'b0;
    ready_int <= 1'b0;
    ready_int_if <= 1'b0;
    ready_int_w <= 1'b0;

    cnt_done <= '0;
    cnt_up <= '0;
    cnt_int <= '0;

    valid_int <= '0;
    data_int <= 'X;
  end
  else begin
    // Control
    post <= 1'b0;
    ready_int <= 1'b0;
    ready_int_if <= 1'b0;
    ready_int_w <= 1'b0;

    // Counters and status
    cnt_done <= post ? 0 : (s_done ? cnt_done + 1 : cnt_done);
    cnt_up <= post ? 1'b1 : ((cnt_done == slv_reg[N_FRAMES_REG] - 1) && s_done) ? 1'b0 : cnt_up;
    slv_reg[PERF_LAT_REG] <= post ? 0 : (cnt_up ? slv_reg[PERF_LAT_REG] + 1 : slv_reg[PERF_LAT_REG]);
    
    for(int i = 0; i < N_INT_MS; i++) begin
      cnt_int[i] <= post ? 0 : (cnt_up ? (done[i] ? 0 : cnt_int[i] + 1) : cnt_int[i]);
      valid_int[i] <= (cnt_up && done[i]);
      data_int[i] <= cnt_int[i] + 1;
      slv_reg[PERF_INT_REG+i] <= data_int_out[i];
    end

    if(slv_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB]) inside
        [CTRL_REG:CTRL_REG]:
            if(axi_ctrl.wstrb[0]) begin
                post <= axi_ctrl.wdata[0];
            end
        [SRC_OFFS_REG:SRC_OFFS_REG]:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        [DST_OFFS_REG:DST_OFFS_REG]:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        [N_FRAMES_REG:N_FRAMES_REG]:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        [LEN_FRAME_REG:LEN_FRAME_REG]:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        [N_LAYERS_REG:N_LAYERS_REG]:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        [PERF_INT_REG:PERF_INT_REG]:
          if(axi_ctrl.wstrb[0]) begin
              ready_int <= axi_ctrl.wdata[0];
              ready_int_if <= axi_ctrl.wdata[1];
              ready_int_w <= axi_ctrl.wdata[2];
          end
      
        default: ;
      endcase
    end

  end
end    

// Read process
assign slv_reg_rden = axi_arready & axi_ctrl.arvalid & ~axi_rvalid;

always_ff @(posedge aclk) begin
  if( aresetn == 1'b0 ) begin
    axi_rdata <= 0;
  end
  else begin
    if(slv_reg_rden) begin
      axi_rdata <= 0;  
      
      case (axi_araddr[ADDR_LSB+:ADDR_MSB]) inside
        [CTRL_REG:CTRL_REG]:
          axi_rdata[0] <= f_ctrl_fs.tready & f_ctrl_se.tready;
        [STAT_REG:STAT_REG]:
          axi_rdata <= (cnt_done == slv_reg[N_FRAMES_REG]);
        [SRC_OFFS_REG:SRC_OFFS_REG]:
          axi_rdata <= slv_reg[SRC_OFFS_REG];
        [DST_OFFS_REG:DST_OFFS_REG]:
          axi_rdata <= slv_reg[DST_OFFS_REG];
        [N_FRAMES_REG:N_FRAMES_REG]:
          axi_rdata <= slv_reg[N_FRAMES_REG];
        [LEN_FRAME_REG:LEN_FRAME_REG]:
          axi_rdata <= slv_reg[LEN_FRAME_REG];
        [N_LAYERS_REG:N_LAYERS_REG]:
          axi_rdata <= slv_reg[N_LAYERS_REG];
        [PERF_LAT_REG:PERF_LAT_REG]:
            axi_rdata <= slv_reg[PERF_LAT_REG];
        [PERF_INT_REG:PERF_INT_REG+N_INT_MS-1]:
            axi_rdata <= slv_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        [PROBE_REG:PROBE_REG]:
          axi_rdata <= PROBE_ID;
        
        default: ;
      endcase
    end
  end 
end

// Interval FIFOs
logic [N_INT_MS-1:0] tmp_rdy, tmp_vld;
logic [N_INT_MS-1:0][15:0] cnt_done_int;
logic [N_INT_MS-1:0][15:0] cnt_ready_out;

for(genvar i = 0; i < N_INT_MS; i++) begin
  /*
  Q_srl #(
    .depth(MAX_INTERVALS),
    .width(64)
  ) inst_int_fifo (
    .clock(aclk),
    .reset(~aresetn || post),
    .i_d(data_int[i]),
    .i_v(valid_int[i]),
    .i_r(tmp_rdy[i]),
    .o_d(data_int_out[i]),
    .o_v(tmp_vld[i]),
    .o_r(ready_int_out[i])
  );
  */
  
  axis_data_fifo_slv inst_fifo_int (
    .s_axis_aclk(aclk),
    .s_axis_aresetn(aresetn && ~post),
    .s_axis_tvalid(valid_int[i]),
    .s_axis_tready(tmp_rdy[i]),
    .s_axis_tdata (data_int[i]),
    .m_axis_tvalid(tmp_vld[i]),
    .m_axis_tready(ready_int_out[i]),
    .m_axis_tdata (data_int_out[i])
  );

end

always_ff @(posedge aclk) begin
  if(~aresetn || post) begin
    cnt_done_int <= '0;
    cnt_ready_out <= '0;
  end
  else begin
    for(int i = 0; i < N_INT_MS; i++) begin
      cnt_done_int[i] <= done[i] ? cnt_done_int[i] + 1 : cnt_done_int[i];
      cnt_ready_out[i] <= ready_int_out[i] ? cnt_ready_out[i] + 1 : cnt_ready_out[i];
    end
  end
end

vio_cnt_done inst_vio_cnt_done (
  .clk(aclk),
  .probe_in0(cnt_done_int[0]),
  .probe_in1(cnt_done_int[1]),
  .probe_in2(cnt_done_int[2]),
  .probe_in3(cnt_done_int[3]),
  .probe_in4(cnt_done_int[4]),
  .probe_in5(cnt_done_int[5]),
  .probe_in6(cnt_done_int[6]),
  .probe_in7(cnt_done_int[7]),
  .probe_in8(cnt_done_int[8]),
  .probe_in9 (cnt_ready_out[0]),
  .probe_in10(cnt_ready_out[1]),
  .probe_in11(cnt_ready_out[2]),
  .probe_in12(cnt_ready_out[3]),
  .probe_in13(cnt_ready_out[4]),
  .probe_in14(cnt_ready_out[5]),
  .probe_in15(cnt_ready_out[6]),
  .probe_in16(cnt_ready_out[7]),
  .probe_in17(cnt_ready_out[8])
);

// IO
assign f_ctrl_fs.tvalid = post;
assign f_ctrl_se.tvalid = post;
assign f_ctrl_fs.tdata = {slv_reg[LEN_FRAME_REG][LEN_BITS-1:0], slv_reg[N_FRAMES_REG][CNT_BITS-1:0], slv_reg[SRC_OFFS_REG][ADDR_BITS-1:0]};
assign f_ctrl_se.tdata = slv_reg[DST_OFFS_REG][ADDR_BITS-1:0];
assign n_layers = slv_reg[N_LAYERS_REG][CNT_BITS-1:0];

// --------------------------------------------------------------------------------------
// AXI CTRL  
// -------------------------------------------------------------------------------------- 
// Don't edit

// I/O
assign axi_ctrl.awready = axi_awready;
assign axi_ctrl.arready = axi_arready;
assign axi_ctrl.bresp = axi_bresp;
assign axi_ctrl.bvalid = axi_bvalid;
assign axi_ctrl.wready = axi_wready;
assign axi_ctrl.rdata = axi_rdata;
assign axi_ctrl.rresp = axi_rresp;
assign axi_ctrl.rvalid = axi_rvalid;

// awready and awaddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_awready <= 1'b0;
      axi_awaddr <= 0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && axi_ctrl.awvalid && axi_ctrl.wvalid && aw_en)
        begin
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
          axi_awaddr <= axi_ctrl.awaddr;
        end
      else if (axi_ctrl.bready && axi_bvalid)
        begin
          aw_en <= 1'b1;
          axi_awready <= 1'b0;
        end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end  

// arready and araddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_arready <= 1'b0;
      axi_araddr  <= 0;
    end 
  else
    begin    
      if (~axi_arready && axi_ctrl.arvalid)
        begin
          axi_arready <= 1'b1;
          axi_araddr  <= axi_ctrl.araddr;
        end
      else
        begin
          axi_arready <= 1'b0;
        end
    end 
end    

// bvalid and bresp
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && axi_ctrl.awvalid && ~axi_bvalid && axi_wready && axi_ctrl.wvalid)
        begin
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0;
        end                   
      else
        begin
          if (axi_ctrl.bready && axi_bvalid) 
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end

// wready
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && axi_ctrl.wvalid && axi_ctrl.awvalid && aw_en )
        begin
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end  

// rvalid and rresp (1Del?)
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_rvalid <= 0;
      axi_rresp  <= 0;
    end 
  else
    begin    
      if (axi_arready && axi_ctrl.arvalid && ~axi_rvalid)
        begin
          axi_rvalid <= 1'b1;
          axi_rresp  <= 2'b0;
        end   
      else if (axi_rvalid && axi_ctrl.rready)
        begin
          axi_rvalid <= 1'b0;
        end                
    end
end    

endmodule // gbm slave
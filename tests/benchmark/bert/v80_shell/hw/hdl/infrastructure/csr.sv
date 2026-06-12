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

module csr #(
    parameter                                           LEN_BITS = HBM_LEN_BITS,
    parameter                                           ADDR_BITS = HBM_ADDR_BITS,
    parameter                                           MAX_INTERVALS = 256
) (
    input  logic                                        aclk,
    input  logic                                        aresetn,

    AXI4L.slave                                         axi_ctrl,

    AXI4S.master                                        rd_ctrl,
    AXI4S.master                                        wr_ctrl,
    input  logic                                        rd_done,
    input  logic                                        wr_done
);

localparam integer N_REGS = 10;

// ! REGMAP_START !
localparam integer CTRL_REG = 0;
localparam integer N_RUNS_REG = 1;
localparam integer PERF_LAT_REG = 2;
localparam integer PERF_INT_REG = 3;
localparam integer CH_CNFG_RD_ADDR_OFFS = 4;
localparam integer CH_CNFG_WR_ADDR_OFFS = 5;
localparam integer CH_CNFG_RD_LEN_OFFS = 6;
localparam integer CH_CNFG_WR_LEN_OFFS = 7;
localparam integer CH_CNFG_RD_DONE_OFFS = 8;
localparam integer CH_CNFG_WR_DONE_OFFS = 9;
// ! REGMAP_END !

// -- Decl ----------------------------------------------------------
// ------------------------------------------------------------------
// Constants
localparam integer AXIL_DATA_BITS = 64;
localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer AXIL_ADDR_BITS = ADDR_LSB + ADDR_MSB;

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
logic cnt_up;
logic [63:0] cnt_int;

logic [15:0] cnt_done_rd;
logic [15:0] cnt_done_wr;

logic valid_int;
logic [63:0] data_int;
logic ready_int_out;
logic [63:0] data_int_out;

logic [15:0] cnt_sent_rd;
logic [15:0] cnt_sent_wr;

logic rd_valid;
logic rd_ready;
logic wr_valid;
logic wr_ready;

logic rd_done_int;
logic wr_done_int;

// -- Def -----------------------------------------------------------
// ------------------------------------------------------------------

// Write process
assign slv_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 ) begin
    slv_reg <= 0;
    slv_reg[CH_CNFG_RD_LEN_OFFS] <= ILEN;
    slv_reg[CH_CNFG_WR_LEN_OFFS] <= OLEN;

    ready_int_out <= 1'b0;

    cnt_up <= 0;
    cnt_int <= 0;
    cnt_done_rd <= '0;
    cnt_done_wr <= '0;

    valid_int <= 1'b0;
    data_int <= 'X;

    cnt_sent_rd <= 0;
    cnt_sent_wr <= 0;
    
    rd_valid <= 1'b0;
    wr_valid <= 1'b0;
  end
  else begin
    // Control
    slv_reg[CTRL_REG] <= '0;
    ready_int_out <= 1'b0;

    // Counters and status
    cnt_done_rd <= slv_reg[CTRL_REG][0] ? 0 : (rd_done_int ? cnt_done_rd + 1 : cnt_done_rd);
    cnt_done_wr <= slv_reg[CTRL_REG][0] ? 0 : (wr_done_int ? cnt_done_wr + 1 : cnt_done_wr);

    cnt_up <= slv_reg[CTRL_REG][0] ? 1'b1 : ((cnt_done_wr == slv_reg[N_RUNS_REG] - 1) && wr_done_int) ? 1'b0 : cnt_up;
    slv_reg[PERF_LAT_REG] <= slv_reg[CTRL_REG][0] ? 0 : (cnt_up ? slv_reg[PERF_LAT_REG] + 1 : slv_reg[PERF_LAT_REG]);

    cnt_int <= slv_reg[CTRL_REG][0] ? 0 : (cnt_up ? (wr_done_int ? 0 : cnt_int + 1) : cnt_int);
    valid_int <= (cnt_up && wr_done_int);
    data_int <= cnt_int + 1;
    slv_reg[PERF_INT_REG] <= data_int_out;

    // DMA control
    cnt_sent_rd <= (rd_valid && rd_ready) ? ((cnt_sent_rd == slv_reg[N_RUNS_REG]-1) ? 0 : cnt_sent_rd + 1) : cnt_sent_rd;
    rd_valid <= slv_reg[CTRL_REG][0] ? 1'b1 : (((cnt_sent_rd == slv_reg[N_RUNS_REG]-1) && rd_ready) ? 1'b0 : rd_valid);

    cnt_sent_wr <= (wr_valid && wr_ready) ? ((cnt_sent_wr == slv_reg[N_RUNS_REG]-1) ? 0 : cnt_sent_wr + 1) : cnt_sent_wr;
    wr_valid <= slv_reg[CTRL_REG][0] ? 1'b1 : (((cnt_sent_wr == slv_reg[N_RUNS_REG]-1) && wr_ready) ? 1'b0 : wr_valid);

    slv_reg[CH_CNFG_RD_ADDR_OFFS] <= (rd_valid && rd_ready) ? slv_reg[CH_CNFG_RD_ADDR_OFFS] + slv_reg[CH_CNFG_RD_LEN_OFFS] : slv_reg[CH_CNFG_RD_ADDR_OFFS];
    slv_reg[CH_CNFG_WR_ADDR_OFFS] <= (wr_valid && wr_ready) ? slv_reg[CH_CNFG_WR_ADDR_OFFS] + slv_reg[CH_CNFG_WR_LEN_OFFS] : slv_reg[CH_CNFG_WR_ADDR_OFFS];

    if(slv_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB]) inside
        [CTRL_REG:CTRL_REG]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
            end
        [N_RUNS_REG:N_RUNS_REG]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
            end
        [PERF_INT_REG:PERF_INT_REG]:
            if(axi_ctrl.wstrb[0]) begin
                ready_int_out <= axi_ctrl.wdata[0];
            end

        [CH_CNFG_RD_ADDR_OFFS:CH_CNFG_RD_ADDR_OFFS]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
            end
        [CH_CNFG_WR_ADDR_OFFS:CH_CNFG_WR_ADDR_OFFS]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
            end
        [CH_CNFG_RD_LEN_OFFS:CH_CNFG_RD_LEN_OFFS]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
            end
        [CH_CNFG_WR_LEN_OFFS:CH_CNFG_WR_LEN_OFFS]:
            for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
                if(axi_ctrl.wstrb[i]) begin
                    slv_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
                end
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
            axi_rdata[0] <= (cnt_done_wr[0] == slv_reg[N_RUNS_REG]);
        [N_RUNS_REG:N_RUNS_REG]:
            axi_rdata <= slv_reg[N_RUNS_REG];
        [PERF_LAT_REG:PERF_LAT_REG]:
            axi_rdata <= slv_reg[PERF_LAT_REG];
        [PERF_INT_REG:PERF_INT_REG]:
            axi_rdata <= slv_reg[PERF_INT_REG];
        
        [CH_CNFG_RD_ADDR_OFFS:CH_CNFG_RD_ADDR_OFFS]:
            axi_rdata <= slv_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        [CH_CNFG_WR_ADDR_OFFS:CH_CNFG_WR_ADDR_OFFS]:
            axi_rdata <= slv_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        [CH_CNFG_RD_LEN_OFFS:CH_CNFG_RD_LEN_OFFS]:
            axi_rdata <= slv_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        [CH_CNFG_WR_LEN_OFFS:CH_CNFG_WR_LEN_OFFS]:
            axi_rdata <= slv_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        
        [CH_CNFG_RD_DONE_OFFS:CH_CNFG_RD_DONE_OFFS]:
            axi_rdata <= cnt_done_rd;
        [CH_CNFG_WR_DONE_OFFS:CH_CNFG_WR_DONE_OFFS]:
            axi_rdata <= cnt_done_wr;
        
        default: ;
      endcase
    end
  end 
end

// I/O

// Interval queue
logic tmp_rdy, tmp_vld;

Q_srl #(
    .depth(MAX_INTERVALS),
    .width(64)
) inst_int_fifo (
    .clock(aclk),
    .reset(~aresetn || slv_reg[CTRL_REG][0]),
    .i_d(data_int),
    .i_v(valid_int),
    .i_r(tmp_rdy),
    .o_d(data_int_out),
    .o_v(tmp_vld),
    .o_r(ready_int_out)
);

AXI4S #(.AXI4S_DATA_BITS(CDMA_CTRL_BITS)) rd_ctrl_int ();
AXI4S #(.AXI4S_DATA_BITS(CDMA_CTRL_BITS)) wr_ctrl_int ();

// DMA control
Q_srl #(
    .depth(8),
    .width(LEN_BITS+ADDR_BITS)
) isnt_dma_rd_fifo (
    .clock(aclk),
    .reset(~aresetn),
    .i_d({slv_reg[CH_CNFG_RD_LEN_OFFS][LEN_BITS-1:0], slv_reg[CH_CNFG_RD_ADDR_OFFS][ADDR_BITS-1:0]}),
    .i_v(rd_valid),
    .i_r(rd_ready),
    .o_d(rd_ctrl_int.tdata),
    .o_v(rd_ctrl_int.tvalid),
    .o_r(rd_ctrl_int.tready)
);

axis_reg_array_rtl #(.N_STAGES(1), .DATA_BITS(ADDR_BITS+LEN_BITS)) inst_reg_rd_ctrl (
  .aclk(aclk), .aresetn(aresetn), .s_axis(rd_ctrl_int), .m_axis(rd_ctrl)
);

Q_srl #(
    .depth(8),
    .width(LEN_BITS+ADDR_BITS)
) isnt_dma_wr_fifo (
    .clock(aclk),
    .reset(~aresetn),
    .i_d({slv_reg[CH_CNFG_WR_LEN_OFFS][LEN_BITS-1:0], slv_reg[CH_CNFG_WR_ADDR_OFFS][ADDR_BITS-1:0]}),
    .i_v(wr_valid),
    .i_r(wr_ready),
    .o_d(wr_ctrl_int.tdata),
    .o_v(wr_ctrl_int.tvalid),
    .o_r(wr_ctrl_int.tready)
);

axis_reg_array_rtl #(.N_STAGES(1), .DATA_BITS(ADDR_BITS+LEN_BITS)) inst_reg_wr_ctrl (
  .aclk(aclk), .aresetn(aresetn), .s_axis(wr_ctrl_int), .m_axis(wr_ctrl)
);

logic_array #(.N_STAGES(4), .DATA_BITS(1)) inst_rd_done_reg (.aclk(aclk), .aresetn(aresetn), .s_tdata(rd_done), .m_tdata(rd_done_int));
logic_array #(.N_STAGES(4), .DATA_BITS(1)) inst_wr_done_reg (.aclk(aclk), .aresetn(aresetn), .s_tdata(wr_done), .m_tdata(wr_done_int));

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

endmodule // CSR
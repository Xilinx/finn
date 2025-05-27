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

module fetch_weights #(
    parameter int unsigned              PE,
    parameter int unsigned              SIMD,
    parameter int unsigned              MH,
    parameter int unsigned              MW,
    parameter int unsigned              N_REPS,
    parameter int unsigned              WEIGHT_WIDTH = 8,

    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              CNT_BITS = 16,

    parameter logic[ADDR_BITS-1:0]      ADDR_WEIGHTS,
    parameter logic[ADDR_BITS-1:0]      LAYER_OFFS,
    parameter int unsigned              N_MAX_LAYERS,

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              DBG = 0
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    output logic                        m_done,

    // AXI
    output logic[ADDR_BITS-1:0]         m_axi_hbm_araddr,
    output logic[1:0]		            m_axi_hbm_arburst,
    output logic[3:0]		            m_axi_hbm_arcache,
    output logic[1:0]      		        m_axi_hbm_arid,
    output logic[7:0]		            m_axi_hbm_arlen,
    output logic[0:0]		            m_axi_hbm_arlock,
    output logic[2:0]		            m_axi_hbm_arprot,
    output logic[2:0]		            m_axi_hbm_arsize,
    input  logic			            m_axi_hbm_arready,
    output logic			            m_axi_hbm_arvalid,
    output logic[ADDR_BITS-1:0] 	    m_axi_hbm_awaddr,
    output logic[1:0]		            m_axi_hbm_awburst,
    output logic[3:0]		            m_axi_hbm_awcache,
    output logic[1:0]		            m_axi_hbm_awid,
    output logic[7:0]		            m_axi_hbm_awlen,
    output logic[0:0]		            m_axi_hbm_awlock,
    output logic[2:0]		            m_axi_hbm_awprot,
    output logic[2:0]		            m_axi_hbm_awsize,
    input  logic			            m_axi_hbm_awready,
    output logic			            m_axi_hbm_awvalid,
    input  logic[DATA_BITS-1:0] 	    m_axi_hbm_rdata,
    input  logic[1:0]      		        m_axi_hbm_rid,
    input  logic			            m_axi_hbm_rlast,
    input  logic[1:0]		            m_axi_hbm_rresp,
    output logic 			            m_axi_hbm_rready,
    input  logic			            m_axi_hbm_rvalid,
    output logic[DATA_BITS-1:0] 	    m_axi_hbm_wdata,
    output logic			            m_axi_hbm_wlast,
    output logic[DATA_BITS/8-1:0] 	    m_axi_hbm_wstrb,
    input  logic			            m_axi_hbm_wready,
    output logic			            m_axi_hbm_wvalid,
    input  logic[1:0]      		        m_axi_hbm_bid,
    input  logic[1:0]		            m_axi_hbm_bresp,
    output logic			            m_axi_hbm_bready,
    input  logic			            m_axi_hbm_bvalid,

    // Index
    input  logic                        s_idx_tvalid,
    output logic                        s_idx_tready,
    input  logic[2*CNT_BITS-1:0]        s_idx_tdata,

    // Stream
    // TODO: Should we reg this? Would be quite wide ...
    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready,
    output logic[2*CNT_BITS-1:0]        m_axis_tdata
);

AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS)) s_idx ();
`AXIS_ASSIGN_S2I(s_idx, s_idx)

AXI4S #(.AXI4S_DATA_BITS(PE*SIMD*WEIGHT_WIDTH)) m_axis ();
`AXIS_ASSIGN_I2S(m_axis, m_axis)

AXI4 #(.AXI4_DATA_BITS(DATA_BITS), .AXI4_ADDR_BITS(ADDR_BITS)) m_axi_hbm ();
`AXI_ASSIGN_I2S(m_axi_hbm, m_axi_hbm)

// Offsets
logic [N_MAX_LAYERS-1:0][ADDR_BITS-1:0] l_offsets;
for(genvar i = 0; i < N_MAX_LAYERS; i++) begin
    assign l_offsets[i] = ADDR_WEIGHTS | (i * LAYER_OFFS);
end

AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS)) q_idx_out ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma_out ();

// Queues
queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS)) inst_queue_in (.aclk(aclk), .aresetn(aresetn), .s_axis(s_idx), .m_axis(q_idx_out));
queue #(.QDEPTH(QDEPTH), .QWIDTH(ADDR_BITS+LEN_BITS)) inst_queue_dma (.aclk(aclk), .aresetn(aresetn), .s_axis(q_dma), .m_axis(q_dma_out));

// Regs
typedef enum logic[0:0] {ST_IDLE, ST_READ} state_t;
state_t state_C = ST_IDLE, state_N;

logic [CNT_BITS-1:0] cnt_frames_C = '0, cnt_frames_N;
logic [CNT_BITS-1:0] n_frames_C = '0, n_frames_N;
logic [ADDR_BITS-1:0] addr_C = '0, addr_N;
logic [LEN_BITS-1:0] len_C = '0, len_N;
logic [CNT_BITS-1:0] layer_C = '0, layer_N;

always_ff @( posedge aclk ) begin : REG
    if(~aresetn) begin
        state_C <= ST_IDLE;

        cnt_frames_C <= 'X;
        n_frames_C <= 'X;
        addr_C <= 'X;
        len_C <= 'X;
    end
    else begin
        state_C <= state_N;

        cnt_frames_C <= cnt_frames_N;
        n_frames_C <= n_frames_N;
        addr_C <= addr_N;
        len_C <= len_N;
    end
end

always_comb begin : NSL
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = q_idx_out.tvalid ? ST_READ : ST_IDLE;

        ST_READ:
            state_N = ((cnt_frames_C == n_frames_C - 1) && q_dma.tready) ? ST_IDLE : ST_READ;

    endcase
end

always_comb begin : DP
    // AL
    cnt_frames_N = cnt_frames_C;
    n_frames_N = n_frames_C;
    addr_N = addr_C;
    len_N = len_C;

    // S
    q_idx_out.tready = 1'b0;
    q_dma.tvalid = 1'b0;
    q_dma.tdata = {len_C, addr_C};

    // RD
    case (state_C)
        ST_IDLE: begin
            q_idx_out.tready = 1'b1;
            if(q_idx_out.tvalid) begin
                cnt_frames_N = 0;
                layer_N = q_idx_out.tdata[0+:CNT_BITS];
                n_frames_N = q_idx_out.tdata[CNT_BITS+:CNT_BITS];
                len_N = MH * MW;
                addr_N = l_offsets[q_idx_out.tdata[0+:CNT_BITS]];
            end
        end

        ST_READ: begin
            q_dma.tvalid = 1'b1;
            if(q_dma.tready) begin
                cnt_frames_N = cnt_frames_C + 1;
            end
        end

    endcase
end

// DMA
AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) axis_tmp_f ();
AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) axis_dma_f ();
AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) axis_dma ();
AXI4S #(.AXI4S_DATA_BITS(PE*WEIGHT_WIDTH)) axis_dwc_lwb ();

cdma_top #(
    .ADDR_BITS(ADDR_BITS),
    .DATA_BITS(DATA_BITS),
    .LEN_BITS(LEN_BITS),
    .CDMA_RD(1),
    .CDMA_WR(0)
) inst_dma (
    .aclk(aclk),
    .aresetn(aresetn),
    .rd_valid(q_dma_out.tvalid),
    .rd_ready(q_dma_out.tready),
    .rd_paddr(q_dma_out.tdata[0+:ADDR_BITS]),
    .rd_len(q_dma_out.tdata[ADDR_BITS+:LEN_BITS]),
    .rd_done(m_done),
    .wr_valid(1'b0),
    .wr_ready(),
    .wr_paddr(0),
    .wr_len(0),
    .wr_done(),
    .m_axi_ddr(m_axi_hbm),
    .s_axis_ddr(axis_tmp_f),
    .m_axis_ddr(axis_dma_f)
);
`AXISF_TIE_OFF_M(axis_tmp_f)
`AXISF_AXIS_ASSIGN(axis_dma_f, axis_dma)

// Width conversion
axis_dwc #(.S_DATA_BITS(DATA_BITS), .M_DATA_BITS(PE*WEIGHT_WIDTH)) inst_dwc (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_dma), .m_axis(axis_dwc_lwb));

// Double buffer
AXI4S #(.AXI4S_DATA_BITS(PE*SIMD*WEIGHT_WIDTH)) m_axis_int ();

local_weight_buffer #(
    .PE(PE), .SIMD(SIMD), .MH(MH), .MW(MW), .N_REPS(N_REPS), .WEIGHT_WIDTH(WEIGHT_WIDTH), .DBG(DBG)
) inst_weight_buff (
    .clk(aclk), .rst(~aresetn),
    .ivld(axis_dwc_lwb.tvalid), .irdy(axis_dwc_lwb.tready), .idat(axis_dwc_lwb.tdata),
    .ovld(m_axis_int.tvalid), .ordy(m_axis_int.tready), .odat(m_axis_int.tdata)
);

// Reg slice
axis_reg_array_rtl #(.N_STAGES(1), .DATA_BITS(PE*SIMD*WEIGHT_WIDTH)) inst_reg_out (.aclk(aclk), .aresetn(aresetn), .s_axis(m_axis_int), .m_axis(m_axis));

endmodule

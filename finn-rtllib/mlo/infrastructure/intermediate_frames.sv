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

module intermediate_frames #(
    parameter int unsigned              ILEN_BITS,
    parameter int unsigned              OLEN_BITS,

    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              IDX_BITS = 16,

    parameter logic[LEN_BITS-1:0]       FM_SIZE,

    parameter int unsigned              N_OUTSTANDING_DMAS = 128,

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              DBG = 0
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    // MM
    output logic [ADDR_BITS-1:0]        m_axi_ddr_araddr,
    output logic [1:0]                  m_axi_ddr_arburst,
    output logic [3:0]                  m_axi_ddr_arcache,
    output logic [1:0]                  m_axi_ddr_arid,
    output logic [7:0]                  m_axi_ddr_arlen,
    output logic                        m_axi_ddr_arlock,
    output logic [2:0]                  m_axi_ddr_arprot,
    output logic [2:0]                  m_axi_ddr_arsize,
    input  logic                        m_axi_ddr_arready,
    output logic                        m_axi_ddr_arvalid,
    output logic [ADDR_BITS-1:0]        m_axi_ddr_awaddr,
    output logic [1:0]                  m_axi_ddr_awburst,
    output logic [3:0]                  m_axi_ddr_awcache,
    output logic [1:0]                  m_axi_ddr_awid,
    output logic [7:0]                  m_axi_ddr_awlen,
    output logic                        m_axi_ddr_awlock,
    output logic [2:0]                  m_axi_ddr_awprot,
    output logic [2:0]                  m_axi_ddr_awsize,
    input  logic                        m_axi_ddr_awready,
    output logic                        m_axi_ddr_awvalid,
    input  logic [DATA_BITS-1:0]        m_axi_ddr_rdata,
    input  logic [1:0]                  m_axi_ddr_rid,
    input  logic                        m_axi_ddr_rlast,
    input  logic [1:0]                  m_axi_ddr_rresp,
    output logic                        m_axi_ddr_rready,
    input  logic                        m_axi_ddr_rvalid,
    output logic [DATA_BITS-1:0]        m_axi_ddr_wdata,
    output logic                        m_axi_ddr_wlast,
    output logic [DATA_BITS/8-1:0]      m_axi_ddr_wstrb,
    input  logic                        m_axi_ddr_wready,
    output logic                        m_axi_ddr_wvalid,
    input  logic [1:0]                  m_axi_ddr_bid,
    input  logic [1:0]                  m_axi_ddr_bresp,
    output logic                        m_axi_ddr_bready,
    input  logic                        m_axi_ddr_bvalid,

    // Idx
    input  logic [IDX_BITS-1:0]         s_idx_tdata,
    input  logic                        s_idx_tvalid,
    output logic                        s_idx_tready,

    output logic [IDX_BITS-1:0]         m_idx_tdata,
    output logic                        m_idx_tvalid,
    input  logic                        m_idx_tready,

    // Data
    input  logic [OLEN_BITS-1:0]        s_axis_tdata,
    input  logic                        s_axis_tvalid,
    output logic                        s_axis_tready,

    output logic [ILEN_BITS-1:0]        m_axis_tdata,
    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready
);

// Offsets
logic [N_OUTSTANDING_DMAS-1:0][ADDR_BITS-1:0] l_offsets;
for(genvar i = 0; i < N_OUTSTANDING_DMAS; i++) begin
    assign l_offsets[i] = (i * FM_SIZE);
end
localparam integer N_OUTSTANDING_DMAS_BITS = $clog2(N_OUTSTANDING_DMAS);

localparam integer FM_BEATS_IN = FM_SIZE/(OLEN_BITS/8);
localparam integer FM_BEATS_IN_BITS = (FM_BEATS_IN == 1) ? 1 : $clog2(FM_BEATS_IN);

//
// Write side
//

// Input queue
logic idx_in_tvalid, idx_in_tready;
logic [IDX_BITS-1:0] idx_in_tdata;

Q_srl #(
    .depth(QDEPTH), .width(IDX_BITS)
) inst_queue_seq (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(s_idx_tdata), .i_v(s_idx_tvalid), .i_r(s_idx_tready),
    .o_d(idx_in_tdata), .o_v(idx_in_tvalid), .o_r(idx_in_tready)
);

// Circ buff
logic wr_sent, wr_rdy;
logic rd_done;

Q_srl #(
    .depth(N_OUTSTANDING_DMAS), .width(1)
) inst_queue_outstanding (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(1'b1), .i_v(wr_sent), .i_r(wr_rdy),
    .o_d(), .o_v(), .o_r(rd_done)
);

// FSM
typedef enum logic[0:0] {ST_WR_IDLE, ST_WR_SEND} state_wr_t;
state_wr_t state_wr_C = ST_WR_IDLE, state_wr_N;

logic [N_OUTSTANDING_DMAS_BITS-1:0] wr_ptr_C = '0, wr_ptr_N;

logic s0_dma_in_tvalid, s0_dma_in_tready;
logic [ADDR_BITS-1:0] s0_dma_in_tdata;
logic s0_dma_out_tvalid, s0_dma_out_tready;
logic [ADDR_BITS-1:0] s0_dma_out_tdata;

Q_srl #(
    .depth(QDEPTH), .width(ADDR_BITS)
) inst_queue_s0_dma (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(s0_dma_in_tdata), .i_v(s0_dma_in_tvalid), .i_r(s0_dma_in_tready),
    .o_d(s0_dma_out_tdata), .o_v(s0_dma_out_tvalid), .o_r(s0_dma_out_tready)
);

always_ff @(posedge aclk) begin: REG_WR
    if(~aresetn) begin
        state_wr_C <= ST_WR_IDLE;
        wr_ptr_C <= '0;
    end else begin
        state_wr_C <= state_wr_N;
        wr_ptr_C <= wr_ptr_N;
    end
end

always_comb begin: NSL_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_IDLE:
            state_wr_N = (idx_in_tvalid && m_idx_tready) ? ST_WR_SEND : ST_WR_IDLE;

        ST_WR_SEND:
            state_wr_N = (wr_rdy && s0_dma_in_tready) ? ST_WR_IDLE : ST_WR_SEND;

    endcase
end

always_comb begin: DP_WR
    wr_ptr_N = wr_ptr_C;

    idx_in_tready = 1'b0;
    m_idx_tvalid = 1'b0;
    m_idx_tdata = idx_in_tdata + 1;

    s0_dma_in_tvalid = 1'b0;
    s0_dma_in_tdata = l_offsets[wr_ptr_C];
    wr_sent = 1'b0;

    case (state_wr_C)
        ST_WR_IDLE: begin
            if(idx_in_tvalid) begin
                m_idx_tvalid = 1'b1;

                if(m_idx_tready) begin
                    idx_in_tready = 1'b1;
                end
            end
        end

        ST_WR_SEND: begin
            if(wr_rdy) begin
                s0_dma_in_tvalid = 1'b1;

                if(s0_dma_in_tready) begin
                    wr_sent = 1'b1;
                    wr_ptr_N = (wr_ptr_C == N_OUTSTANDING_DMAS-1) ? 0 : wr_ptr_C + 1;
                end
            end
        end

    endcase
end

//
// Completion queue
//

logic done_wr_in, done_wr_out;
logic rd_start;

Q_srl #(
    .depth(N_OUTSTANDING_DMAS), .width(1)
) inst_queue_done (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(1'b1), .i_v(done_wr_in), .i_r(),
    .o_d(), .o_v(done_wr_out), .o_r(rd_start)
);

//
// Read side
//

typedef enum logic[0:0] {ST_RD_IDLE, ST_RD_SEND} state_rd_t;
state_rd_t state_rd_C = ST_RD_IDLE, state_rd_N;

logic [N_OUTSTANDING_DMAS_BITS-1:0] rd_ptr_C = '0, rd_ptr_N;

logic s1_dma_in_tvalid, s1_dma_in_tready;
logic [ADDR_BITS-1:0] s1_dma_in_tdata;
logic s1_dma_out_tvalid, s1_dma_out_tready;
logic [ADDR_BITS-1:0] s1_dma_out_tdata;

Q_srl #(
    .depth(QDEPTH), .width(ADDR_BITS)
) inst_queue_s1_dma (
    .clock(aclk), .reset(!aresetn),
    .count(), .maxcount(),
    .i_d(s1_dma_in_tdata), .i_v(s1_dma_in_tvalid), .i_r(s1_dma_in_tready),
    .o_d(s1_dma_out_tdata), .o_v(s1_dma_out_tvalid), .o_r(s1_dma_out_tready)
);

always_ff @(posedge aclk) begin: REG_RD
    if(~aresetn) begin
        state_rd_C <= ST_RD_IDLE;
        rd_ptr_C <= '0;
    end else begin
        state_rd_C <= state_rd_N;
        rd_ptr_C <= rd_ptr_N;
    end
end

always_comb begin: NSL_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_IDLE:
            state_rd_N = done_wr_out ? ST_RD_SEND : ST_RD_IDLE;

        ST_RD_SEND:
            state_rd_N = s1_dma_in_tready ? ST_RD_IDLE : ST_RD_SEND;

    endcase
end

always_comb begin: DP_RD
    rd_ptr_N = rd_ptr_C;

    rd_start = 1'b0;
    s1_dma_in_tvalid = 1'b0;
    s1_dma_in_tdata = l_offsets[rd_ptr_C];

    case (state_rd_C)
        ST_RD_IDLE: begin
            if(done_wr_out) begin
                rd_start = 1'b1;
            end
        end

        ST_RD_SEND: begin
            s1_dma_in_tvalid = 1'b1;

            if(s1_dma_in_tready) begin
                rd_ptr_N = (rd_ptr_C == N_OUTSTANDING_DMAS-1) ? 0 : rd_ptr_C + 1;
            end
        end

    endcase
end

//
// DMA
//

logic axis_dma_rd_tvalid, axis_dma_rd_tready;
logic [DATA_BITS-1:0] axis_dma_rd_tdata;
logic [DATA_BITS/8-1:0] axis_dma_rd_tkeep;
logic axis_dma_rd_tlast;

logic axis_dma_wr_tvalid, axis_dma_wr_tready;
logic [DATA_BITS-1:0] axis_dma_wr_tdata;
logic [DATA_BITS/8-1:0] axis_dma_wr_tkeep;
logic axis_dma_wr_tlast;

cdma_u #(
    .ADDR_BITS(ADDR_BITS),
    .LEN_BITS(LEN_BITS),
    .DATA_BITS(DATA_BITS)
) inst_dma (
    .aclk(aclk),
    .aresetn(aresetn),

    .m_axi_ddr_arvalid(m_axi_ddr_arvalid),
    .m_axi_ddr_arready(m_axi_ddr_arready),
    .m_axi_ddr_araddr(m_axi_ddr_araddr),
    .m_axi_ddr_arid(m_axi_ddr_arid),
    .m_axi_ddr_arlen(m_axi_ddr_arlen),
    .m_axi_ddr_arsize(m_axi_ddr_arsize),
    .m_axi_ddr_arburst(m_axi_ddr_arburst),
    .m_axi_ddr_arlock(m_axi_ddr_arlock),
    .m_axi_ddr_arcache(m_axi_ddr_arcache),
    .m_axi_ddr_arprot(m_axi_ddr_arprot),
    .m_axi_ddr_rvalid(m_axi_ddr_rvalid),
    .m_axi_ddr_rready(m_axi_ddr_rready),
    .m_axi_ddr_rdata(m_axi_ddr_rdata),
    .m_axi_ddr_rlast(m_axi_ddr_rlast),
    .m_axi_ddr_rid(m_axi_ddr_rid),
    .m_axi_ddr_rresp(m_axi_ddr_rresp),
    .m_axi_ddr_awvalid(m_axi_ddr_awvalid),
    .m_axi_ddr_awready(m_axi_ddr_awready),
    .m_axi_ddr_awaddr(m_axi_ddr_awaddr),
    .m_axi_ddr_awid(m_axi_ddr_awid),
    .m_axi_ddr_awlen(m_axi_ddr_awlen),
    .m_axi_ddr_awsize(m_axi_ddr_awsize),
    .m_axi_ddr_awburst(m_axi_ddr_awburst),
    .m_axi_ddr_awlock(m_axi_ddr_awlock),
    .m_axi_ddr_awcache(m_axi_ddr_awcache),
    .m_axi_ddr_wdata(m_axi_ddr_wdata),
    .m_axi_ddr_wstrb(m_axi_ddr_wstrb),
    .m_axi_ddr_wlast(m_axi_ddr_wlast),
    .m_axi_ddr_wvalid(m_axi_ddr_wvalid),
    .m_axi_ddr_wready(m_axi_ddr_wready),
    .m_axi_ddr_bid(m_axi_ddr_bid),
    .m_axi_ddr_bresp(m_axi_ddr_bresp),
    .m_axi_ddr_bvalid(m_axi_ddr_bvalid),
    .m_axi_ddr_bready(m_axi_ddr_bready),

    .rd_valid(s1_dma_out_tvalid),
    .rd_ready(s1_dma_out_tready),
    .rd_paddr(s1_dma_out_tdata),
    .rd_len  (FM_SIZE),
    .rd_done (rd_done),

    .wr_valid(s0_dma_out_tvalid),
    .wr_ready(s0_dma_out_tready),
    .wr_paddr(s0_dma_out_tdata),
    .wr_len  (FM_SIZE),
    .wr_done (done_wr_in),

    .m_axis_ddr_tvalid(axis_dma_rd_tvalid),
    .m_axis_ddr_tready(axis_dma_rd_tready),
    .m_axis_ddr_tdata (axis_dma_rd_tdata),
    .m_axis_ddr_tkeep (axis_dma_rd_tkeep),
    .m_axis_ddr_tlast (axis_dma_rd_tlast),

    .s_axis_ddr_tvalid(axis_dma_wr_tvalid),
    .s_axis_ddr_tready(axis_dma_wr_tready),
    .s_axis_ddr_tdata (axis_dma_wr_tdata),
    .s_axis_ddr_tkeep (axis_dma_wr_tkeep),
    .s_axis_ddr_tlast (axis_dma_wr_tlast)
);

// DWC
logic s_axis_int_tvalid, s_axis_int_tready;
logic [OLEN_BITS-1:0] s_axis_int_tdata;

logic m_axis_int_tvalid, m_axis_int_tready;
logic [OLEN_BITS-1:0] m_axis_int_tdata;

logic [FM_BEATS_IN_BITS-1:0] cnt_dwc_C = '0;
always_ff @(posedge aclk) begin
    if(~aresetn) cnt_dwc_C <= '0;
    else cnt_dwc_C <= (s_axis_int_tvalid && s_axis_int_tready) ? ((cnt_dwc_C == FM_BEATS_IN-1) ? 0 : cnt_dwc_C + 1) : cnt_dwc_C;
end

logic last_dwc_in;
assign last_dwc_in = (cnt_dwc_C == FM_BEATS_IN-1);

axis_dwc #(.S_DATA_BITS(OLEN_BITS), .M_DATA_BITS(DATA_BITS)) inst_dwc_wr (
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tvalid(s_axis_int_tvalid),
    .s_axis_tready(s_axis_int_tready),
    .s_axis_tdata (s_axis_int_tdata),
    .s_axis_tkeep ('1),
    .s_axis_tlast (last_dwc_in),

    .m_axis_tvalid(axis_dma_wr_tvalid),
    .m_axis_tready(axis_dma_wr_tready),
    .m_axis_tdata (axis_dma_wr_tdata),
    .m_axis_tkeep (axis_dma_wr_tkeep),
    .m_axis_tlast (axis_dma_wr_tlast)
);

axis_dwc #(.S_DATA_BITS(DATA_BITS), .M_DATA_BITS(ILEN_BITS)) inst_dwc_rd (
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tvalid(axis_dma_rd_tvalid),
    .s_axis_tready(axis_dma_rd_tready),
    .s_axis_tdata (axis_dma_rd_tdata),
    .s_axis_tkeep (axis_dma_rd_tkeep),
    .s_axis_tlast (axis_dma_rd_tlast),

    .m_axis_tvalid(m_axis_int_tvalid),
    .m_axis_tready(m_axis_int_tready),
    .m_axis_tdata (m_axis_int_tdata),
    .m_axis_tkeep (),
    .m_axis_tlast ()
);

// REG
axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(OLEN_BITS)) inst_reg_wr (
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .s_axis_tdata (s_axis_tdata),

    .m_axis_tvalid(s_axis_int_tvalid),
    .m_axis_tready(s_axis_int_tready),
    .m_axis_tdata (s_axis_int_tdata)
);

axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(ILEN_BITS)) inst_reg_rd (
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tvalid(m_axis_int_tvalid),
    .s_axis_tready(m_axis_int_tready),
    .s_axis_tdata (m_axis_int_tdata),

    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready),
    .m_axis_tdata (m_axis_tdata)
);

endmodule

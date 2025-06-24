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
    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              CNT_BITS = 16,

    parameter logic[ADDR_BITS-1:0]      ADDR_INT,
    parameter logic[ADDR_BITS-1:0]      LAYER_OFFS_INT,
    parameter int unsigned              N_MAX_LAYERS,
    parameter int unsigned              ILEN_BITS,
    parameter int unsigned              OLEN_BITS,
    parameter int unsigned              N_OUTSTANDING_DMAS = 32,
    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1,
    parameter int unsigned              DBG = 0
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    output logic [1:0]                  m_done,

    AXI4.master                         m_axi_hbm,

    AXI4S.slave                         s_idx,
    AXI4S.master                        m_idx,

    AXI4S.slave                         s_axis,
    AXI4S.master                        m_axis
);

// Offsets
logic [N_MAX_LAYERS-1:0][ADDR_BITS-1:0] l_offsets;
for(genvar i = 0; i < N_MAX_LAYERS; i++) begin
    assign l_offsets[i] = ADDR_INT | (i * LAYER_OFFS_INT);
end

// -------------------------------------------------------------------------
// Input queueing - S0
// -------------------------------------------------------------------------

AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s0_dma ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s0_dma_out ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s0_buf ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s0_buf_out ();

queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_dma_s0 (.aclk(aclk), .aresetn(aresetn), .s_axis(q_s0_dma), .m_axis(q_s0_dma_out));
queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_buf_s0 (.aclk(aclk), .aresetn(aresetn), .s_axis(q_s0_buf), .m_axis(q_s0_buf_out));

assign s_idx.tready = q_s0_dma.tready & q_s0_buf.tready;
assign q_s0_dma.tvalid = q_s0_buf.tready & s_idx.tvalid;
assign q_s0_buf.tvalid = q_s0_dma.tready & s_idx.tvalid;
assign q_s0_dma.tdata = s_idx.tdata;
assign q_s0_buf.tdata = s_idx.tdata;

// -------------------------------------------------------------------------
// Store DMA
// -------------------------------------------------------------------------

typedef enum logic[0:0] {ST_WR_IDLE, ST_WR_STORE} state_wr_t;
state_wr_t state_wr_C = ST_WR_IDLE, state_wr_N;

logic [CNT_BITS-1:0] cnt_frames_wr_C = '0, cnt_frames_wr_N;
logic [CNT_BITS-1:0] n_frames_wr_C = '0, n_frames_wr_N;
logic [LEN_BITS-1:0] len_wr_C = '0, len_wr_N;
logic [ADDR_BITS-1:0] addr_wr_C = '0, addr_wr_N;

logic [CNT_BITS-1:0] cnt_outstanding_C = N_OUTSTANDING_DMAS;
logic cnt_outstanding_decr, cnt_outstanding_incr;

// -------------------------------------------------------------------------

AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_wr_dma ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_wr_dma_out ();
AXI4S #(.AXI4S_DATA_BITS(1)) q_wr_fr_done ();
AXI4S #(.AXI4S_DATA_BITS(1)) q_wr_fr_done_out ();

// DMA write queue
queue #(.QDEPTH(QDEPTH), .QWIDTH(ADDR_BITS+LEN_BITS)) inst_wr_dma (.aclk(aclk), .aresetn(aresetn), .s_axis(q_wr_dma), .m_axis(q_wr_dma_out));
// Frame done queue (signal fetch DMA)
queue #(.QDEPTH(2*N_OUTSTANDING_DMAS), .QWIDTH(1)) inst_wr_fr_done (.aclk(aclk), .aresetn(aresetn), .s_axis(q_wr_fr_done), .m_axis(q_wr_fr_done_out));

assign q_wr_fr_done.tvalid = wr_done;
assign q_wr_fr_done.tdata = 1'b1;

// -------------------------------------------------------------------------

always_ff @( posedge aclk ) begin : REG_WR
    if(~aresetn) begin
        state_wr_C <= ST_WR_IDLE;

        cnt_frames_wr_C <= 'X;
        n_frames_wr_C <= 'X;
        len_wr_C <= 'X;
        addr_wr_C <= 'X;
    end
    else begin
        state_wr_C <= state_wr_N;

        cnt_frames_wr_C <= cnt_frames_wr_N;
        n_frames_wr_C <= n_frames_wr_N;
        len_wr_C <= len_wr_N;
        addr_wr_C <= addr_wr_N;
    end
end

always_comb begin : NSL_WR
    state_wr_N = state_wr_C;

    case (state_wr_C)
        ST_WR_IDLE:
            state_wr_N = q_s0_dma_out.tvalid ? ST_WR_STORE : ST_WR_IDLE;

        ST_WR_STORE:
            state_wr_N = ((cnt_frames_wr_C == n_frames_wr_C - 1) && (q_wr_dma.tready && cnt_outstanding_C > 0)) ? ST_WR_IDLE : ST_WR_STORE;

    endcase
end

always_comb begin : DP_WR
    cnt_frames_wr_N = cnt_frames_wr_C;
    n_frames_wr_N = n_frames_wr_C;
    len_wr_N = len_wr_C;
    addr_wr_N = addr_wr_C;

    q_s0_dma_out.tready = 1'b0;
    q_wr_dma.tvalid = 1'b0;
    q_wr_dma.tdata = {len_wr_C, addr_wr_C};

    cnt_outstanding_decr = 1'b0;

    case (state_wr_C)
        ST_WR_IDLE: begin
            q_s0_dma_out.tready = 1'b1;
            if(q_s0_dma_out.tvalid) begin
                cnt_frames_wr_N = 0;
                n_frames_wr_N = q_s0_dma_out.tdata[CNT_BITS+:CNT_BITS];
                len_wr_N = q_s0_dma_out.tdata[2*CNT_BITS+:LEN_BITS];
                addr_wr_N = l_offsets[q_s0_dma_out.tdata[0+:CNT_BITS]];
            end
        end

        ST_WR_STORE: begin
            if(q_wr_dma.tready && cnt_outstanding_C > 0) begin
                q_wr_dma.tvalid = 1'b1;
                cnt_outstanding_decr = 1'b1;

                cnt_frames_wr_N = cnt_frames_wr_C + 1;
                addr_wr_N = addr_wr_C + len_wr_C;
            end
        end

    endcase
end

// -------------------------------------------------------------------------
// Queueing - S1
// -------------------------------------------------------------------------

AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s1_dma ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s1_dma_out ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s1_buf ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_s1_buf_out ();
logic [CNT_BITS-1:0] incr_lyr;

queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_dma_s1 (.aclk(aclk), .aresetn(aresetn), .s_axis(q_s1_dma), .m_axis(q_s1_dma_out));
queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_buf_s1 (.aclk(aclk), .aresetn(aresetn), .s_axis(q_s1_buf), .m_axis(m_idx));

assign q_s0_buf_out.tready = q_s1_dma.tready & q_s1_buf.tready;
assign q_s1_dma.tvalid = q_s1_buf.tready & q_s0_buf_out.tvalid;
assign q_s1_buf.tvalid = q_s1_dma.tready & q_s0_buf_out.tvalid;
assign q_s1_dma.tdata = q_s0_buf_out.tdata;
assign incr_lyr = q_s0_buf_out.tdata[0+:CNT_BITS] + 1;
assign q_s1_buf.tdata = {q_s0_buf_out.tdata[CNT_BITS+:CNT_BITS+LEN_BITS], incr_lyr};

// -------------------------------------------------------------------------
// Fetch DMA
// -------------------------------------------------------------------------

typedef enum logic[0:0] {ST_RD_IDLE, ST_RD_FETCH} state_rd_t;
state_rd_t state_rd_C = ST_RD_IDLE, state_rd_N;

logic [CNT_BITS-1:0] cnt_frames_rd_C = '0, cnt_frames_rd_N;
logic [CNT_BITS-1:0] n_frames_rd_C = '0, n_frames_rd_N;
logic [LEN_BITS-1:0] len_rd_C = '0, len_rd_N;
logic [ADDR_BITS-1:0] addr_rd_C = '0, addr_rd_N;

logic rd_done;

// -------------------------------------------------------------------------

AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_rd_dma ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_rd_dma_out ();

// DMA read queue
queue #(.QDEPTH(QDEPTH), .QWIDTH(ADDR_BITS+LEN_BITS)) inst_rd_dma (.aclk(aclk), .aresetn(aresetn), .s_axis(q_rd_dma), .m_axis(q_rd_dma_out));

// -------------------------------------------------------------------------

always_ff @( posedge aclk ) begin : REG_RD
    if(~aresetn) begin
        state_rd_C <= ST_RD_IDLE;

        cnt_frames_rd_C <= 'X;
        n_frames_rd_C <= 'X;
        len_rd_C <= 'X;
        addr_rd_C <= 'X;
    end
    else begin
        state_rd_C <= state_rd_N;

        cnt_frames_rd_C <= cnt_frames_rd_N;
        n_frames_rd_C <= n_frames_rd_N;
        len_rd_C <= len_rd_N;
        addr_rd_C <= addr_rd_N;
    end
end

always_comb begin : NSL_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_IDLE:
            state_rd_N = q_s1_dma_out.tvalid ? ST_RD_FETCH : ST_RD_IDLE;

        ST_RD_FETCH: begin
            state_rd_N = ((cnt_frames_rd_C == n_frames_rd_C-1) && (q_rd_dma.tready && q_wr_fr_done_out.tvalid)) ? ST_RD_IDLE : ST_RD_FETCH;
        end

    endcase
end

always_comb begin : DP_RD
    cnt_frames_rd_N = cnt_frames_rd_C;
    n_frames_rd_N = n_frames_rd_C;
    len_rd_N = len_rd_C;
    addr_rd_N = addr_rd_C;

    q_s1_dma_out.tready = 1'b0;
    q_rd_dma.tvalid = 1'b0;
    q_rd_dma.tdata = {len_rd_C, addr_rd_C};
    q_wr_fr_done_out.tready = 1'b0;

    cnt_outstanding_incr = 1'b0;

    case (state_rd_C)
        ST_RD_IDLE: begin
            q_s1_dma_out.tready = 1'b1;
            if(q_s1_dma_out.tvalid) begin
                cnt_frames_rd_N = 0;
                n_frames_rd_N = q_s1_dma_out.tdata[CNT_BITS+:CNT_BITS];
                len_rd_N = q_s1_dma_out.tdata[2*CNT_BITS+:LEN_BITS];
                addr_rd_N = l_offsets[q_s1_dma_out.tdata[0+:CNT_BITS]];
            end
        end

        ST_RD_FETCH: begin
            if(q_rd_dma.tready && q_wr_fr_done_out.tvalid) begin
                q_rd_dma.tvalid = 1'b1;
                q_wr_fr_done_out.tready = 1'b1;
                cnt_outstanding_incr = 1'b1;

                cnt_frames_rd_N = cnt_frames_rd_C + 1;
                addr_rd_N = addr_rd_C + len_rd_C;
            end
        end

    endcase
end

// -------------------------------------------------------------------------
// Sync between DMAs
// -------------------------------------------------------------------------

always_ff @( posedge aclk ) begin : REG_SYNC
    if(~aresetn) begin
        cnt_outstanding_C <= N_OUTSTANDING_DMAS;
    end
    else begin
        cnt_outstanding_C <= (cnt_outstanding_decr & cnt_outstanding_incr) ? cnt_outstanding_C :
                            (cnt_outstanding_decr ? cnt_outstanding_C - 1 :
                            (cnt_outstanding_incr ? cnt_outstanding_C + 1 :
                            cnt_outstanding_C));

    end
end

// -------------------------------------------------------------------------
// Central DMA
// -------------------------------------------------------------------------

AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) dma_rd_f ();
AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) dma_wr_f ();
AXI4SF #(.AXI4S_DATA_BITS(ILEN_BITS)) dma_rd_dwc ();
AXI4SF #(.AXI4S_DATA_BITS(OLEN_BITS)) dma_wr_dwc ();

cdma_top #(
    .ADDR_BITS(ADDR_BITS),
    .LEN_BITS(LEN_BITS),
    .DATA_BITS(DATA_BITS),
    .CDMA_RD(1),
    .CDMA_WR(1)
) inst_dma (
    .aclk(aclk),
    .aresetn(aresetn),

    .m_axi_ddr(m_axi_hbm),

    .rd_valid(q_rd_dma_out.tvalid),
    .rd_ready(q_rd_dma_out.tready),
    .rd_paddr(q_rd_dma_out.tdata[0+:ADDR_BITS]),
    .rd_len(q_rd_dma_out.tdata[ADDR_BITS+:LEN_BITS]),
    .rd_done(rd_done),

    .wr_valid(q_wr_dma_out.tvalid),
    .wr_ready(q_wr_dma_out.tready),
    .wr_paddr(q_wr_dma_out.tdata[0+:ADDR_BITS]),
    .wr_len(q_wr_dma_out.tdata[ADDR_BITS+:LEN_BITS]),
    .wr_done(wr_done),

    .s_axis_ddr(dma_wr_f),
    .m_axis_ddr(dma_rd_f)
);
//`AXIS_AXISF_ASSIGN(dma_wr, dma_wr_f)
//`AXISF_AXIS_ASSIGN(dma_rd_f, dma_rd)
assign m_done = {rd_done, wr_done};

// DWC
axis_dwc #(.S_DATA_BITS(OLEN_BITS), .M_DATA_BITS(DATA_BITS))
          inst_dwc_wr (.aclk(aclk),
                       .aresetn(aresetn),

                       .s_axis_tvalid(dma_wr_dwc.tvalid),
                       .s_axis_tready(dma_wr_dwc.tready),
                       .s_axis_tdata(dma_wr_dwc.tdata),
                       .s_axis_tkeep(dma_wr_dwc.tkeep),
                       .s_axis_tlast(dma_wr_dwc.tlast),

                       .m_axis_tvalid(dma_wr_f.tvalid),
                       .m_axis_tready(dma_wr_f.tready),
                       .m_axis_tdata(dma_wr_f.tdata),
                       .m_axis_tkeep(dma_wr_f.tkeep),
                       .m_axis_tlast(dma_wr_f.tlast)
                    );
axis_dwc #(.S_DATA_BITS(DATA_BITS), .M_DATA_BITS(ILEN_BITS))
         inst_dwc_rd (.aclk(aclk),
                      .aresetn(aresetn),

                      .s_axis_tvalid(dma_rd_f.tvalid),
                      .s_axis_tready(dma_rd_f.tready),
                      .s_axis_tdata(dma_rd_f.tdata),
                      .s_axis_tkeep(dma_rd_f.tkeep),
                      .s_axis_tlast(dma_rd_f.tlast),

                      .m_axis_tvalid(dma_rd_dwc.tvalid),
                      .m_axis_tready(dma_rd_dwc.tready),
                      .m_axis_tdata(dma_rd_dwc.tdata),
                      .m_axis_tkeep(dma_rd_dwc.tkeep),
                      .m_axis_tlast(dma_rd_dwc.tlast));


// REG
axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(OLEN_BITS))
                     inst_reg_wr (.aclk(aclk),
                                  .aresetn(aresetn),

                                  .s_axis_tvalid(s_axis.tvalid),
                                  .s_axis_tready(s_axis.tready),
                                  .s_axis_tdata(s_axis.tdata),

                                  .m_axis_tvalid(dma_wr_dwc.tvalid),
                                  .m_axis_tready(dma_wr_dwc.tready),
                                  .m_axis_tdata(dma_wr_dwc.tdata)
                                  );

axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(ILEN_BITS))
                     inst_reg_rd (.aclk(aclk),
                                  .aresetn(aresetn),

                                  .s_axis_tvalid(dma_rd_dwc.tvalid),
                                  .s_axis_tready(dma_rd_dwc.tready),
                                  .s_axis_tdata(dma_rd_dwc.tdata),

                                  .m_axis_tvalid(m_axis.tvalid),
                                  .m_axis_tready(m_axis.tready),
                                  .m_axis_tdata(m_axis.tdata)
                                  );

//
// DBG
//

//if(DBG == 1) begin
//    ila_if inst_ila_if (
//        .clk(aclk),
//        .probe0(s_idx.tvalid),
//        .probe1(s_idx.tready),
//        .probe2(q_s0_dma_out.tvalid),
//        .probe3(q_s0_dma_out.tready),
//        .probe4(q_s0_buf_out.tvalid),
//        .probe5(q_s0_buf_out.tready),
//        .probe6(state_wr_C),
//        .probe7(cnt_frames_wr_C), // 16
//        .probe8(n_frames_wr_C), // 16
//        .probe9(len_wr_C), // 32
//        .probe10(addr_wr_C), // 64
//        .probe11(cnt_outstanding_C), // 16
//        .probe12(cnt_outstanding_decr),
//        .probe13(cnt_outstanding_incr),
//        .probe14(q_wr_dma_out.tvalid),
//        .probe15(q_wr_dma_out.tready),
//        .probe16(q_wr_fr_done_out.tvalid),
//        .probe17(q_wr_fr_done_out.tready),
//        .probe18(wr_done),
//        .probe19(q_s1_dma_out.tvalid),
//        .probe20(q_s1_dma_out.tready),
//        .probe21(m_idx.tvalid),
//        .probe22(m_idx.tready),
//        .probe23(state_rd_C),
//        .probe24(cnt_frames_rd_C), // 16
//        .probe25(n_frames_rd_C), // 16
//        .probe26(len_rd_C), // 32
//        .probe27(addr_rd_C), // 64
//        .probe28(rd_done),
//        .probe29(q_rd_dma_out.tvalid),
//        .probe30(q_rd_dma_out.tready),
//        .probe31(s_axis.tvalid),
//        .probe32(s_axis.tready),
//        .probe33(m_axis.tvalid),
//        .probe34(m_axis.tready),
//        .probe35(m_idx.tdata[0+:CNT_BITS]), // 16
//        .probe36(dma_wr_dwc.tready),
//        .probe37(dma_wr_dwc.tvalid),
//        .probe38(dma_wr.tvalid),
//        .probe39(dma_wr.tready),
//        .probe40(q_rd_dma_out.tdata[0+:64]), // 64
//        .probe41(q_wr_dma_out.tdata[0+:64]), // 64
//        .probe42(q_rd_dma_out.tdata[64+:32]), // 32
//        .probe43(q_wr_dma_out.tdata[64+:32]), // 32
//        .probe44(wr_done),
//        .probe45(rd_done),
//        .probe46(m_axi_hbm.arvalid),
//        .probe47(m_axi_hbm.arready),
//        .probe48(m_axi_hbm.awvalid),
//        .probe49(m_axi_hbm.awready),
//        .probe50(m_axi_hbm.araddr), // 64
//        .probe51(m_axi_hbm.awaddr), // 64
//        .probe52(m_axi_hbm.rvalid),
//        .probe53(m_axi_hbm.rready),
//        .probe54(m_axi_hbm.wvalid),
//        .probe55(m_axi_hbm.wready)
//    );
//end

endmodule

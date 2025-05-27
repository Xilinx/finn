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

module fetch_start #(
    parameter int unsigned              ADDR_BITS = 64,
    parameter int unsigned              DATA_BITS = 256,
    parameter int unsigned              LEN_BITS = 32,
    parameter int unsigned              CNT_BITS = 16,

    parameter int unsigned              ILEN_BITS = 32,
    parameter logic[ADDR_BITS-1:0]      ADDR_SRC,

    parameter int unsigned              QDEPTH = 8,
    parameter int unsigned              N_DCPL_STGS = 1
) (
    input  wire                         aclk,
    input  wire                         aresetn,

    AXI4.master                         m_axi_hbm,

    AXI4S.slave                         s_ctrl,
    AXI4S.master                        m_idx,

    AXI4S.master                        m_axis
);

AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+CNT_BITS+LEN_BITS)) q_ctrl_out ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma ();
AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+LEN_BITS)) q_dma_out ();
AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) q_idx ();

// Queues
queue #(.QDEPTH(QDEPTH), .QWIDTH(ADDR_BITS+CNT_BITS+LEN_BITS)) inst_queue_ctrl (.aclk(aclk), .aresetn(aresetn), .s_axis(s_ctrl), .m_axis(q_ctrl_out));
queue #(.QDEPTH(QDEPTH), .QWIDTH(ADDR_BITS+LEN_BITS)) inst_queue_dma (.aclk(aclk), .aresetn(aresetn), .s_axis(q_dma), .m_axis(q_dma_out));
queue #(.QDEPTH(QDEPTH), .QWIDTH(2*CNT_BITS+LEN_BITS)) inst_queue_idx (.aclk(aclk), .aresetn(aresetn), .s_axis(q_idx), .m_axis(m_idx));

// Regs
typedef enum logic[0:0] {ST_IDLE, ST_READ} state_t;
state_t state_C = ST_IDLE, state_N;

logic [ADDR_BITS-1:0] addr_C = '0, addr_N;
logic [LEN_BITS-1:0] len_C = '0, len_N;
logic [CNT_BITS-1:0] n_frames_C = '0, n_frames_N;
logic [CNT_BITS-1:0] cnt_frames_C = '0, cnt_frames_N;

logic rd_done;

always_ff @( posedge aclk ) begin
    if(~aresetn) begin
        state_C <= ST_IDLE;

        addr_C <= 'X;
        len_C <= 'X;
        n_frames_C <= 'X;
        cnt_frames_C <= 0;
    end
    else begin
        state_C <= state_N;

        addr_C <= addr_N;
        len_C <= len_N;
        n_frames_C <= n_frames_N;
        cnt_frames_C <= cnt_frames_N;
    end
end

always_comb begin
    state_N = state_C;

    case (state_C)
        ST_IDLE:
            state_N = q_ctrl_out.tvalid && q_idx.tready ? ST_READ : ST_IDLE;

        ST_READ:
            state_N = ((cnt_frames_C == n_frames_C-1) && q_dma.tready) ? ST_IDLE : ST_READ;

    endcase
end

always_comb begin
    // AL
    n_frames_N = n_frames_C;
    cnt_frames_N = cnt_frames_C;
    addr_N = addr_C;
    len_N = len_C;

    // S
    q_ctrl_out.tready = 1'b0;
    q_idx.tvalid = 1'b0;
    q_idx.tdata = {q_ctrl_out.tdata[ADDR_BITS+CNT_BITS+:LEN_BITS], q_ctrl_out.tdata[ADDR_BITS+:CNT_BITS], {CNT_BITS{1'b0}}};
    q_dma.tvalid = 1'b0;
    q_dma.tdata = {len_C, addr_C};

    // RD
    case (state_C)

        ST_IDLE: begin
            q_ctrl_out.tready = q_idx.tready;
            q_idx.tvalid = q_ctrl_out.tvalid;

            if(q_ctrl_out.tvalid && q_idx.tready) begin
                addr_N = ADDR_SRC | q_ctrl_out.tdata[0+:ADDR_BITS];
                n_frames_N = q_ctrl_out.tdata[ADDR_BITS+:CNT_BITS];
                len_N = q_ctrl_out.tdata[ADDR_BITS+CNT_BITS+:LEN_BITS];
                cnt_frames_N = 0;
            end
        end

        ST_READ: begin
            q_dma.tvalid = 1'b1;
            if(q_dma.tready) begin
                addr_N = addr_C + len_C;
                cnt_frames_N = cnt_frames_C + 1;
            end
        end

    endcase
end

// DMA
AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) tmp_rd_f ();
AXI4SF #(.AXI4S_DATA_BITS(DATA_BITS), .AXI4S_USER_BITS(1)) dma_rd_f ();
AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) dma_rd ();
AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) dma_rd_dwc ();

cdma_top #(
    .ADDR_BITS(ADDR_BITS),
    .LEN_BITS(LEN_BITS),
    .DATA_BITS(DATA_BITS),
    .CDMA_RD(1),
    .CDMA_WR(0)
) inst_dma_rd (
    .aclk(aclk),
    .aresetn(aresetn),

    .m_axi_ddr(m_axi_hbm),

    .rd_valid(q_dma_out.tvalid),
    .rd_ready(q_dma_out.tready),
    .rd_paddr(q_dma_out.tdata[0+:ADDR_BITS]),
    .rd_len(q_dma_out.tdata[ADDR_BITS+:LEN_BITS]),
    .rd_done(rd_done),

    .wr_valid(1'b0),
    .wr_ready(),
    .wr_paddr('0),
    .wr_len('0),
    .wr_done(),

    .s_axis_ddr(tmp_rd_f),
    .m_axis_ddr(dma_rd_f)
);
`AXISF_AXIS_ASSIGN(dma_rd_f, dma_rd)
`AXISF_TIE_OFF_M(tmp_rd_f)

// DWC
axis_dwc #(.S_DATA_BITS(DATA_BITS), .M_DATA_BITS(ILEN_BITS)) inst_dwc_rd (.aclk(aclk), .aresetn(aresetn), .s_axis(dma_rd), .m_axis(dma_rd_dwc));

// REG
axis_reg_array_tmplt #(.N_STAGES(N_DCPL_STGS), .DATA_BITS(ILEN_BITS)) inst_reg_rd (.aclk(aclk), .aresetn(aresetn), .s_axis(dma_rd_dwc), .m_axis(m_axis));

endmodule

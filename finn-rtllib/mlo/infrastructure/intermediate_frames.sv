/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

module intermediate_frames #(
    int unsigned                    ELEM_BITS,
    int unsigned                    ILEN_BITS,
    int unsigned                    OLEN_BITS,

    int unsigned                    ADDR_BITS,
    int unsigned                    DATA_BITS,
    int unsigned                    LEN_BITS,
    int unsigned                    IDX_BITS,

    int unsigned                    FM_SIZE,

    int unsigned                    N_OUTSTANDING_DMAS = 128,

    int unsigned                    QDEPTH = 8,
    int unsigned                    N_DCPL_STGS = 1,
    int unsigned                    DBG = 0,

    bit [ADDR_BITS-1:0]             ADDRESS_OFFSET = 0
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
    input  logic                        m_axis_tready,

    // Base Address
    input  logic [ADDR_BITS-1:0]        base_address
);

// Offsets
logic [N_OUTSTANDING_DMAS-1:0][ADDR_BITS-1:0] l_offsets;
for(genvar i = 0; i < N_OUTSTANDING_DMAS; i++) begin
    assign l_offsets[i] = (i * FM_SIZE);
end
localparam int unsigned  N_OUTSTANDING_DMAS_BITS = $clog2(N_OUTSTANDING_DMAS);

localparam int unsigned  EBYTES       = (ELEM_BITS + 7)/8;
localparam int unsigned  OELEM        = OLEN_BITS / ELEM_BITS;
localparam int unsigned  IELEM        = ILEN_BITS / ELEM_BITS;
localparam int unsigned  OLEN_BITS_BA = OELEM * EBYTES * 8;
localparam int unsigned  ILEN_BITS_BA = IELEM * EBYTES * 8;

localparam int unsigned  FM_BEATS_IN  = FM_SIZE/(OLEN_BITS_BA/8);

initial begin
    if(ELEM_BITS == 0) begin
        $error("%m: ELEM_BITS must be non-zero.");
        $finish;
    end
    if(FM_SIZE == 0) begin
        $error("%m: FM_SIZE must be non-zero.");
        $finish;
    end
    if(OLEN_BITS % ELEM_BITS != 0) begin
        $error("%m: OLEN_BITS (%0d) not a multiple of ELEM_BITS (%0d).",
            OLEN_BITS, ELEM_BITS);
        $finish;
    end
    if(ILEN_BITS % ELEM_BITS != 0) begin
        $error("%m: ILEN_BITS (%0d) not a multiple of ELEM_BITS (%0d).",
            ILEN_BITS, ELEM_BITS);
        $finish;
    end
    if(FM_SIZE % (OLEN_BITS_BA/8) != 0) begin
        $error("%m: FM_SIZE (%0d) not a multiple of write-side byte width (%0d).",
            FM_SIZE, OLEN_BITS_BA/8);
        $finish;
    end
    if(FM_SIZE % (ILEN_BITS_BA/8) != 0) begin
        $error("%m: FM_SIZE (%0d) not a multiple of read-side byte width (%0d).",
            FM_SIZE, ILEN_BITS_BA/8);
        $finish;
    end
    if(FM_SIZE % (DATA_BITS/8) != 0) begin
        $error("%m: FM_SIZE (%0d) not a multiple of DMA bus width (%0d).",
            FM_SIZE, DATA_BITS/8);
        $finish;
    end
    if(DATA_BITS < OLEN_BITS_BA) begin
        $error("%m: DATA_BITS (%0d) must be >= OLEN_BITS_BA (%0d).",
            DATA_BITS, OLEN_BITS_BA);
        $finish;
    end
    if(DATA_BITS < ILEN_BITS_BA) begin
        $error("%m: DATA_BITS (%0d) must be >= ILEN_BITS_BA (%0d).",
            DATA_BITS, ILEN_BITS_BA);
        $finish;
    end
end

//
// Write side
//

// Input queue
logic idx_in_tvalid, idx_in_tready;
logic [IDX_BITS-1:0] idx_in_tdata;

fifo #(
    .DEPTH(QDEPTH), .DATA_WIDTH(IDX_BITS)) inst_queue_seq (
    .clk(aclk), .rst(!aresetn),
    .count(), .maxcount(),
    .idat(s_idx_tdata), .ivld(s_idx_tvalid), .irdy(s_idx_tready),
    .odat(idx_in_tdata), .ovld(idx_in_tvalid), .ordy(idx_in_tready)
);

// Outstanding DMA frame credit
logic  wr_sent;
uwire  rd_done;
uwire  wr_rdy;
if(1) begin : blkCredit
    logic signed [$clog2(N_OUTSTANDING_DMAS):0]  DmaCredit = -N_OUTSTANDING_DMAS;  // -N_OUTSTANDING_DMAS, .., -1, 0 (exhausted)
    always_ff @(posedge aclk) begin
        if(!aresetn)  DmaCredit <= -N_OUTSTANDING_DMAS;
        else          DmaCredit <= DmaCredit + (wr_sent == rd_done? 0 : wr_sent? 1 : -1);
    end
    assign  wr_rdy = DmaCredit[$left(DmaCredit)];
end : blkCredit

// FSM
typedef enum logic[0:0] {ST_WR_IDLE, ST_WR_SEND} state_wr_t;
state_wr_t state_wr_C = ST_WR_IDLE, state_wr_N;

logic [N_OUTSTANDING_DMAS_BITS-1:0] wr_ptr_C = '0, wr_ptr_N;

logic s0_dma_in_tvalid, s0_dma_in_tready;
logic [ADDR_BITS-1:0] s0_dma_in_tdata;
logic s0_dma_out_tvalid, s0_dma_out_tready;
logic [ADDR_BITS-1:0] s0_dma_out_tdata;

fifo #(
    .DEPTH(QDEPTH), .DATA_WIDTH(ADDR_BITS)) inst_queue_s0_dma (
    .clk(aclk), .rst(!aresetn),
    .count(), .maxcount(),
    .idat(s0_dma_in_tdata), .ivld(s0_dma_in_tvalid), .irdy(s0_dma_in_tready),
    .odat(s0_dma_out_tdata), .ovld(s0_dma_out_tvalid), .ordy(s0_dma_out_tready)
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
    s0_dma_in_tdata = base_address + ADDRESS_OFFSET + l_offsets[wr_ptr_C];
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

uwire  done_wr_in;
uwire  done_wr_out;
logic  rd_start;
if(1) begin : blkCompletion
    logic signed [$clog2(N_OUTSTANDING_DMAS):0]  WritesDone = 0;  // 0 (none), -1, .., -N_OUTSTANDING_DMAS
    always_ff @(posedge aclk) begin
        if(!aresetn)  WritesDone <= 0;
        else          WritesDone <= WritesDone + (done_wr_in == rd_start? 0 : done_wr_in? -1 : 1);
    end
    assign  done_wr_out = WritesDone[$left(WritesDone)];
end : blkCompletion

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

fifo #(
    .DEPTH(QDEPTH), .DATA_WIDTH(ADDR_BITS)) inst_queue_s1_dma (
    .clk(aclk), .rst(!aresetn),
    .count(), .maxcount(),
    .idat(s1_dma_in_tdata), .ivld(s1_dma_in_tvalid), .irdy(s1_dma_in_tready),
    .odat(s1_dma_out_tdata), .ovld(s1_dma_out_tvalid), .ordy(s1_dma_out_tready)
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
    s1_dma_in_tdata = base_address + ADDRESS_OFFSET + l_offsets[rd_ptr_C];

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

logic axis_dma_wr_tvalid, axis_dma_wr_tready;
logic [DATA_BITS-1:0] axis_dma_wr_tdata;

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
    .m_axis_ddr_tkeep (),
    .m_axis_ddr_tlast (),

    .s_axis_ddr_tvalid(axis_dma_wr_tvalid),
    .s_axis_ddr_tready(axis_dma_wr_tready),
    .s_axis_ddr_tdata (axis_dma_wr_tdata),
    .s_axis_ddr_tkeep ('1),
    .s_axis_ddr_tlast ('0)
);

// DWC
logic s_axis_int_tvalid, s_axis_int_tready;
logic [OLEN_BITS-1:0] s_axis_int_tdata;

logic m_axis_int_tvalid, m_axis_int_tready;
logic [ILEN_BITS-1:0] m_axis_int_tdata;

logic s_axis_ba_tvalid, s_axis_ba_tready;
logic [OLEN_BITS_BA-1:0] s_axis_ba_tdata;
logic m_axis_ba_tvalid, m_axis_ba_tready;
logic [ILEN_BITS_BA-1:0] m_axis_ba_tdata;

assign s_axis_ba_tvalid  = s_axis_int_tvalid;
assign s_axis_int_tready = s_axis_ba_tready;
for(genvar e = 0; e < OELEM; e++) begin : gen_wr_byte_align
    assign s_axis_ba_tdata[e*EBYTES*8 +: EBYTES*8] =
        { {(EBYTES*8-ELEM_BITS){1'b0}}, s_axis_int_tdata[e*ELEM_BITS +: ELEM_BITS] };
end

assign m_axis_int_tvalid = m_axis_ba_tvalid;
assign m_axis_ba_tready  = m_axis_int_tready;
for(genvar e = 0; e < IELEM; e++) begin : gen_rd_byte_align
    assign m_axis_int_tdata[e*ELEM_BITS +: ELEM_BITS] =
        m_axis_ba_tdata[e*EBYTES*8 +: ELEM_BITS];
end

// VPC write: OLEN_BITS_BA -> DATA_BITS (byte-aligned body output -> DMA)
vpc #(.W(OLEN_BITS_BA), .N(FM_BEATS_IN), .PI(1), .PO(DATA_BITS/OLEN_BITS_BA)) inst_dwc_wr (
    .clk(aclk), .rst(!aresetn),
    .ivld(s_axis_ba_tvalid), .irdy(s_axis_ba_tready),
    .idat(s_axis_ba_tdata),
    .ovld(axis_dma_wr_tvalid), .ordy(axis_dma_wr_tready),
    .odat(axis_dma_wr_tdata)
);

// VPC read: DATA_BITS -> ILEN_BITS_BA (DMA -> byte-aligned body input)
vpc #(.W(ILEN_BITS_BA), .N(DATA_BITS/ILEN_BITS_BA), .PI(DATA_BITS/ILEN_BITS_BA), .PO(1)) inst_dwc_rd (
    .clk(aclk), .rst(!aresetn),
    .ivld(axis_dma_rd_tvalid), .irdy(axis_dma_rd_tready),
    .idat(axis_dma_rd_tdata),
    .ovld(m_axis_ba_tvalid), .ordy(m_axis_ba_tready),
    .odat(m_axis_ba_tdata)
);

// REG
skid #(.FEED_STAGES(N_DCPL_STGS), .DATA_WIDTH(OLEN_BITS)) inst_reg_wr (
    .clk(aclk),
    .rst(~aresetn),

    .ivld(s_axis_tvalid),
    .irdy(s_axis_tready),
    .idat(s_axis_tdata),
    .ovld(s_axis_int_tvalid),
    .ordy(s_axis_int_tready),
    .odat(s_axis_int_tdata)
);

skid #(.FEED_STAGES(N_DCPL_STGS), .DATA_WIDTH(ILEN_BITS)) inst_reg_rd (
    .clk(aclk),
    .rst(~aresetn),

    .ivld(m_axis_int_tvalid),
    .irdy(m_axis_int_tready),
    .idat(m_axis_int_tdata),
    .ovld(m_axis_tvalid),
    .ordy(m_axis_tready),
    .odat(m_axis_tdata)
);

endmodule

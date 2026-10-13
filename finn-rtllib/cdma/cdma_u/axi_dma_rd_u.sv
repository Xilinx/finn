/*
Copyright (c) 2018 Alex Forencich
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @brief   Unaligned CDMA AXI read engine
 *
 * The unaligned CDMA read engine, AXI to stream. Supports outstanding transactions (N_OUTSTANDING).
 * High resource overhead. Used in one channel mode.
 */
module axi_dma_rd_u #
(
    // Width of AXI data bus in bits
    parameter AXI_DATA_WIDTH = 32,
    // Width of AXI address bus in bits
    parameter AXI_ADDR_WIDTH = 16,
    // Width of AXI wstrb (width of data bus in words)
    parameter AXI_STRB_WIDTH = (AXI_DATA_WIDTH/8),
    // Maximum AXI burst length to generate
    parameter AXI_MAX_BURST_LEN = 16,
    // Width of AXI stream interfaces in bits
    parameter AXIS_DATA_WIDTH = AXI_DATA_WIDTH,
    // Use AXI stream tkeep signal
    parameter AXIS_KEEP_ENABLE = (AXIS_DATA_WIDTH>8),
    // AXI stream tkeep signal width (words per cycle)
    parameter AXIS_KEEP_WIDTH = (AXIS_DATA_WIDTH/8),
    // Use AXI stream tlast signal
    parameter AXIS_LAST_ENABLE = 1,
    // Width of length field
    parameter LEN_WIDTH = 20,
    // ID bits
    parameter AXI_ID_BITS = 1,
    // Credit-based read pacing: gate AR bursts and buffer read data in a FIFO
    // so that m_axi_rready never deasserts due to downstream backpressure.
    // Set PACED=0 when the caller guarantees backpressure avoidance externally.
    bit           PACED = 1,
    int unsigned  N_OUTSTANDING = 0
)
(
    input  logic                       aclk,
    input  logic                       aresetn,

    /*
     * AXI read descriptor input
     */
    input  logic [AXI_ADDR_WIDTH-1:0]  s_axis_read_desc_addr,
    input  logic [LEN_WIDTH-1:0]       s_axis_read_desc_len,
    input  logic                       s_axis_read_desc_valid,
    output logic                       s_axis_read_desc_ready,

    /*
     * AXI read descriptor status output
     */
    output logic                       m_axis_read_desc_status_valid,

    /*
     * AXI stream read data output
     */
    output logic [AXIS_DATA_WIDTH-1:0] m_axis_read_data_tdata,
    output logic [AXIS_KEEP_WIDTH-1:0] m_axis_read_data_tkeep,
    output logic                       m_axis_read_data_tvalid,
    input  logic                       m_axis_read_data_tready,
    output logic                       m_axis_read_data_tlast,

    /*
     * AXI master interface
     */
    output logic [AXI_ID_BITS-1:0]     m_axi_arid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,
    output logic [2:0]                 m_axi_arsize,
    output logic [1:0]                 m_axi_arburst,
    output logic                       m_axi_arlock,
    output logic [3:0]                 m_axi_arcache,
    output logic [2:0]                 m_axi_arprot,
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,
    input  logic [AXI_ID_BITS-1:0]     m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready
);

initial begin
    if (N_OUTSTANDING < 2) begin
        $error("%m: N_OUTSTANDING (%0d) must be >= 2.", N_OUTSTANDING);
        $finish;
    end
end

localparam AXI_WORD_WIDTH = AXI_STRB_WIDTH;
localparam AXI_WORD_SIZE = AXI_DATA_WIDTH/AXI_WORD_WIDTH;
localparam AXI_BURST_SIZE = $clog2(AXI_STRB_WIDTH);
localparam AXI_MAX_BURST_SIZE = AXI_MAX_BURST_LEN << AXI_BURST_SIZE;

localparam AXIS_KEEP_WIDTH_INT = AXIS_KEEP_ENABLE ? AXIS_KEEP_WIDTH : 1;
localparam AXIS_WORD_WIDTH = AXIS_KEEP_WIDTH_INT;
localparam AXIS_WORD_SIZE = AXIS_DATA_WIDTH/AXIS_WORD_WIDTH;

localparam OFFSET_WIDTH = AXI_STRB_WIDTH > 1 ? $clog2(AXI_STRB_WIDTH) : 1;
localparam OFFSET_MASK = AXI_STRB_WIDTH > 1 ? {OFFSET_WIDTH{1'b1}} : 0;
localparam ADDR_MASK = {AXI_ADDR_WIDTH{1'b1}} << $clog2(AXI_STRB_WIDTH);
localparam CYCLE_COUNT_WIDTH = LEN_WIDTH - AXI_BURST_SIZE + 1;

localparam [0:0]
    AXI_STATE_IDLE = 1'd0,
    AXI_STATE_START = 1'd1;

logic [0:0] axi_state_reg = AXI_STATE_IDLE, axi_state_next;

localparam [0:0]
    AXIS_STATE_IDLE = 1'd0,
    AXIS_STATE_READ = 1'd1;

logic [0:0] axis_state_reg = AXIS_STATE_IDLE, axis_state_next;

// datapath control signals
logic transfer_in_save;

logic [AXI_ADDR_WIDTH-1:0] addr_reg = {AXI_ADDR_WIDTH{1'b0}}, addr_next;
logic [LEN_WIDTH-1:0] op_word_count_reg = {LEN_WIDTH{1'b0}}, op_word_count_next;
logic [LEN_WIDTH-1:0] tr_word_count_reg = {LEN_WIDTH{1'b0}}, tr_word_count_next;

typedef struct packed {
    logic [OFFSET_WIDTH-1:0] axis_cmd_offset;
    logic [OFFSET_WIDTH-1:0] axis_cmd_last_cycle_offset;
    logic axis_cmd_bubble_cycle;
    logic [CYCLE_COUNT_WIDTH-1:0] axis_cmd_input_cycle_count;
    logic [CYCLE_COUNT_WIDTH-1:0] axis_cmd_output_cycle_count;
} cdma_rd_cmd_t;

logic ost_snk_valid, ost_snk_ready;
cdma_rd_cmd_t ost_snk_data;
logic ost_src_valid, ost_src_ready;
cdma_rd_cmd_t ost_src_data;

logic [OFFSET_WIDTH-1:0] offset_reg = {OFFSET_WIDTH{1'b0}}, offset_next;
logic [OFFSET_WIDTH-1:0] last_cycle_offset_reg = {OFFSET_WIDTH{1'b0}}, last_cycle_offset_next;
logic [CYCLE_COUNT_WIDTH-1:0] input_cycle_count_reg = {CYCLE_COUNT_WIDTH{1'b0}}, input_cycle_count_next;
logic [CYCLE_COUNT_WIDTH-1:0] output_cycle_count_reg = {CYCLE_COUNT_WIDTH{1'b0}}, output_cycle_count_next;
logic input_active_reg = 1'b0, input_active_next;
logic output_active_reg = 1'b0, output_active_next;
logic bubble_cycle_reg = 1'b0, bubble_cycle_next;
logic first_cycle_reg = 1'b0, first_cycle_next;
logic output_last_cycle_reg = 1'b0, output_last_cycle_next;

logic m_axis_read_desc_status_valid_reg = 1'b0, m_axis_read_desc_status_valid_next;

logic [AXI_ADDR_WIDTH-1:0] m_axi_araddr_reg = {AXI_ADDR_WIDTH{1'b0}}, m_axi_araddr_next;
logic [7:0] m_axi_arlen_reg = 8'd0, m_axi_arlen_next;
logic m_axi_arvalid_reg = 1'b0, m_axi_arvalid_next;
logic m_axi_rready_reg = 1'b0, m_axi_rready_next;

// AXI read data as consumed by the realignment (through the read-data FIFO if PACED)
uwire [AXI_DATA_WIDTH-1:0] r_data;
uwire r_valid;
uwire r_ready = m_axi_rready_reg;

logic [AXI_DATA_WIDTH-1:0] save_axi_rdata_reg = {AXI_DATA_WIDTH{1'b0}};
uwire [AXI_DATA_WIDTH-1:0] shift_axi_rdata = {r_data, save_axi_rdata_reg} >> ((AXI_STRB_WIDTH-offset_reg)*AXI_WORD_SIZE);

// internal datapath
logic  [AXIS_DATA_WIDTH-1:0] m_axis_read_data_tdata_int;
logic  [AXIS_KEEP_WIDTH-1:0] m_axis_read_data_tkeep_int;
logic                        m_axis_read_data_tvalid_int;
logic                        m_axis_read_data_tready_int_reg = 1'b0;
logic                        m_axis_read_data_tlast_int;
logic                        m_axis_read_data_tready_int_early;

// output stage registers
logic [AXIS_DATA_WIDTH-1:0] m_axis_read_data_tdata_reg  = {AXIS_DATA_WIDTH{1'b0}};
logic [AXIS_KEEP_WIDTH-1:0] m_axis_read_data_tkeep_reg  = {AXIS_KEEP_WIDTH{1'b0}};
logic                       m_axis_read_data_tvalid_reg = 1'b0, m_axis_read_data_tvalid_next;
logic                       m_axis_read_data_tlast_reg  = 1'b0;

logic [AXIS_DATA_WIDTH-1:0] temp_m_axis_read_data_tdata_reg  = {AXIS_DATA_WIDTH{1'b0}};
logic [AXIS_KEEP_WIDTH-1:0] temp_m_axis_read_data_tkeep_reg  = {AXIS_KEEP_WIDTH{1'b0}};
logic                       temp_m_axis_read_data_tvalid_reg = 1'b0, temp_m_axis_read_data_tvalid_next;
logic                       temp_m_axis_read_data_tlast_reg  = 1'b0;

assign m_axis_read_desc_status_valid = m_axis_read_desc_status_valid_reg;

assign m_axi_arid = 0;
assign m_axi_araddr = m_axi_araddr_reg;
assign m_axi_arlen = m_axi_arlen_reg;
assign m_axi_arsize = AXI_BURST_SIZE;
assign m_axi_arburst = 2'b01;
assign m_axi_arlock = 1'b0;
assign m_axi_arcache = 4'b0011;
assign m_axi_arprot = 3'b010;
assign m_axi_arvalid = m_axi_arvalid_reg;

// === Credit-based AR pacing (PACED generate) =============================
// Module-scope bridge signal — assigned inside the generate, referenced
// by the AR FSM outside it.
uwire  credit_ok;
if(PACED) begin : genPaced
    localparam int unsigned  RDATA_FIFO_DEPTH = N_OUTSTANDING * AXI_MAX_BURST_LEN;

    // FIFO-backed credit: take the beats of each burst, return individual beats
    //   no credit for full burst | can issue another burst
    // -AXI_MAX_BURST_LEN, ..., -1, 0, ..., RDATA_FIFO_DEPTH-AXI_MAX_BURST_LEN
    localparam int signed  CREDIT_INIT = RDATA_FIFO_DEPTH - AXI_MAX_BURST_LEN;
    logic signed [$clog2(CREDIT_INIT+1):0]  Credit = CREDIT_INIT;

    uwire  ar_accept  = m_axi_arvalid & m_axi_arready;
    uwire  credit_in = r_valid & r_ready;
    always_ff @(posedge aclk) begin
        if (~aresetn)  Credit <= CREDIT_INIT;
        else           Credit <= Credit - (ar_accept? m_axi_arlen + 1 : 0) + credit_in;
    end
    assign  credit_ok = !Credit[$left(Credit)];

    // Read-data FIFO: absorbs AXI R beats while the realignment and downstream stall.
    // Every AR burst has its capacity pre-reserved via the credit counter.
    fifo #(
        .DEPTH(RDATA_FIFO_DEPTH),
        .DATA_WIDTH(AXI_DATA_WIDTH)
    ) inst_rdata_fifo (
        .clk(aclk), .rst(!aresetn),
        .count(), .maxcount(),
        .idat(m_axi_rdata), .ivld(m_axi_rvalid), .irdy(m_axi_rready),
        .odat(r_data), .ovld(r_valid), .ordy(r_ready)
    );

    // Check that rready is never backpressured under PACED mode
    always_ff @(posedge aclk) begin
        assert(!aresetn || !m_axi_rvalid || m_axi_rready) else begin
            $error("%m: rready backpressure under PACED — credit invariant violated.");
            $stop;
        end
    end

end : genPaced
else begin : genBypass

    assign  credit_ok = 1;
    assign  m_axi_rready = r_ready;
    assign  r_data  = m_axi_rdata;
    assign  r_valid = m_axi_rvalid;

end

assign  m_axis_read_data_tdata  = m_axis_read_data_tdata_reg;
assign  m_axis_read_data_tkeep  = AXIS_KEEP_ENABLE ? m_axis_read_data_tkeep_reg : {AXIS_KEEP_WIDTH{1'b1}};
assign  m_axis_read_data_tvalid = m_axis_read_data_tvalid_reg;
assign  m_axis_read_data_tlast  = AXIS_LAST_ENABLE ? m_axis_read_data_tlast_reg : 1'b1;

uwire [AXI_ADDR_WIDTH-1:0] addr_plus_max_burst = addr_reg + AXI_MAX_BURST_SIZE;
uwire [AXI_ADDR_WIDTH-1:0] addr_plus_count = addr_reg + op_word_count_reg;

// Outstanding descriptor queue
fifo #(
    .DEPTH(N_OUTSTANDING),
    .DATA_WIDTH($bits(cdma_rd_cmd_t))
) inst_q_rd (
    .clk(aclk),
    .rst(!aresetn),
    .count(),
    .maxcount(),
    .idat(ost_snk_data),
    .ivld(ost_snk_valid),
    .irdy(ost_snk_ready),
    .odat(ost_src_data),
    .ovld(ost_src_valid),
    .ordy(ost_src_ready)
);

// NSL Requests
always_comb begin
    axi_state_next = AXI_STATE_IDLE;

    m_axi_araddr_next = m_axi_araddr_reg;
    m_axi_arlen_next = m_axi_arlen_reg;
    m_axi_arvalid_next = m_axi_arvalid_reg && !m_axi_arready;

    addr_next = addr_reg;
    op_word_count_next = op_word_count_reg;
    tr_word_count_next = tr_word_count_reg;

    s_axis_read_desc_ready = 1'b0;

    ost_snk_valid = 1'b0;
    ost_snk_data = 0;

    case (axi_state_reg)
        AXI_STATE_IDLE: begin
            // idle state - load new descriptor to start operation
            s_axis_read_desc_ready = ost_snk_ready;

            if (s_axis_read_desc_valid & s_axis_read_desc_ready) begin
                ost_snk_valid = 1'b1;

                addr_next = s_axis_read_desc_addr;
                op_word_count_next = s_axis_read_desc_len;

                ost_snk_data.axis_cmd_offset = AXI_STRB_WIDTH > 1 ? AXI_STRB_WIDTH - (s_axis_read_desc_addr & OFFSET_MASK) : 0;
                ost_snk_data.axis_cmd_last_cycle_offset = s_axis_read_desc_len & OFFSET_MASK;
                ost_snk_data.axis_cmd_bubble_cycle = ost_snk_data.axis_cmd_offset > 0;
                ost_snk_data.axis_cmd_input_cycle_count = (op_word_count_next + (s_axis_read_desc_addr & OFFSET_MASK) - 1) >> AXI_BURST_SIZE;
                ost_snk_data.axis_cmd_output_cycle_count = (op_word_count_next - 1) >> AXI_BURST_SIZE;

                axi_state_next = AXI_STATE_START;
            end else begin
                axi_state_next = AXI_STATE_IDLE;
            end
        end

        AXI_STATE_START: begin
            // start state - initiate new AXI transfer
            // Per-burst reservation: only launch when credit_ok (PACED=1) or
            // unconditionally (PACED=0, credit_ok is constant 1'b1).
            if (!m_axi_arvalid && credit_ok) begin
                if (op_word_count_reg <= AXI_MAX_BURST_SIZE - (addr_reg & OFFSET_MASK) || AXI_MAX_BURST_SIZE >= 4096) begin
                    // packet smaller than max burst size
                    if (addr_reg[12] != addr_plus_count[12]) begin
                        // crosses 4k boundary
                        tr_word_count_next = 13'h1000 - addr_reg[11:0];
                    end else begin
                        // does not cross 4k boundary
                        tr_word_count_next = op_word_count_reg;
                    end
                end else begin
                    // packet larger than max burst size
                    if (addr_reg[12] != addr_plus_max_burst[12]) begin
                        // crosses 4k boundary
                        tr_word_count_next = 13'h1000 - addr_reg[11:0];
                    end else begin
                        // does not cross 4k boundary
                        tr_word_count_next = AXI_MAX_BURST_SIZE - (addr_reg & OFFSET_MASK);
                    end
                end

                m_axi_araddr_next = addr_reg;
                m_axi_arlen_next = (tr_word_count_next + (addr_reg & OFFSET_MASK) - 1) >> AXI_BURST_SIZE;
                m_axi_arvalid_next = 1'b1;

                addr_next = addr_reg + tr_word_count_next;
                op_word_count_next = op_word_count_reg - tr_word_count_next;

                if (op_word_count_next > 0) begin
                    axi_state_next = AXI_STATE_START;
                end else begin
                    axi_state_next = AXI_STATE_IDLE;
                end
            end else begin
                axi_state_next = AXI_STATE_START;
            end
        end
    endcase
end

// NSL Data
always_comb begin
    axis_state_next = AXIS_STATE_IDLE;

    m_axis_read_desc_status_valid_next = 1'b0;

    m_axis_read_data_tdata_int = shift_axi_rdata;
    m_axis_read_data_tkeep_int = {AXIS_KEEP_WIDTH{1'b1}};
    m_axis_read_data_tlast_int = 1'b0;
    m_axis_read_data_tvalid_int = 1'b0;

    m_axi_rready_next = 1'b0;

    transfer_in_save = 1'b0;

    offset_next = offset_reg;
    last_cycle_offset_next = last_cycle_offset_reg;
    input_cycle_count_next = input_cycle_count_reg;
    output_cycle_count_next = output_cycle_count_reg;
    bubble_cycle_next = bubble_cycle_reg;

    output_last_cycle_next = output_last_cycle_reg;
    input_active_next = input_active_reg;
    output_active_next = output_active_reg;
    first_cycle_next = first_cycle_reg;

    ost_src_ready = 1'b0;

    case (axis_state_reg)
        AXIS_STATE_IDLE: begin
            // idle state - load new descriptor to start operation
            m_axi_rready_next = 1'b0;

            // store transfer parameters
            offset_next = ost_src_data.axis_cmd_offset;
            last_cycle_offset_next = ost_src_data.axis_cmd_last_cycle_offset;
            input_cycle_count_next = ost_src_data.axis_cmd_input_cycle_count;
            output_cycle_count_next = ost_src_data.axis_cmd_output_cycle_count;
            bubble_cycle_next = ost_src_data.axis_cmd_bubble_cycle;

            output_last_cycle_next = output_cycle_count_next == 0;
            input_active_next = 1'b1;
            output_active_next = 1'b1;
            first_cycle_next = 1'b1;

            if (ost_src_valid) begin
                ost_src_ready = 1'b1;

                m_axi_rready_next = m_axis_read_data_tready_int_early;
                axis_state_next = AXIS_STATE_READ;
            end
        end
        AXIS_STATE_READ: begin
            // handle AXI read data
            m_axi_rready_next = m_axis_read_data_tready_int_early && input_active_reg;

            if (m_axis_read_data_tready_int_reg && ((r_ready && r_valid) || !input_active_reg)) begin
                // transfer in AXI read data
                transfer_in_save = r_ready && r_valid;

                if (first_cycle_reg && bubble_cycle_reg) begin
                    if (input_active_reg) begin
                        input_cycle_count_next = input_cycle_count_reg - 1;
                        input_active_next = input_cycle_count_reg > 0;
                    end
                    bubble_cycle_next = 1'b0;
                    first_cycle_next = 1'b0;

                    m_axi_rready_next = m_axis_read_data_tready_int_early && input_active_next;
                    axis_state_next = AXIS_STATE_READ;
                end else begin
                    // update counters
                    if (input_active_reg) begin
                        input_cycle_count_next = input_cycle_count_reg - 1;
                        input_active_next = input_cycle_count_reg > 0;
                    end
                    if (output_active_reg) begin
                        output_cycle_count_next = output_cycle_count_reg - 1;
                        output_active_next = output_cycle_count_reg > 0;
                    end
                    output_last_cycle_next = output_cycle_count_next == 0;
                    bubble_cycle_next = 1'b0;
                    first_cycle_next = 1'b0;

                    // pass through read data
                    m_axis_read_data_tdata_int = shift_axi_rdata;
                    m_axis_read_data_tkeep_int = {AXIS_KEEP_WIDTH_INT{1'b1}};
                    m_axis_read_data_tvalid_int = 1'b1;

                    if (output_last_cycle_reg) begin
                        // no more data to transfer, finish operation
                        if (last_cycle_offset_reg > 0) begin
                            m_axis_read_data_tkeep_int = {AXIS_KEEP_WIDTH_INT{1'b1}} >> (AXIS_KEEP_WIDTH_INT - last_cycle_offset_reg);
                        end
                        m_axis_read_data_tlast_int = 1'b1;

                        m_axis_read_desc_status_valid_next = 1'b1;

                        m_axi_rready_next = 1'b0;
                        axis_state_next = AXIS_STATE_IDLE;
                    end else begin
                        // more cycles in AXI transfer
                        m_axi_rready_next = m_axis_read_data_tready_int_early && input_active_next;
                        axis_state_next = AXIS_STATE_READ;
                    end
                end
            end else begin
                axis_state_next = AXIS_STATE_READ;
            end
        end
    endcase
end

always_ff @(posedge aclk) begin
    if (~aresetn) begin
        axi_state_reg <= AXI_STATE_IDLE;
        axis_state_reg <= AXIS_STATE_IDLE;
        m_axis_read_desc_status_valid_reg <= 1'b0;
        m_axi_arvalid_reg <= 1'b0;
        m_axi_rready_reg <= 1'b0;
    end else begin
        axi_state_reg <= axi_state_next;
        axis_state_reg <= axis_state_next;
        m_axis_read_desc_status_valid_reg <= m_axis_read_desc_status_valid_next;
        m_axi_arvalid_reg <= m_axi_arvalid_next;
        m_axi_rready_reg <= m_axi_rready_next;
    end

    m_axi_araddr_reg <= m_axi_araddr_next;
    m_axi_arlen_reg <= m_axi_arlen_next;

    addr_reg <= addr_next;
    op_word_count_reg <= op_word_count_next;
    tr_word_count_reg <= tr_word_count_next;

    offset_reg <= offset_next;
    last_cycle_offset_reg <= last_cycle_offset_next;
    input_cycle_count_reg <= input_cycle_count_next;
    output_cycle_count_reg <= output_cycle_count_next;
    input_active_reg <= input_active_next;
    output_active_reg <= output_active_next;
    bubble_cycle_reg <= bubble_cycle_next;
    first_cycle_reg <= first_cycle_next;
    output_last_cycle_reg <= output_last_cycle_next;

    if (transfer_in_save) begin
        save_axi_rdata_reg <= r_data;
    end
end

// output datapath logic

// datapath control
logic store_axis_int_to_output;
logic store_axis_int_to_temp;
logic store_axis_temp_to_output;

// enable ready input next cycle if output is ready or the temp reg will not be filled on the next cycle (output reg empty or no input)
assign m_axis_read_data_tready_int_early = m_axis_read_data_tready || (!temp_m_axis_read_data_tvalid_reg && (!m_axis_read_data_tvalid_reg || !m_axis_read_data_tvalid_int));

always_comb begin
    // transfer sink ready state to source
    m_axis_read_data_tvalid_next = m_axis_read_data_tvalid_reg;
    temp_m_axis_read_data_tvalid_next = temp_m_axis_read_data_tvalid_reg;

    store_axis_int_to_output = 1'b0;
    store_axis_int_to_temp = 1'b0;
    store_axis_temp_to_output = 1'b0;

    if (m_axis_read_data_tready_int_reg) begin
        // input is ready
        if (m_axis_read_data_tready || !m_axis_read_data_tvalid_reg) begin
            // output is ready or currently not valid, transfer data to output
            m_axis_read_data_tvalid_next = m_axis_read_data_tvalid_int;
            store_axis_int_to_output = 1'b1;
        end else begin
            // output is not ready, store input in temp
            temp_m_axis_read_data_tvalid_next = m_axis_read_data_tvalid_int;
            store_axis_int_to_temp = 1'b1;
        end
    end else if (m_axis_read_data_tready) begin
        // input is not ready, but output is ready
        m_axis_read_data_tvalid_next = temp_m_axis_read_data_tvalid_reg;
        temp_m_axis_read_data_tvalid_next = 1'b0;
        store_axis_temp_to_output = 1'b1;
    end
end

always_ff @(posedge aclk) begin
    if (~aresetn) begin
        m_axis_read_data_tvalid_reg <= 1'b0;
        m_axis_read_data_tready_int_reg <= 1'b0;
        temp_m_axis_read_data_tvalid_reg <= 1'b0;
    end else begin
        m_axis_read_data_tvalid_reg <= m_axis_read_data_tvalid_next;
        m_axis_read_data_tready_int_reg <= m_axis_read_data_tready_int_early;
        temp_m_axis_read_data_tvalid_reg <= temp_m_axis_read_data_tvalid_next;
    end

    // datapath
    if (store_axis_int_to_output) begin
        m_axis_read_data_tdata_reg <= m_axis_read_data_tdata_int;
        m_axis_read_data_tkeep_reg <= m_axis_read_data_tkeep_int;
        m_axis_read_data_tlast_reg <= m_axis_read_data_tlast_int;
    end else if (store_axis_temp_to_output) begin
        m_axis_read_data_tdata_reg <= temp_m_axis_read_data_tdata_reg;
        m_axis_read_data_tkeep_reg <= temp_m_axis_read_data_tkeep_reg;
        m_axis_read_data_tlast_reg <= temp_m_axis_read_data_tlast_reg;
    end

    if (store_axis_int_to_temp) begin
        temp_m_axis_read_data_tdata_reg <= m_axis_read_data_tdata_int;
        temp_m_axis_read_data_tkeep_reg <= m_axis_read_data_tkeep_int;
        temp_m_axis_read_data_tlast_reg <= m_axis_read_data_tlast_int;
    end
end

endmodule

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
 * @brief   Unaligned CDMA AXI write engine
 *
 * The unaligned CDMA write engine, AXI to stream. Supports outstanding transactions (N_OUTSTANDING).
 * High resource overhead. Used in one channel mode.
 */
module axi_dma_wr_u #
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
    parameter AXIS_LAST_ENABLE = 0,
    // Width of length field
    parameter LEN_WIDTH = 20,
    // ID bits
    parameter AXI_ID_BITS = 1
)
(
    input  logic                       aclk,
    input  logic                       aresetn,

    /*
     * AXI write descriptor input
     */
    input  logic [AXI_ADDR_WIDTH-1:0]  s_axis_write_desc_addr,
    input  logic [LEN_WIDTH-1:0]       s_axis_write_desc_len,
    input  logic                       s_axis_write_desc_valid,
    output logic                       s_axis_write_desc_ready,

    /*
     * AXI write descriptor status output
     */
    output logic                       m_axis_write_desc_status_valid,

    /*
     * AXI stream write data input
     */
    input  logic [AXIS_DATA_WIDTH-1:0] s_axis_write_data_tdata,
    input  logic [AXIS_KEEP_WIDTH-1:0] s_axis_write_data_tkeep,
    input  logic                       s_axis_write_data_tvalid,
    output logic                       s_axis_write_data_tready,
    input  logic                       s_axis_write_data_tlast,

    /*
     * AXI master interface
     */
    output logic [AXI_ID_BITS-1:0]     m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,
    output logic [1:0]                 m_axi_awburst,
    output logic                       m_axi_awlock,
    output logic [3:0]                 m_axi_awcache,
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_STRB_WIDTH-1:0]  m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,
    input  logic [AXI_ID_BITS-1:0]     m_axi_bid,
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready
);

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

localparam STATUS_FIFO_ADDR_WIDTH = 5;

localparam [1:0]
    STATE_IDLE = 3'd0,
    STATE_START = 3'd1,
    STATE_WRITE = 3'd2;

logic[1:0] state_reg = STATE_IDLE, state_next;

// datapath control signals
logic transfer_in_save;
logic flush_save;
logic status_fifo_we;

integer i;
logic [OFFSET_WIDTH:0] cycle_size;

logic [AXI_ADDR_WIDTH-1:0] addr_reg = {AXI_ADDR_WIDTH{1'b0}}, addr_next;
logic [LEN_WIDTH-1:0] op_word_count_reg = {LEN_WIDTH{1'b0}}, op_word_count_next;
logic [LEN_WIDTH-1:0] tr_word_count_reg = {LEN_WIDTH{1'b0}}, tr_word_count_next;

logic [OFFSET_WIDTH-1:0] offset_reg = {OFFSET_WIDTH{1'b0}}, offset_next;
logic [AXI_STRB_WIDTH-1:0] strb_offset_mask_reg = {AXI_STRB_WIDTH{1'b1}}, strb_offset_mask_next;
logic zero_offset_reg = 1'b1, zero_offset_next;
logic [OFFSET_WIDTH-1:0] last_cycle_offset_reg = {OFFSET_WIDTH{1'b0}}, last_cycle_offset_next;
logic [LEN_WIDTH-1:0] length_reg = {LEN_WIDTH{1'b0}}, length_next;
logic [CYCLE_COUNT_WIDTH-1:0] input_cycle_count_reg = {CYCLE_COUNT_WIDTH{1'b0}}, input_cycle_count_next;
logic [CYCLE_COUNT_WIDTH-1:0] output_cycle_count_reg = {CYCLE_COUNT_WIDTH{1'b0}}, output_cycle_count_next;
logic input_active_reg = 1'b0, input_active_next;
logic first_cycle_reg = 1'b0, first_cycle_next;
logic input_last_cycle_reg = 1'b0, input_last_cycle_next;
logic output_last_cycle_reg = 1'b0, output_last_cycle_next;
logic last_transfer_reg = 1'b0, last_transfer_next;

logic [STATUS_FIFO_ADDR_WIDTH+1-1:0] status_fifo_wr_ptr_reg = 0, status_fifo_wr_ptr_next;
logic [STATUS_FIFO_ADDR_WIDTH+1-1:0] status_fifo_rd_ptr_reg = 0, status_fifo_rd_ptr_next;
logic status_fifo_last[(2**STATUS_FIFO_ADDR_WIDTH)-1:0];
logic status_fifo_wr_last;

logic s_axis_write_desc_ready_reg = 1'b0, s_axis_write_desc_ready_next;

logic m_axis_write_desc_status_valid_reg = 1'b0, m_axis_write_desc_status_valid_next;

logic [AXI_ADDR_WIDTH-1:0] m_axi_awaddr_reg = {AXI_ADDR_WIDTH{1'b0}}, m_axi_awaddr_next;
logic [7:0] m_axi_awlen_reg = 8'd0, m_axi_awlen_next;
logic m_axi_awvalid_reg = 1'b0, m_axi_awvalid_next;
logic m_axi_bready_reg = 1'b0, m_axi_bready_next;

logic s_axis_write_data_tready_reg = 1'b0, s_axis_write_data_tready_next;

logic [AXIS_DATA_WIDTH-1:0] save_axis_tdata_reg = {AXIS_DATA_WIDTH{1'b0}};
logic [AXIS_KEEP_WIDTH_INT-1:0] save_axis_tkeep_reg = {AXIS_KEEP_WIDTH_INT{1'b0}};
logic save_axis_tlast_reg = 1'b0;

logic [AXIS_DATA_WIDTH-1:0] shift_axis_tdata;
logic [AXIS_KEEP_WIDTH_INT-1:0] shift_axis_tkeep;
logic shift_axis_tvalid;
logic shift_axis_tlast;
logic shift_axis_input_tready;
logic shift_axis_extra_cycle_reg = 1'b0;

// internal datapath
logic  [AXI_DATA_WIDTH-1:0] m_axi_wdata_int;
logic  [AXI_STRB_WIDTH-1:0] m_axi_wstrb_int;
logic                       m_axi_wlast_int;
logic                       m_axi_wvalid_int;
logic                       m_axi_wready_int_reg = 1'b0;
logic                      m_axi_wready_int_early;

logic [14:0] tmp_loc_reg, tmp_loc_next;

assign s_axis_write_desc_ready = s_axis_write_desc_ready_reg;

assign m_axis_write_desc_status_valid = m_axis_write_desc_status_valid_reg;

assign s_axis_write_data_tready = s_axis_write_data_tready_reg;

assign m_axi_awid = 0;
assign m_axi_awaddr = m_axi_awaddr_reg;
assign m_axi_awlen = m_axi_awlen_reg;
assign m_axi_awsize = AXI_BURST_SIZE;
assign m_axi_awburst = 2'b01;
assign m_axi_awlock = 1'b0;
assign m_axi_awcache = 4'b0011;
assign m_axi_awvalid = m_axi_awvalid_reg;
assign m_axi_bready = m_axi_bready_reg;

logic [AXI_ADDR_WIDTH-1:0] addr_plus_max_burst = addr_reg + AXI_MAX_BURST_SIZE;
logic [AXI_ADDR_WIDTH-1:0] addr_plus_count = addr_reg + op_word_count_reg;

always_comb begin
    shift_axis_tdata = {s_axis_write_data_tdata, save_axis_tdata_reg} >> ((AXIS_KEEP_WIDTH_INT-offset_reg)*AXIS_WORD_SIZE);
    shift_axis_tkeep = {s_axis_write_data_tkeep, save_axis_tkeep_reg} >> (AXIS_KEEP_WIDTH_INT-offset_reg);
    shift_axis_tvalid = s_axis_write_data_tvalid;
    shift_axis_tlast = 1'b0;
    shift_axis_input_tready = 1'b1;
end

always_comb begin
    state_next = STATE_IDLE;

    s_axis_write_desc_ready_next = 1'b0;

    m_axis_write_desc_status_valid_next = 1'b0;

    s_axis_write_data_tready_next = 1'b0;

    m_axi_awaddr_next = m_axi_awaddr_reg;
    m_axi_awlen_next = m_axi_awlen_reg;
    m_axi_awvalid_next = m_axi_awvalid_reg && !m_axi_awready;
    m_axi_wdata_int = shift_axis_tdata;
    m_axi_wstrb_int = shift_axis_tkeep;
    m_axi_wlast_int = 1'b0;
    m_axi_wvalid_int = 1'b0;
    m_axi_bready_next = 1'b0;

    transfer_in_save = 1'b0;
    flush_save = 1'b0;
    status_fifo_we = 1'b0;

    cycle_size = AXIS_KEEP_WIDTH_INT;

    addr_next = addr_reg;
    offset_next = offset_reg;
    strb_offset_mask_next = strb_offset_mask_reg;
    zero_offset_next = zero_offset_reg;
    last_cycle_offset_next = last_cycle_offset_reg;
    length_next = length_reg;
    op_word_count_next = op_word_count_reg;
    tr_word_count_next = tr_word_count_reg;
    input_cycle_count_next = input_cycle_count_reg;
    output_cycle_count_next = output_cycle_count_reg;
    input_active_next = input_active_reg;
    first_cycle_next = first_cycle_reg;
    input_last_cycle_next = input_last_cycle_reg;
    output_last_cycle_next = output_last_cycle_reg;
    last_transfer_next = last_transfer_reg;

    status_fifo_rd_ptr_next = status_fifo_rd_ptr_reg;

    status_fifo_wr_last = 1'b0;

    tmp_loc_next = 0;

    case (state_reg)
        STATE_IDLE: begin
            // idle state - load new descriptor to start operation
            flush_save = 1'b1;
            s_axis_write_desc_ready_next = 1'b1;

            addr_next = s_axis_write_desc_addr;
            offset_next = s_axis_write_desc_addr & OFFSET_MASK;
            strb_offset_mask_next = {AXI_STRB_WIDTH{1'b1}} << (s_axis_write_desc_addr & OFFSET_MASK);
            zero_offset_next = (s_axis_write_desc_addr & OFFSET_MASK) == 0;
            last_cycle_offset_next = offset_next + (s_axis_write_desc_len & OFFSET_MASK);

            op_word_count_next = s_axis_write_desc_len;
            first_cycle_next = 1'b1;
            length_next = 0;

            if (s_axis_write_desc_ready && s_axis_write_desc_valid) begin
                s_axis_write_desc_ready_next = 1'b0;
                state_next = STATE_START;
            end else begin
                state_next = STATE_IDLE;
            end
        end
        STATE_START: begin
            // start state - initiate new AXI transfer
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

            input_cycle_count_next = (tr_word_count_next - 1) >> $clog2(AXIS_KEEP_WIDTH_INT);
            input_last_cycle_next = input_cycle_count_next == 0;

            output_cycle_count_next = (tr_word_count_next + (addr_reg & OFFSET_MASK) - 1) >> AXI_BURST_SIZE;
            output_last_cycle_next = output_cycle_count_next == 0;

            last_transfer_next = tr_word_count_next == op_word_count_reg;
            input_active_next = 1'b1;

            if (!first_cycle_reg && last_transfer_next) begin
                if (offset_reg >= last_cycle_offset_reg && last_cycle_offset_reg > 0) begin
                    // last cycle will be served by stored partial cycle
                    input_active_next = input_cycle_count_next > 0;
                    input_cycle_count_next = input_cycle_count_next - 1;
                end
            end

            if (!m_axi_awvalid_reg) begin
                m_axi_awaddr_next = addr_reg;
                m_axi_awlen_next = output_cycle_count_next;
                m_axi_awvalid_next = s_axis_write_data_tvalid || !first_cycle_reg;

                if (m_axi_awvalid_next) begin
                    addr_next = addr_reg + tr_word_count_next;
                    op_word_count_next = op_word_count_reg - tr_word_count_next;

                    s_axis_write_data_tready_next = m_axi_wready_int_early && input_active_next;
                    state_next = STATE_WRITE;
                end else begin
                    state_next = STATE_START;
                end
            end else begin
                state_next = STATE_START;
            end
        end

        STATE_WRITE: begin
            //s_axis_write_data_tready_next = m_axi_wready_int_early && (last_transfer_reg || input_active_reg) && shift_axis_input_tready;
            s_axis_write_data_tready_next = m_axi_wready_int_early && (input_active_reg) && shift_axis_input_tready;

            tmp_loc_next[0] = 1'b1;

            //if (m_axi_wready_int_reg && ((s_axis_write_data_tready && shift_axis_tvalid) || (!input_active_reg && !last_transfer_reg) || !shift_axis_input_tready)) begin
            if (m_axi_wready_int_reg && ((s_axis_write_data_tready && shift_axis_tvalid) || (!input_active_reg) || !shift_axis_input_tready)) begin
                tmp_loc_next[1] = 1'b1;

                if (s_axis_write_data_tready && s_axis_write_data_tvalid) begin
                    transfer_in_save = 1'b1;

                    tmp_loc_next[2] = 1'b1;
                end

                // update counters
                if (first_cycle_reg) begin
                    length_next = length_reg + (AXIS_KEEP_WIDTH_INT - offset_reg);

                    tmp_loc_next[3] = 1'b1;
                end else begin
                    length_next = length_reg + AXIS_KEEP_WIDTH_INT;

                    tmp_loc_next[4] = 1'b1;
                end
                if (input_active_reg) begin
                    input_cycle_count_next = input_cycle_count_reg - 1;
                    input_active_next = input_cycle_count_reg > 0;

                    tmp_loc_next[5] = 1'b1;
                end
                input_last_cycle_next = input_cycle_count_next == 0;
                output_cycle_count_next = output_cycle_count_reg - 1;
                output_last_cycle_next = output_cycle_count_next == 0;
                first_cycle_next = 1'b0;
                strb_offset_mask_next = {AXI_STRB_WIDTH{1'b1}};

                m_axi_wdata_int = shift_axis_tdata;
                m_axi_wstrb_int = strb_offset_mask_reg;
                m_axi_wvalid_int = 1'b1;

                if (output_last_cycle_reg) begin
                    m_axi_wlast_int = 1'b1;

                    tmp_loc_next[6] = 1'b1;

                    if (op_word_count_reg > 0) begin
                        // current AXI transfer complete, but there is more data to transfer
                        // enqueue status FIFO entry for write completion
                        status_fifo_we = 1'b1;
                        status_fifo_wr_last = 1'b0;

                        tmp_loc_next[7] = 1'b1;

                        s_axis_write_data_tready_next = 1'b0;
                        state_next = STATE_START;
                    end else begin
                        // no more data to transfer, finish operation
                        if (last_cycle_offset_reg > 0) begin
                            m_axi_wstrb_int = strb_offset_mask_reg & {AXI_STRB_WIDTH{1'b1}} >> (AXI_STRB_WIDTH - last_cycle_offset_reg);

                            tmp_loc_next[8] = 1'b1;

                            if (first_cycle_reg) begin
                                length_next = length_reg + (last_cycle_offset_reg - offset_reg);

                                tmp_loc_next[9] = 1'b1;
                            end else begin
                                length_next = length_reg + last_cycle_offset_reg;

                                tmp_loc_next[10] = 1'b1;
                            end
                        end

                        tmp_loc_next[11] = 1'b1;

                        // enqueue status FIFO entry for write completion
                        status_fifo_we = 1'b1;
                        status_fifo_wr_last = 1'b1;

                        tmp_loc_next[12] = 1'b1;

                        // no framing; return to idle
                        s_axis_write_data_tready_next = 1'b0;
                        s_axis_write_desc_ready_next = 1'b1;
                        state_next = STATE_IDLE;
                    end
                end else begin
                    tmp_loc_next[13] = 1'b1;

                    s_axis_write_data_tready_next = m_axi_wready_int_early && input_active_next && shift_axis_input_tready;
                    state_next = STATE_WRITE;
                end
            end else begin
                tmp_loc_next[14] = 1'b1;

                state_next = STATE_WRITE;
            end
        end
    endcase

    if (status_fifo_rd_ptr_reg != status_fifo_wr_ptr_reg) begin
        // status FIFO not empty
        if (m_axi_bready && m_axi_bvalid) begin
            // got write completion, pop and return status
            m_axis_write_desc_status_valid_next = status_fifo_last[status_fifo_rd_ptr_reg[STATUS_FIFO_ADDR_WIDTH-1:0]];
            status_fifo_rd_ptr_next = status_fifo_rd_ptr_reg + 1;
            m_axi_bready_next = 1'b0;
        end else begin
            // wait for write completion
            m_axi_bready_next = 1'b1;
        end
    end
end

always_ff @(posedge aclk) begin
    if (~aresetn) begin
        state_reg <= STATE_IDLE;
        s_axis_write_desc_ready_reg <= 1'b0;
        m_axis_write_desc_status_valid_reg <= 1'b0;
        s_axis_write_data_tready_reg <= 1'b0;
        m_axi_awvalid_reg <= 1'b0;
        m_axi_bready_reg <= 1'b0;
        save_axis_tlast_reg <= 1'b0;
        shift_axis_extra_cycle_reg <= 1'b0;

        status_fifo_wr_ptr_reg <= 0;
        status_fifo_rd_ptr_reg <= 0;

        tmp_loc_reg <= 0;
    end else begin
        state_reg <= state_next;
        s_axis_write_desc_ready_reg <= s_axis_write_desc_ready_next;
        m_axis_write_desc_status_valid_reg <= m_axis_write_desc_status_valid_next;
        s_axis_write_data_tready_reg <= s_axis_write_data_tready_next;
        m_axi_awvalid_reg <= m_axi_awvalid_next;
        m_axi_bready_reg <= m_axi_bready_next;

        tmp_loc_reg <= tmp_loc_next;

        // datapath
        if (flush_save) begin
            save_axis_tlast_reg <= 1'b0;
            shift_axis_extra_cycle_reg <= 1'b0;
        end else if (transfer_in_save) begin
            save_axis_tlast_reg <= s_axis_write_data_tlast;
            shift_axis_extra_cycle_reg <= s_axis_write_data_tlast & ((s_axis_write_data_tkeep >> (AXIS_KEEP_WIDTH_INT-offset_reg)) != 0);
        end

        if (status_fifo_we) begin
            status_fifo_wr_ptr_reg <= status_fifo_wr_ptr_reg + 1;
        end
        status_fifo_rd_ptr_reg <= status_fifo_rd_ptr_next;
    end

    m_axi_awaddr_reg <= m_axi_awaddr_next;
    m_axi_awlen_reg <= m_axi_awlen_next;

    addr_reg <= addr_next;
    offset_reg <= offset_next;
    strb_offset_mask_reg <= strb_offset_mask_next;
    zero_offset_reg <= zero_offset_next;
    last_cycle_offset_reg <= last_cycle_offset_next;
    length_reg <= length_next;
    op_word_count_reg <= op_word_count_next;
    tr_word_count_reg <= tr_word_count_next;
    input_cycle_count_reg <= input_cycle_count_next;
    output_cycle_count_reg <= output_cycle_count_next;
    input_active_reg <= input_active_next;
    first_cycle_reg <= first_cycle_next;
    input_last_cycle_reg <= input_last_cycle_next;
    output_last_cycle_reg <= output_last_cycle_next;
    last_transfer_reg <= last_transfer_next;

    if (flush_save) begin
        save_axis_tkeep_reg <= {AXIS_KEEP_WIDTH_INT{1'b0}};
    end else if (transfer_in_save) begin
        save_axis_tdata_reg <= s_axis_write_data_tdata;
        save_axis_tkeep_reg <= AXIS_KEEP_ENABLE ? s_axis_write_data_tkeep : {AXIS_KEEP_WIDTH_INT{1'b1}};
    end

    if (status_fifo_we) begin
        status_fifo_last[status_fifo_wr_ptr_reg[STATUS_FIFO_ADDR_WIDTH-1:0]] <= status_fifo_wr_last;
        status_fifo_wr_ptr_reg <= status_fifo_wr_ptr_reg + 1;
    end
end

// output datapath logic
logic [AXI_DATA_WIDTH-1:0] m_axi_wdata_reg  = {AXI_DATA_WIDTH{1'b0}};
logic [AXI_STRB_WIDTH-1:0] m_axi_wstrb_reg  = {AXI_STRB_WIDTH{1'b0}};
logic                      m_axi_wlast_reg  = 1'b0;
logic                      m_axi_wvalid_reg = 1'b0, m_axi_wvalid_next;

logic [AXI_DATA_WIDTH-1:0] temp_m_axi_wdata_reg  = {AXI_DATA_WIDTH{1'b0}};
logic [AXI_STRB_WIDTH-1:0] temp_m_axi_wstrb_reg  = {AXI_STRB_WIDTH{1'b0}};
logic                      temp_m_axi_wlast_reg  = 1'b0;
logic                      temp_m_axi_wvalid_reg = 1'b0, temp_m_axi_wvalid_next;

// datapath control
logic store_axi_w_int_to_output;
logic store_axi_w_int_to_temp;
logic store_axi_w_temp_to_output;

assign m_axi_wdata  = m_axi_wdata_reg;
assign m_axi_wstrb  = m_axi_wstrb_reg;
assign m_axi_wvalid = m_axi_wvalid_reg;
assign m_axi_wlast  = m_axi_wlast_reg;

// enable ready input next cycle if output is ready or the temp reg will not be filled on the next cycle (output reg empty or no input)
assign m_axi_wready_int_early = m_axi_wready || (!temp_m_axi_wvalid_reg && (!m_axi_wvalid_reg || !m_axi_wvalid_int));

always_comb begin
    // transfer sink ready state to source
    m_axi_wvalid_next = m_axi_wvalid_reg;
    temp_m_axi_wvalid_next = temp_m_axi_wvalid_reg;

    store_axi_w_int_to_output = 1'b0;
    store_axi_w_int_to_temp = 1'b0;
    store_axi_w_temp_to_output = 1'b0;

    if (m_axi_wready_int_reg) begin
        // input is ready
        if (m_axi_wready || !m_axi_wvalid_reg) begin
            // output is ready or currently not valid, transfer data to output
            m_axi_wvalid_next = m_axi_wvalid_int;
            store_axi_w_int_to_output = 1'b1;
        end else begin
            // output is not ready, store input in temp
            temp_m_axi_wvalid_next = m_axi_wvalid_int;
            store_axi_w_int_to_temp = 1'b1;
        end
    end else if (m_axi_wready) begin
        // input is not ready, but output is ready
        m_axi_wvalid_next = temp_m_axi_wvalid_reg;
        temp_m_axi_wvalid_next = 1'b0;
        store_axi_w_temp_to_output = 1'b1;
    end
end

always_ff @(posedge aclk) begin
    if (~aresetn) begin
        m_axi_wvalid_reg <= 1'b0;
        m_axi_wready_int_reg <= 1'b0;
        temp_m_axi_wvalid_reg <= 1'b0;
    end else begin
        m_axi_wvalid_reg <= m_axi_wvalid_next;
        m_axi_wready_int_reg <= m_axi_wready_int_early;
        temp_m_axi_wvalid_reg <= temp_m_axi_wvalid_next;
    end

    // datapath
    if (store_axi_w_int_to_output) begin
        m_axi_wdata_reg <= m_axi_wdata_int;
        m_axi_wstrb_reg <= m_axi_wstrb_int;
        m_axi_wlast_reg <= m_axi_wlast_int;
    end else if (store_axi_w_temp_to_output) begin
        m_axi_wdata_reg <= temp_m_axi_wdata_reg;
        m_axi_wstrb_reg <= temp_m_axi_wstrb_reg;
        m_axi_wlast_reg <= temp_m_axi_wlast_reg;
    end

    if (store_axi_w_int_to_temp) begin
        temp_m_axi_wdata_reg <= m_axi_wdata_int;
        temp_m_axi_wstrb_reg <= m_axi_wstrb_int;
        temp_m_axi_wlast_reg <= m_axi_wlast_int;
    end
end

/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_CDMA_WR_U

`endif

endmodule

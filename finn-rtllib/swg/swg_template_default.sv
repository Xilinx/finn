module $TOP_MODULE_NAME$_controller #(
    int unsigned  LOOP_H_ITERATIONS    = $LOOP_H_ITERATIONS$,
    int unsigned  LOOP_W_ITERATIONS    = $LOOP_W_ITERATIONS$,
    int unsigned  LOOP_KH_ITERATIONS   = $LOOP_KH_ITERATIONS$,
    int unsigned  LOOP_KW_ITERATIONS   = $LOOP_KW_ITERATIONS$,
    int unsigned  LOOP_SIMD_ITERATIONS = $LOOP_SIMD_ITERATIONS$,

    int unsigned  INCR_BITWIDTH = $INCR_BITWIDTH$,
    bit [INCR_BITWIDTH-1:0]  ADDR_INCREMENT_MAP[6] = $ADDR_INCREMENT_MAP$
)(
    input   logic  clk,
    input   logic  rst_n,

    input   logic  advance,
    output  logic [INCR_BITWIDTH-1:0]  addr_incr,
    output  logic [INCR_BITWIDTH-1:0]  tail_incr
);

    //State and counters
    typedef enum logic [2:0] {
        STATE_START,
        STATE_LOOP_SIMD,
        STATE_LOOP_KW,
        STATE_LOOP_KH,
        STATE_LOOP_W,
        STATE_LOOP_H
    }  state_e;
    state_e  State = $INNERMOST_STATE$;
    state_e  state_next;

    logic signed [$clog2(LOOP_H_ITERATIONS   +2)+1-1:0]  counter_loop_h    = LOOP_H_ITERATIONS-1;
    logic signed [$clog2(LOOP_W_ITERATIONS   +2)+1-1:0]  counter_loop_w    = LOOP_W_ITERATIONS-1;
    logic signed [$clog2(LOOP_KH_ITERATIONS  +2)+1-1:0]  counter_loop_kh   = LOOP_KH_ITERATIONS-1;
    logic signed [$clog2(LOOP_KW_ITERATIONS  +2)+1-1:0]  counter_loop_kw   = LOOP_KW_ITERATIONS-1;
    logic signed [$clog2(LOOP_SIMD_ITERATIONS+2)+1-1:0]  counter_loop_simd = LOOP_SIMD_ITERATIONS-1;

    logic [INCR_BITWIDTH-1:0]  tail_incr_reg = 'x;
    assign  addr_incr = ADDR_INCREMENT_MAP[State];
    assign  tail_incr = tail_incr_reg;

    //combinational logic for tail_incr generation
    $TAIL_INCR_GENERATION$

    //combinational next state logic
    always_comb begin : blkState
        state_next = State;
        if(State != $INNERMOST_STATE$)  state_next = $INNERMOST_STATE$;
        else begin
            if(counter_loop_simd < 0) begin
                state_next =
                    (counter_loop_kw >= 0)? STATE_LOOP_KW :
                    (counter_loop_kh >= 0)? STATE_LOOP_KH :
                    (counter_loop_w  >= 0)? STATE_LOOP_W :
                    (counter_loop_h  >= 0)? STATE_LOOP_H :
                    /* else */              STATE_START;
            end
        end
    end : blkState

    //sequential logic
    always_ff @ (posedge clk) begin
        if(!rst_n) begin
            State <= $INNERMOST_STATE$;
            counter_loop_h    <= LOOP_H_ITERATIONS-1;
            counter_loop_w    <= LOOP_W_ITERATIONS-1;
            counter_loop_kh   <= LOOP_KH_ITERATIONS-1;
            counter_loop_kw   <= LOOP_KW_ITERATIONS-1;
            counter_loop_simd <= LOOP_SIMD_ITERATIONS-1;
        end
        else if(advance) begin
            State <= state_next;
            if (State == $INNERMOST_STATE$) begin
                if(counter_loop_simd >= 0)  counter_loop_simd <= counter_loop_simd-1;
                else begin
                    counter_loop_simd <= LOOP_SIMD_ITERATIONS-1;
                    if(counter_loop_kw >= 0)  counter_loop_kw <= counter_loop_kw-1;
                    else begin
                        counter_loop_kw <= LOOP_KW_ITERATIONS-1;
                        if(counter_loop_kh >= 0)  counter_loop_kh <= counter_loop_kh-1;
                        else begin
                            counter_loop_kh <= LOOP_KH_ITERATIONS-1;
                            if(counter_loop_w >= 0)  counter_loop_w <= counter_loop_w-1;
                            else begin
                                counter_loop_w <= LOOP_W_ITERATIONS-1;
                                if(counter_loop_h >= 0)  counter_loop_h <= counter_loop_h-1;
                                else  counter_loop_h <= LOOP_H_ITERATIONS-1;
                            end 
                        end 
                    end
                end 
            end
        end
    end

endmodule :  $TOP_MODULE_NAME$_controller

module $TOP_MODULE_NAME$_cyclic_buffer_addressable #(
    int unsigned  WIDTH,
    int unsigned  DEPTH
)(
    input   logic  clk,
    input   logic  rst_n,

    input   logic  write_enable,
    input   logic [$clog2(DEPTH)-1:0] write_addr,
    input   logic [WIDTH-1:0]  data_in,

    input   logic  read_enable,
    input   logic [$clog2(DEPTH)-1:0]  read_addr, // absolute (!) read address of cyclic buffer
    output  logic [WIDTH-1:0]  data_out
);

    $RAM_STYLE$ logic [WIDTH-1:0] Ram[DEPTH];
    logic [WIDTH-1:0]  Out = 'x;
    always_ff @(posedge clk) begin 
        if (!rst_n) begin
            Out       <= 'x;
        end
        else begin
            if (read_enable)  Out <= Ram[read_addr];
            if (write_enable) Ram[write_addr] <= data_in;
        end
    end
    assign  data_out = Out;

endmodule : $TOP_MODULE_NAME$_cyclic_buffer_addressable

module $TOP_MODULE_NAME$_impl #(
    int  BIT_WIDTH = $BIT_WIDTH$,
    int  SIMD = $SIMD$,
    int  MMV_IN = $MMV_IN$,
    int  MMV_OUT = $MMV_OUT$,
    int  LAST_READ_ELEM = $LAST_READ_ELEM$,
    int  LAST_WRITE_ELEM = $LAST_WRITE_ELEM$,
    int  BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$,
    int  ELEM_PER_WINDOW = $ELEM_PER_WINDOW$,
    int  INCR_BITWIDTH = $INCR_BITWIDTH$
)(
    input   logic  ap_clk,
    input   logic  ap_rst_n,

    input   logic  in0_V_V_TVALID,
    output  logic  in0_V_V_TREADY,
    input   logic [BIT_WIDTH * SIMD * MMV_IN-1:0]  in0_V_V_TDATA,

    output  logic  out_V_V_TVALID,
    input   logic  out_V_V_TREADY,
    output  logic [BIT_WIDTH * SIMD * MMV_OUT-1:0]  out_V_V_TDATA
);
    // Derived Constants
    localparam int unsigned  BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
    localparam int unsigned  BUF_OUT_ELEM_WIDTH = BIT_WIDTH * SIMD;
    localparam int unsigned  BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;

   //main buffer instantiation
    uwire [BUF_IN_WIDTH -1:0]  window_buffer_in;
    uwire [BUF_OUT_WIDTH-1:0]  window_buffer_out;
    uwire  window_buffer_write_enable;
    uwire  window_buffer_read_enable;
    uwire [$clog2(BUF_ELEM_TOTAL)-1:0]  window_buffer_write_addr;
    uwire [$clog2(BUF_ELEM_TOTAL)-1:0]  window_buffer_read_addr;
    $TOP_MODULE_NAME$_cyclic_buffer_addressable #(
        .WIDTH(BUF_IN_WIDTH),
        .DEPTH(BUF_ELEM_TOTAL)
    ) window_buffer_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),

        .write_enable(window_buffer_write_enable),
        .write_addr(window_buffer_write_addr),
        .data_in(window_buffer_in),

        .read_enable(window_buffer_read_enable),
        .read_addr(window_buffer_read_addr),
        .data_out(window_buffer_out)
    );

    //controller instantiation
    uwire  advance_controller;
    uwire signed [INCR_BITWIDTH-1:0]  addr_incr;
    uwire        [INCR_BITWIDTH-1:0]  tail_incr;
    $TOP_MODULE_NAME$_controller controller_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .advance(advance_controller),
        .addr_incr(addr_incr),
        .tail_incr(tail_incr)
    );

    // Counters/address registers
    // Add a sign bit even to (most) unsigned counters and window_buffer_read_addr_reg,
    // so we can use automatic sign extension and simplify calculations w/ signed increment.
    // Alternatively, we could manually sign-extend and shave off a bit here or there.
    logic signed [$clog2(LAST_READ_ELEM+1)+1-1:0]  newest_buffered_elem = -1;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  current_elem = 0;
    logic        [$clog2(LAST_READ_ELEM+1)+1-1:0]  first_elem_next_window = 0;
    logic        [$clog2(ELEM_PER_WINDOW)   -1:0]  k = 0;
    logic        [$clog2(BUF_ELEM_TOTAL)+1  -1:0]  window_buffer_read_addr_reg = 0;
    logic        [$clog2(BUF_ELEM_TOTAL)-1:0]      window_buffer_write_addr_reg = 0;

    // Control signals/registers
    uwire  read_cmd =
        !reading_done && ( // if there is still an input element left to read
            fetching_done || ( // if fetching is done (e.g. for skipped rows at FM end due to stride)  
                $signed(((newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(first_elem_next_window) &&
                $signed(((newest_buffered_elem - (BUF_ELEM_TOTAL - 1)))) < $signed(current_elem)
            ) // (over-)write to buffer if oldest buffered element will no longer be needed  
        ); 
    uwire  read_ok      = read_cmd && in0_V_V_TVALID;
    uwire  reading_done = newest_buffered_elem == LAST_READ_ELEM;

    uwire  fetch_cmd = !($signed(current_elem) > newest_buffered_elem) && !write_blocked && !fetching_done;
    logic  fetching_done = 0;

    logic  write_cmd    = 0;
    logic  writing_done = 0;
    uwire  write_ok      = write_cmd &&  out_V_V_TREADY;
    uwire  write_blocked = write_cmd && !out_V_V_TREADY;;

    //assign buffer control
    assign  window_buffer_write_addr = window_buffer_write_addr_reg;
    assign  window_buffer_read_addr = window_buffer_read_addr_reg;
    assign  window_buffer_write_enable = read_ok;
    assign  window_buffer_read_enable = fetch_cmd;
    assign  advance_controller = fetch_cmd;

    //assign I/O ports
    assign  window_buffer_in = in0_V_V_TDATA;
    assign  out_V_V_TDATA = window_buffer_out;
    assign  in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
    assign  out_V_V_TVALID = ap_rst_n && write_cmd; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

    //main process for advancing counters
    always_ff @(posedge ap_clk) begin
        if(!ap_rst_n) begin
            newest_buffered_elem <= -1;
            current_elem <= 0;
            first_elem_next_window <= 0;
            k <= 0;
            window_buffer_read_addr_reg <= 0;
            window_buffer_write_addr_reg <= 0;
            fetching_done <= 0;
            write_cmd <= 0;
            writing_done <= 0;
        end
        else begin
            if (read_ok) begin
                window_buffer_write_addr_reg <= (window_buffer_write_addr_reg == BUF_ELEM_TOTAL-1)? 0 : window_buffer_write_addr_reg + 1;
                newest_buffered_elem <= newest_buffered_elem+1;

                if (newest_buffered_elem == LAST_READ_ELEM-1) begin
                    window_buffer_write_addr_reg <= 0;
                end
                //check if this is the last read cycle (reading_done will be true afterwards)
                if ((newest_buffered_elem == LAST_READ_ELEM-1) && writing_done) begin
                    //start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                    //todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                    newest_buffered_elem <= -1;
                    current_elem <= 0;
                    window_buffer_read_addr_reg <= 0;
                    first_elem_next_window <= 0;
                    writing_done <= 0;
                    fetching_done <= 0;
                end
            end
                    
            if (fetch_cmd) begin
                //count up to track which element index is about to be read from the buffer, and where it is located within the buffer
                //use increment value calculated by controller

                // absolute buffer address wrap-around
                automatic logic signed [$clog2(BUF_ELEM_TOTAL)+1:0]  ra = $signed(window_buffer_read_addr_reg) + $signed(addr_incr);
                automatic logic signed [$clog2(BUF_ELEM_TOTAL+1):0]  ra_correct =
                    (ra >= BUF_ELEM_TOTAL)? -BUF_ELEM_TOTAL :
                    (ra <               0)?  BUF_ELEM_TOTAL : 0;
                window_buffer_read_addr_reg <= ra + ra_correct;

                //keep track where we are within a window
                k <= (k != ELEM_PER_WINDOW - 1)? k+1 : 0;

                //update first element of next window to allow buffer overwrite up until that point
                if (k == 0)
                    first_elem_next_window <= first_elem_next_window + tail_incr;

                //check if this is the last write cycle (writing_done will be true afterwards)
                if (current_elem == LAST_WRITE_ELEM)
                    fetching_done <= 1;
                else
                    current_elem <= $signed(current_elem) + addr_incr;

                // determine if prefetched data will be outstanding in the next cycle
                // if we fetch in this cycle -> yes
                // if we do not fetch nor write -> do not change
                // if we do not fetch but write successfully-> clear outstanding data
                write_cmd <= fetch_cmd;
            end       

            if (write_ok)
                write_cmd <= fetch_cmd;

            if (write_ok && fetching_done) begin
                //check if this is the last write cycle (writing_done will be true afterwards)
                if (reading_done || (read_ok && (newest_buffered_elem == LAST_READ_ELEM - 1))) begin
                    //start processing of next FM if reading is done already, or completes in the same cycle
                    newest_buffered_elem <= -1;
                    current_elem <= 0;
                    window_buffer_read_addr_reg <= 0;
                    first_elem_next_window <= 0;
                    fetching_done <= 0;
                end else
                    writing_done <= 1;
            end
        end
    end

endmodule : $TOP_MODULE_NAME$_impl

`timescale 1 ns / 1 ps

module $TOP_MODULE_NAME$_controller
(
    CLK,
    RST,
    advance,
    addr_incr,
    tail_incr
);

input CLK;
input RST;
input advance;
output [31:0] addr_incr; //todo: minimize width
output [31:0] tail_incr; //todo: minimize width

////code generation part:
localparam LOOP_H_ITERATIONS = $LOOP_H_ITERATIONS$;
localparam LOOP_W_ITERATIONS = $LOOP_W_ITERATIONS$;
localparam LOOP_KH_ITERATIONS = $LOOP_KH_ITERATIONS$;
localparam LOOP_KW_ITERATIONS = $LOOP_KW_ITERATIONS$;
localparam LOOP_SIMD_ITERATIONS = $LOOP_SIMD_ITERATIONS$;
localparam [31:0] ADDR_INCREMENT_MAP [0:5] = $ADDR_INCREMENT_MAP$; //todo: minimize width
////

//state and counters
reg [2:0] state, state_next;
parameter STATE_START = 0, STATE_LOOP_SIMD = 1, STATE_LOOP_KW = 2, STATE_LOOP_KH = 3, STATE_LOOP_W = 4, STATE_LOOP_H = 5;
integer counter_loop_h; //todo: minimize width
integer counter_loop_w;
integer counter_loop_kh;
integer counter_loop_kw;
integer counter_loop_simd;

assign addr_incr = ADDR_INCREMENT_MAP[state];

//combinational logic for tail_incr generation
$TAIL_INCR_GENERATION$

//combinational next state logic
always @ (state, counter_loop_simd, counter_loop_kw, counter_loop_kh, counter_loop_w, counter_loop_h) begin
    state_next = state; //default
    if (state == $INNERMOST_STATE$) begin
        if (counter_loop_simd == 0)
            if (counter_loop_kw != 0)
                state_next = STATE_LOOP_KW;
            else
                if(counter_loop_kh != 0)
                    state_next = STATE_LOOP_KH;
                else
                    if(counter_loop_w != 0)
                        state_next = STATE_LOOP_W;
                    else
                        if(counter_loop_h != 0)
                            state_next = STATE_LOOP_H;
                        else
                            state_next = STATE_START;
    end else
        state_next = $INNERMOST_STATE$;
end

//sequential logic
always @ (posedge CLK) begin
    if (RST == 1'b0) begin
        counter_loop_h <= LOOP_H_ITERATIONS;
        counter_loop_w <= LOOP_W_ITERATIONS;
        counter_loop_kh <= LOOP_KH_ITERATIONS;
        counter_loop_kw <= LOOP_KW_ITERATIONS;
        counter_loop_simd <= LOOP_SIMD_ITERATIONS;
        state <= $INNERMOST_STATE$; //STATE_START; //debug: omit start state to fix timing, maybe omit during FM transition as well
    end else begin
        if (advance) begin
            state <= state_next;

            if (state == $INNERMOST_STATE$) begin
                if (counter_loop_simd == 0) begin
                    counter_loop_simd <= LOOP_SIMD_ITERATIONS;
                    if (counter_loop_kw == 0) begin
                        counter_loop_kw <= LOOP_KW_ITERATIONS;
                        if (counter_loop_kh == 0) begin
                            counter_loop_kh <= LOOP_KH_ITERATIONS;
                            if (counter_loop_w == 0) begin
                                counter_loop_w <= LOOP_W_ITERATIONS;
                                if (counter_loop_h == 0) begin
                                    counter_loop_h <= LOOP_H_ITERATIONS;
                                end else
                                    counter_loop_h <= counter_loop_h-1;
                            end else
                                counter_loop_w <= counter_loop_w-1;
                        end else
                            counter_loop_kh <= counter_loop_kh-1;
                    end else
                        counter_loop_kw <= counter_loop_kw-1;
                end else
                    counter_loop_simd <= counter_loop_simd-1;
            end
        end
    end
end
endmodule //controller

module $TOP_MODULE_NAME$_cyclic_buffer_addressable
#(
    parameter WIDTH = 1,
    parameter DEPTH = 1
)
(
    CLK,
    RST,
    read_addr,
    read_enable,
    write_enable,
    data_in,
    data_out
);

input CLK, RST, read_enable, write_enable;
input [$clog2(DEPTH)-1:0] read_addr; // absolute (!) read address of cyclic buffer
input [WIDTH-1:0] data_in;
output [WIDTH-1:0] data_out;

integer addr_w; //todo: minimize width (as reg)

$RAM_STYLE$ reg [WIDTH-1:0] ram [DEPTH-1:0];

reg [WIDTH-1:0] out_reg;
assign data_out = out_reg;

always @(posedge CLK) begin 
    if (RST == 1'b0) begin
        addr_w <= 0;
    end else begin
        if (read_enable)
            out_reg <= ram[read_addr];

        if (write_enable) begin
            ram[addr_w] <= data_in;
            
            if (addr_w == DEPTH-1)
                addr_w <= 0;
            else
                addr_w <= addr_w + 1;
        end
    end
end
endmodule //cyclic_buffer_addressable

module $TOP_MODULE_NAME$_impl (
        ap_clk,
        ap_rst_n,
        in0_V_V_TDATA,
        in0_V_V_TVALID,
        in0_V_V_TREADY,
        out_V_V_TDATA,
        out_V_V_TVALID,
        out_V_V_TREADY
);

parameter BIT_WIDTH = $BIT_WIDTH$;
parameter SIMD = $SIMD$;
parameter MMV_IN = $MMV_IN$;
parameter MMV_OUT = $MMV_OUT$;
parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
parameter BUF_OUT_ELEM_WIDTH = BIT_WIDTH * SIMD;
parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;
parameter LAST_READ_ELEM = $LAST_READ_ELEM$;
parameter LAST_WRITE_ELEM = $LAST_WRITE_ELEM$;
parameter BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$;
parameter ELEM_PER_WINDOW = $ELEM_PER_WINDOW$;

//IO ports
input   ap_clk;
input   ap_rst_n;
input  [BUF_IN_WIDTH-1:0] in0_V_V_TDATA;
input   in0_V_V_TVALID;
output   in0_V_V_TREADY;
output  [BUF_OUT_WIDTH-1:0] out_V_V_TDATA;
output   out_V_V_TVALID;
input   out_V_V_TREADY;

//main buffer instantiation
wire [BUF_IN_WIDTH-1:0] window_buffer_in;
wire [BUF_OUT_WIDTH-1:0] window_buffer_out;
wire window_buffer_write_enable;
wire window_buffer_read_enable;
wire [$clog2(BUF_ELEM_TOTAL)-1:0] window_buffer_read_addr;
$TOP_MODULE_NAME$_cyclic_buffer_addressable
#(
    .WIDTH(BUF_IN_WIDTH),
    .DEPTH(BUF_ELEM_TOTAL)
)
window_buffer_inst
(
    .CLK(ap_clk),
    .RST(ap_rst_n),
    .read_addr(window_buffer_read_addr),
    .read_enable(window_buffer_read_enable),
    .write_enable(window_buffer_write_enable),
    .data_in(window_buffer_in),
    .data_out(window_buffer_out)
);

//counters to keep track when to read/write
integer newest_buffered_elem; //todo: minimize width
integer newest_buffered_elem_available; //todo: minimize width
integer current_elem;
integer current_elem_available;
integer first_elem_next_window;
integer k;

reg [$clog2(BUF_ELEM_TOTAL)-1:0] window_buffer_read_addr_reg;
assign window_buffer_read_addr = window_buffer_read_addr_reg;

//reg write_done; //keep track if W of current cycle was already completed, but we still wait on a R in the same cycle

wire advance_controller;
wire [31:0] addr_incr;
wire [31:0] tail_incr;

$TOP_MODULE_NAME$_controller
controller_inst
(
    .CLK(ap_clk),
    .RST(ap_rst_n),
    .advance(advance_controller),
    .addr_incr(addr_incr),
    .tail_incr(tail_incr)
);

wire reading_done;
assign reading_done = newest_buffered_elem == LAST_READ_ELEM;

reg fetching_done;
reg writing_done; //instead of a separate write cycle/element counter, trigger this flag once current_element reaches LAST_WRITE_ELEM
//assign writing_done = current_elem == LAST_WRITE_ELEM;


wire write_blocked;

//reg write_prefetch_available; // stores if the write of prefetched data is still outstanding

wire fetch_cmd;
assign fetch_cmd = !(current_elem > newest_buffered_elem) && !write_blocked && !fetching_done;
    
    
//determine whether to read/write in this cycle
//wire write_cmd;
//assign write_cmd = write_prefetch_available && !writing_done;
reg write_cmd;                 



wire read_cmd;
assign read_cmd = 
    (
      (  
          (newest_buffered_elem - BUF_ELEM_TOTAL+1) < first_elem_next_window
        &&(newest_buffered_elem - BUF_ELEM_TOTAL+1) < current_elem
      )  // (over-)write to buffer if oldest buffered element is no longer needed  
      || fetching_done
    )                                                      //or if fetching is done (e.g. for skipped rows at FM end due to stride)
    && !reading_done;                                                    //and if there is still an input element left to read

//todo: optmize (e.g. is < or != more efficient?)
// ToDo: ideally this should point to the oldest elem of the next window,
// to allow reading while still writing the remainder of the current window                 



assign write_blocked = write_cmd && !out_V_V_TREADY; //&& !write_done;

wire read_ok;
// with transition to next cycle:
//              want to read      can read       source is ready (waiting on VALID allowed)
assign read_ok = read_cmd && !write_blocked && in0_V_V_TVALID;

wire write_ok;
// with transition to next cycle:
//              output is VALID   sink is ready  sink has already read (we are waiting on source)
//assign write_ok = write_cmd && (out_V_V_TREADY || write_done);
assign write_ok = write_cmd && out_V_V_TREADY;

//wire advance;
//            includes waiting on W    if W-only cycle: wait only on W     no R/W to wait for
//assign advance =      read_ok        ||   (!read_cmd && write_ok)    || (!read_cmd && !write_cmd);
//todo: optimize/simplify advance logic for write_done generation

//assign buffer control
assign window_buffer_write_enable = read_ok;
assign window_buffer_read_enable = fetch_cmd;
assign advance_controller = fetch_cmd; //write_ok

//assign I/O ports
assign window_buffer_in = in0_V_V_TDATA;
assign out_V_V_TDATA = window_buffer_out;
assign in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
assign out_V_V_TVALID = ap_rst_n && write_cmd; //&& !write_done; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

//main process for advancing counters
always @ (posedge ap_clk) begin
    if (ap_rst_n == 1'b0) begin
        newest_buffered_elem <= -1;
        //newest_buffered_elem_available <= -1;
        current_elem <= 0;
        //current_elem_available <= 0;
        first_elem_next_window <= 0;
        k <= 0;
        window_buffer_read_addr_reg <= 0;
        fetching_done <= 0;
        writing_done <= 0;
        //write_prefetch_available <= 0;
        write_cmd <= 0;
    end else begin
        if (read_ok) begin
            //check if this is the last read cycle (reading_done will be true afterwards)
            if ((newest_buffered_elem == LAST_READ_ELEM-1) && writing_done) begin
                //start processing of next FM if writing is done already (possible due to unused input elements at the tail end)
                //todo: allow for read overlapping between feature maps (i.e., reading first elements from next FM while still writing last window of current FM)
                newest_buffered_elem <= -1;
                current_elem <= 0;
                first_elem_next_window <= 0;
                writing_done <= 0;
                fetching_done <= 0;
            end
            
            newest_buffered_elem <= newest_buffered_elem+1;
        end
                 
        if (fetch_cmd) begin
            //count up to track which element index is about to be read from the buffer, and where it is located within the buffer
            //use increment value calculated by controller

            //keep track where we are within a window
            if (k == ELEM_PER_WINDOW-1)
                k <= 0;
            else
                k <= k+1;

            //absolute buffer address always wraps around (in both directions for depthwise support)
            if ($signed(window_buffer_read_addr_reg + addr_incr) > BUF_ELEM_TOTAL-1)
                window_buffer_read_addr_reg <= window_buffer_read_addr_reg + addr_incr - BUF_ELEM_TOTAL;
            else if ($signed(window_buffer_read_addr_reg + addr_incr) < 0)
                window_buffer_read_addr_reg <= window_buffer_read_addr_reg + addr_incr + BUF_ELEM_TOTAL;
            else
                window_buffer_read_addr_reg <= window_buffer_read_addr_reg + addr_incr;

            //check if this is the last write cycle (writing_done will be true afterwards)
            if (current_elem == LAST_WRITE_ELEM) begin
                fetching_done <= 1;
            end else begin
                //current element index wraps around only at window boundary
                //if (((current_elem + addr_incr) > BUF_ELEM_TOTAL-1) && (k == ELEM_PER_WINDOW-1))
                
                //if (k == ELEM_PER_WINDOW-1)
                //    current_elem <= current_elem + addr_incr - BUF_ELEM_TOTAL;
                //else
                    current_elem <= current_elem + addr_incr;
            end

            if (k == 0)
                first_elem_next_window <= first_elem_next_window + tail_incr;

            // determine if prefetched data will be outstanding in the next cycle
            // if we fetch in this cycle -> yes
            // if we do not fetch nor write successfully -> do not change
            // if we do not fetch but write -> clear outstanding data
            //write_prefetch_available <= fetch_cmd;
            write_cmd <= fetch_cmd;
        end       

        if (write_ok)
            // determine if prefetched data will be outstanding in the next cycle
            // if we fetch in this cycle -> yes
            // if we do not fetch nor write successfully -> do not change
            // if we do not fetch but write -> clear outstanding data
            //write_prefetch_available <= fetch_cmd;
            write_cmd <= fetch_cmd;

        if (write_ok && fetching_done) begin
            //check if this is the last write cycle (writing_done will be true afterwards)
            if (reading_done || (read_ok && (newest_buffered_elem == LAST_READ_ELEM-1))) begin
                //start processing of next FM if reading is done already, or completes in the same cycle
                newest_buffered_elem <= -1;
                current_elem <= 0;
                first_elem_next_window <= 0;
                fetching_done <= 0;
            end else
                writing_done <= 1;
        end

        //if (advance)
        //    write_done <= 1'b0; //reset flag
        //else if (write_ok) // successful W in this cycle, but R still outstanding
        //    write_done <= 1'b1; //write can happen even if read is blocked, but only for the current cycle!
    end
end

endmodule //TOP_MODULE_NAME_impl

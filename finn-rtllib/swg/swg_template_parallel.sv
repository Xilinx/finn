`timescale 1 ns / 1 ps

module $TOP_MODULE_NAME$_controller
(
    CLK,
    RST,
    advance,
    cmd_read,
    cmd_write
);

input CLK;
input RST;
input advance;
output cmd_read;
output cmd_write;

////code generation part:
//mapping of R/W command values to each state (START, MAIN_1, MAIN_2, INTER_1, INTER_2, END_1, END_2)
localparam [0:6] READ_CMD_MAP = $READ_CMD_MAP$;
localparam [0:6] WRITE_CMD_MAP = $WRITE_CMD_MAP$;

localparam START_COUNTER = $START_COUNTER$;
localparam LOOP_MAIN_COUNTER = $LOOP_MAIN_COUNTER$;
localparam LOOP_MAIN_1_COUNTER = $LOOP_MAIN_1_COUNTER$;
localparam LOOP_MAIN_2_COUNTER = $LOOP_MAIN_2_COUNTER$;
localparam LOOP_INTER_COUNTER = $LOOP_INTER_COUNTER$;
localparam LOOP_INTER_1_COUNTER = $LOOP_INTER_1_COUNTER$;
localparam LOOP_INTER_2_COUNTER = $LOOP_INTER_2_COUNTER$;
localparam LOOP_END_1_COUNTER = $LOOP_END_1_COUNTER$;
localparam LOOP_END_2_COUNTER = $LOOP_END_2_COUNTER$;
////

//state and counters
reg [2:0] state, state_next;
parameter STATE_START = 0, STATE_LOOP_MAIN_1 = 1, STATE_LOOP_MAIN_2 = 2, STATE_LOOP_INTER_1 = 3, STATE_LOOP_INTER_2 = 4, STATE_END_1 = 5, STATE_END_2 = 6;
integer counter_current; //todo: minimize width
integer counter_loop_main;
integer counter_loop_inter;

assign cmd_read = READ_CMD_MAP[state_next]; //read command indicates read in *upcoming* cycle, due to how schedule is constructed
assign cmd_write = WRITE_CMD_MAP[state];

//combinational next state logic
always @ (state, counter_current, counter_loop_main, counter_loop_inter) begin
    state_next = state; //default
    case (state)
        STATE_START:
            if (counter_current == START_COUNTER-1)
                state_next = STATE_LOOP_MAIN_1;

        STATE_LOOP_MAIN_1:
            if (counter_current == LOOP_MAIN_1_COUNTER-1)
                state_next = STATE_LOOP_MAIN_2;

        STATE_LOOP_MAIN_2: begin
            if (counter_current == LOOP_MAIN_2_COUNTER-1) begin
                state_next = STATE_LOOP_MAIN_1;
                if (counter_loop_main == LOOP_MAIN_COUNTER-1) begin
                    //no -1 because this counter marks the currently active iteration, not finished iterations
                    if ((LOOP_INTER_COUNTER != 0) && (counter_loop_inter != LOOP_INTER_COUNTER))
                        state_next = STATE_LOOP_INTER_1;
                    else begin
                        //there might not be an end sequence -> restart immediately
                        if (LOOP_END_1_COUNTER != 0)
                            state_next = STATE_END_1;
                        else
                            state_next = STATE_LOOP_MAIN_2; //wait in current state until reset
                    end
                end
            end
        end

        STATE_LOOP_INTER_1: begin
            if (counter_current == LOOP_INTER_1_COUNTER-1) begin
                if (LOOP_INTER_2_COUNTER != 0)
                    state_next = STATE_LOOP_INTER_2;
                else
                    state_next = STATE_LOOP_MAIN_1;
            end
        end

        STATE_LOOP_INTER_2:
            if (counter_current == LOOP_INTER_2_COUNTER-1)
                state_next = STATE_LOOP_MAIN_1;

        STATE_END_1: begin
            if (counter_current == LOOP_END_1_COUNTER-1) begin
                if (LOOP_END_2_COUNTER != 0)
                    state_next = STATE_END_2;
                else
                    state_next = STATE_END_1; //wait in current state until reset
            end
        end

        STATE_END_2:
            if (counter_current == LOOP_END_2_COUNTER-1)
                state_next = STATE_END_2; //wait in current state until reset
    endcase
end

//sequential logic
always @ (posedge CLK) begin
    if (RST) begin
        counter_current <= -1;
        counter_loop_main <= 0;
        counter_loop_inter <= 0;
        state <= STATE_START;
    end else begin
        if (advance) begin
            counter_current <= counter_current+1;
            state <= state_next;

            if (state != state_next) begin
                counter_current <= 0;

                //count up main loop upon re-entering this loop (not on first enter from start)
                if ((state_next == STATE_LOOP_MAIN_1) && (state != STATE_START)) begin
                    if (counter_loop_main == LOOP_MAIN_COUNTER-1) begin
                        counter_loop_main <= 0;
                    end else begin
                        counter_loop_main <= counter_loop_main+1;
                    end
                end

                if (state_next == STATE_LOOP_INTER_1) begin
                    if (counter_loop_inter == LOOP_INTER_COUNTER) begin //no -1 because this counter marks the currently active iteration, not finished iterations
                        counter_loop_inter <= 0;
                    end else begin
                        counter_loop_inter <= counter_loop_inter+1;
                    end
                end
            end
        end
    end
end
endmodule //controller

module $TOP_MODULE_NAME$_reg_buffer
#(
    parameter WIDTH = 1,
    parameter DEPTH = 1
)
(
    CLK,
    shift_enable,
    shift_in,
    shift_out,
    data_out
);

input CLK, shift_enable;
input [WIDTH-1:0] shift_in;
output [WIDTH-1:0] shift_out;
output [WIDTH*DEPTH-1:0] data_out;

// ToDo: experiment with SRL instead of FF-based shift register
// by force or by achieving automatic SRL inference
//UG901 template for SRL inference:
// 32-bit Shift Register
// Rising edge clock
// Active high clock enable
// For-loop based template
// File: shift_registers_1.v
//
//module shift_registers_1 (clk, clken, SI, SO);
//parameter WIDTH = 32;
//input clk, clken, SI;
//output SO;
//reg [WIDTH-1:0] shreg;
//
//integer i;
//always @(posedge clk)
//begin
//  if (clken)
//    begin
//    for (i = 0; i < WIDTH-1; i = i+1)
//        shreg[i+1] <= shreg[i];
//      shreg[0] <= SI;
//    end
//end
//assign SO = shreg[WIDTH-1];
//endmodule

reg [WIDTH-1:0] data [DEPTH-1:0];

assign shift_out = data[DEPTH-1];

for (genvar e=0; e<DEPTH; e=e+1)
    assign data_out[e*WIDTH +: WIDTH] = data[e];

always @ (posedge CLK) begin
    if (shift_enable) begin
        for (integer i=DEPTH-1; i>0; i=i-1)
            data[i] <= data[i-1];
        data[0] <= shift_in;
    end
end
endmodule //reg_buffer

module $TOP_MODULE_NAME$_ram_buffer
#(
    parameter WIDTH = 1,
    parameter DEPTH = 1
)
(
    CLK,
    RST,
    shift_enable,
    shift_in,
    shift_out
);

input CLK, RST, shift_enable;
input [WIDTH-1:0] shift_in;
output [WIDTH-1:0] shift_out;

reg [WIDTH-1:0] out_reg;
assign shift_out = out_reg;

integer addr_w, addr_r; //todo: minimize width (as reg), make r addr depend on w

$RAM_STYLE$ reg [WIDTH-1:0] ram [DEPTH-1:0];

always @(posedge CLK) begin
    if (RST == 1'b0) begin
        addr_w <= 0;
        addr_r <= 1;
    end else begin
        if (shift_enable) begin
            ram[addr_w] <= shift_in;
            out_reg <= ram[addr_r];

            if (addr_w == DEPTH-1)
                addr_w <= 0;
            else
                addr_w <= addr_w + 1;

            if (addr_r == DEPTH-1)
                addr_r <= 0;
            else
                addr_r <= addr_r + 1;
        end
    end
end
endmodule //ram_buffer

module $TOP_MODULE_NAME$_wb
#(
    parameter IN_WIDTH = 1, //bit-width*C*MMV_in
    parameter OUT_ELEM_WIDTH = 1, //bit-width*C
    parameter OUT_WIDTH = 1, //bit-width*C*MMV_out
    parameter BUFFER_ELEM_TOTAL = 1
)
(
    CLK,
    RST,
    data_in,
    shift_enable,
    data_out
);

input CLK, RST;
input [IN_WIDTH-1:0] data_in;
input shift_enable;
output [OUT_WIDTH-1:0] data_out;

//Input REG to enable simultaneous R/W
reg [IN_WIDTH-1:0] reg_input;

$GENERATE_REG_FIFOS$

$GENERATE_BRAM_FIFOS$

//Fixed interconnect between linear buffers
$GENERATE_BUFFER_CONNECTION$

//Fixed REG FIFO <-> output mapping
$GENERATE_OUTPUT_MAPPING$

//input register logic
integer i;
always @ (posedge CLK) begin
    if (shift_enable) begin
        reg_input <= data_in;
    end
end

endmodule //window_buffer

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
parameter SIMD = $SIMD$; //assuming SIMD = C for this implementation style
parameter MMV_IN = $MMV_IN$;
parameter MMV_OUT = $MMV_OUT$;
parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
parameter BUF_OUT_ELEM_WIDTH = BIT_WIDTH * SIMD;
parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;
parameter CYCLES_TOTAL = $CYCLES_TOTAL$;
parameter BUF_ELEM_TOTAL = $BUF_ELEM_TOTAL$;

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
wire window_buffer_shift_enable;
$TOP_MODULE_NAME$_wb
#(
    .IN_WIDTH(BUF_IN_WIDTH),
    .OUT_ELEM_WIDTH(BUF_OUT_ELEM_WIDTH),
    .OUT_WIDTH(BUF_OUT_WIDTH),
    .BUFFER_ELEM_TOTAL(BUF_ELEM_TOTAL)
)
window_buffer_inst
(
    .CLK(ap_clk),
    .RST(ap_rst_n),
    .data_in(window_buffer_in),
    .shift_enable(window_buffer_shift_enable),
    .data_out(window_buffer_out)
);

integer cycle; //main cycle counter (where either read/write/both happen, resets for each image)
wire read_cmd;
wire write_cmd;
reg write_done; //keep track if W of current cycle was already completed, but we still wait on a R in the same cycle

wire controller_reset;
wire controller_advance;

$TOP_MODULE_NAME$_controller
controller_inst
(
    .CLK(ap_clk),
    .RST(controller_reset),
    .advance(controller_advance),
    .cmd_read(read_cmd),
    .cmd_write(write_cmd)
);

wire write_blocked;
assign write_blocked = write_cmd && !out_V_V_TREADY && !write_done;

wire read_ok;
// with transition to next cycle:
//              want to read      can read       source is ready (waiting on VALID allowed)
assign read_ok = read_cmd && !write_blocked && in0_V_V_TVALID;

wire write_ok;
// with transition to next cycle:
//              output is VALID   sink is ready  sink has already read (we are waiting on source)
assign write_ok = write_cmd && (out_V_V_TREADY || write_done);

wire advance;
//            includes waiting on W    if W-only cycle: wait only on W     no R/W to wait for
assign advance =      read_ok        ||   (!read_cmd && write_ok)    || (!read_cmd && !write_cmd);

//assign buffer control
//todo: if mmv_out < k: might not shift and/or write for multiple read_cmd cycles
assign window_buffer_shift_enable = advance;

assign controller_reset = !ap_rst_n || ((cycle == CYCLES_TOTAL-1) && advance);
assign controller_advance = advance;

//assign I/O ports
assign window_buffer_in = in0_V_V_TDATA;
assign out_V_V_TDATA = window_buffer_out;
assign in0_V_V_TREADY = ap_rst_n && read_ok; //only asserted if data is available and we can store it (allowed)
assign out_V_V_TVALID = ap_rst_n && write_cmd && !write_done; //only asserted if we have data available and it has not been read yet (don't wait for READY from sink)

//main process for advancing cycle count
always @ (posedge ap_clk) begin
    if (ap_rst_n == 1'b0) begin
        cycle <= 0;
    end else begin
        if (advance) begin
            write_done <= 1'b0; //reset flag

            //count cycle (completed R or W or both (depending on current cycle))
            if (cycle == CYCLES_TOTAL-1)
                cycle <= 0;
            else
                cycle <= cycle+1;

        end else if (write_ok) // successful W in this cycle, but R still outstanding
            write_done <= 1'b1; //write can happen even if read is blocked, but only for the current cycle!
    end
end

endmodule //TOP_MODULE_NAME_impl

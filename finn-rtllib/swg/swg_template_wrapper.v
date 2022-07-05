`timescale 1 ns / 1 ps

module $TOP_MODULE_NAME$ (
        ap_clk,
        ap_rst_n,
        in0_V_TDATA,
        in0_V_TVALID,
        in0_V_TREADY,
        out_V_TDATA,
        out_V_TVALID,
        out_V_TREADY
);

parameter BIT_WIDTH = $BIT_WIDTH$;
parameter SIMD = $SIMD$;
parameter MMV_IN = $MMV_IN$;
parameter MMV_OUT = $MMV_OUT$;
parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN;
parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT;

input  ap_clk;
input  ap_rst_n;
(* X_INTERFACE_PARAMETER = "FREQ_HZ 100000000.000000" *) //todo: make configurable or set later
input  [BUF_IN_WIDTH-1:0] in0_V_TDATA;
input  in0_V_TVALID;
output in0_V_TREADY;
(* X_INTERFACE_PARAMETER = "FREQ_HZ 100000000.000000" *)
output [BUF_OUT_WIDTH-1:0] out_V_TDATA;
output out_V_TVALID;
input  out_V_TREADY;

$TOP_MODULE_NAME$_impl
#()
impl
(
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .in0_V_V_TDATA(in0_V_TDATA),
    .in0_V_V_TVALID(in0_V_TVALID),
    .in0_V_V_TREADY(in0_V_TREADY),
    .out_V_V_TDATA(out_V_TDATA),
    .out_V_V_TVALID(out_V_TVALID),
    .out_V_V_TREADY(out_V_TREADY)
);

endmodule //TOP_MODULE_NAME

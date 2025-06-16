module $TOP_MODULE_NAME$ #(
	parameter  DEPTH = $DEPTH$,
	parameter  WIDTH = $WIDTH$,
    parameter  AXILITE_ADDR_WIDTH = $clog2(DEPTH * (2**$clog2((WIDTH+31)/32))) + 2
)(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF $TOP_PORT_NAMES$, ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   $CLK_NAME$,
$CLK2X$
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   $RST_NAME$,

//- AXI Lite ------------------------
$AXILITE$
//- AXI Stream - Input --------------
$S_AXIS$
//- AXI Stream - Output -------------
$M_AXIS$
);


$MVAU$


$MEMSTREAM$

endmodule

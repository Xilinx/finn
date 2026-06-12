module $MODULE_NAME_AXI_WRAPPER$ #(
    parameter   BASE_ADDR_BITS = 64,
    parameter   AXIL_DATA_BITS = 32,
    parameter   AXIL_ADDR_BITS = 3
)(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input   ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input   ap_rst_n,

    input                                      s_axilite_AWVALID,
    output                                     s_axilite_AWREADY,
    input  [AXIL_ADDR_BITS-1:0]                s_axilite_AWADDR,
    input  [2:0]                               s_axilite_AWPROT,
    input                                      s_axilite_WVALID,
    output                                     s_axilite_WREADY,
    input  [AXIL_DATA_BITS-1:0]                s_axilite_WDATA,
    input  [AXIL_DATA_BITS/8-1:0]              s_axilite_WSTRB,
    output                                     s_axilite_BVALID,
    input                                      s_axilite_BREADY,
    output [1:0]                               s_axilite_BRESP,
    input                                      s_axilite_ARVALID,
    output                                     s_axilite_ARREADY,
    input  [AXIL_ADDR_BITS-1:0]                s_axilite_ARADDR,
    input  [2:0]                               s_axilite_ARPROT,
    output                                     s_axilite_RVALID,
    input                                      s_axilite_RREADY,
    output [AXIL_DATA_BITS-1:0]                s_axilite_RDATA,
    output [1:0]                               s_axilite_RRESP,

    output [BASE_ADDR_BITS-1:0]                base_address
);

address_config #(
    .BASE_ADDR_BITS(BASE_ADDR_BITS),
    .AXIL_DATA_BITS(AXIL_DATA_BITS),
    .AXIL_ADDR_BITS(AXIL_ADDR_BITS)
) inst (
    .aclk               (ap_clk),
    .aresetn            (ap_rst_n),
    .s_axilite_awaddr   (s_axilite_AWADDR),
    .s_axilite_awprot   (s_axilite_AWPROT),
    .s_axilite_awvalid  (s_axilite_AWVALID),
    .s_axilite_awready  (s_axilite_AWREADY),
    .s_axilite_wdata    (s_axilite_WDATA),
    .s_axilite_wstrb    (s_axilite_WSTRB),
    .s_axilite_wvalid   (s_axilite_WVALID),
    .s_axilite_wready   (s_axilite_WREADY),
    .s_axilite_bresp    (s_axilite_BRESP),
    .s_axilite_bvalid   (s_axilite_BVALID),
    .s_axilite_bready   (s_axilite_BREADY),
    .s_axilite_araddr   (s_axilite_ARADDR),
    .s_axilite_arprot   (s_axilite_ARPROT),
    .s_axilite_arvalid  (s_axilite_ARVALID),
    .s_axilite_arready  (s_axilite_ARREADY),
    .s_axilite_rdata    (s_axilite_RDATA),
    .s_axilite_rresp    (s_axilite_RRESP),
    .s_axilite_rvalid   (s_axilite_RVALID),
    .s_axilite_rready   (s_axilite_RREADY),
    .base_address       (base_address)
);

endmodule

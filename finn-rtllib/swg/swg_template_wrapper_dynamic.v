`timescale 1 ns / 1 ps

module $TOP_MODULE_NAME$ #(
    // top-level parameters (set via code-generation)
    parameter BIT_WIDTH = $BIT_WIDTH$,
    parameter SIMD = $SIMD$,
    parameter MMV_IN = $MMV_IN$,
    parameter MMV_OUT = $MMV_OUT$,

    parameter CNTR_BITWIDTH = $CNTR_BITWIDTH$,
    parameter INCR_BITWIDTH = $INCR_BITWIDTH$,

    // derived constants
    parameter BUF_IN_WIDTH = BIT_WIDTH * SIMD * MMV_IN,
    parameter BUF_OUT_WIDTH = BIT_WIDTH * SIMD * MMV_OUT,

    parameter integer C_s_axi_cfg_DATA_WIDTH	= 32,
    parameter integer C_s_axi_cfg_ADDR_WIDTH	= 6
)
(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axi_cfg" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V:s_axi_cfg" *)
    input  ap_rst_n,
    input  [BUF_IN_WIDTH-1:0] in0_V_TDATA,
    input  in0_V_TVALID,
    output in0_V_TREADY,
    output [BUF_OUT_WIDTH-1:0] out_V_TDATA,
    output out_V_TVALID,
    input  out_V_TREADY,

    // Ports of Axi Slave Bus Interface s_axi_cfg
    //input wire  s_axi_cfg_aclk,
    //input wire  s_axi_cfg_aresetn,
    input wire [C_s_axi_cfg_ADDR_WIDTH-1 : 0] s_axi_cfg_awaddr,
    input wire [2 : 0] s_axi_cfg_awprot,
    input wire  s_axi_cfg_awvalid,
    output wire  s_axi_cfg_awready,
    input wire [C_s_axi_cfg_DATA_WIDTH-1 : 0] s_axi_cfg_wdata,
    input wire [(C_s_axi_cfg_DATA_WIDTH/8)-1 : 0] s_axi_cfg_wstrb,
    input wire  s_axi_cfg_wvalid,
    output wire  s_axi_cfg_wready,
    output wire [1 : 0] s_axi_cfg_bresp,
    output wire  s_axi_cfg_bvalid,
    input wire  s_axi_cfg_bready,
    input wire [C_s_axi_cfg_ADDR_WIDTH-1 : 0] s_axi_cfg_araddr,
    input wire [2 : 0] s_axi_cfg_arprot,
    input wire  s_axi_cfg_arvalid,
    output wire  s_axi_cfg_arready,
    output wire [C_s_axi_cfg_DATA_WIDTH-1 : 0] s_axi_cfg_rdata,
    output wire [1 : 0] s_axi_cfg_rresp,
    output wire  s_axi_cfg_rvalid,
    input wire  s_axi_cfg_rready
);

wire                     cfg_valid;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_simd;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_kw;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_kh;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_w;
wire [CNTR_BITWIDTH-1:0] cfg_cntr_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_simd;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_kw;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_kh;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_w;
wire [INCR_BITWIDTH-1:0] cfg_incr_head_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_w;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_h;
wire [INCR_BITWIDTH-1:0] cfg_incr_tail_last;
wire [31:0]              cfg_last_read;
wire [31:0]              cfg_last_write;

// Instantiation of Axi Bus Interface s_axi_cfg
$TOP_MODULE_NAME$_axilite # (
    .C_S_AXI_DATA_WIDTH(C_s_axi_cfg_DATA_WIDTH),
    .C_S_AXI_ADDR_WIDTH(C_s_axi_cfg_ADDR_WIDTH)
) axilite_cfg_inst (
    .S_AXI_ACLK(ap_clk),
    .S_AXI_ARESETN(ap_rst_n),
    .S_AXI_AWADDR(s_axi_cfg_awaddr),
    .S_AXI_AWPROT(s_axi_cfg_awprot),
    .S_AXI_AWVALID(s_axi_cfg_awvalid),
    .S_AXI_AWREADY(s_axi_cfg_awready),
    .S_AXI_WDATA(s_axi_cfg_wdata),
    .S_AXI_WSTRB(s_axi_cfg_wstrb),
    .S_AXI_WVALID(s_axi_cfg_wvalid),
    .S_AXI_WREADY(s_axi_cfg_wready),
    .S_AXI_BRESP(s_axi_cfg_bresp),
    .S_AXI_BVALID(s_axi_cfg_bvalid),
    .S_AXI_BREADY(s_axi_cfg_bready),
    .S_AXI_ARADDR(s_axi_cfg_araddr),
    .S_AXI_ARPROT(s_axi_cfg_arprot),
    .S_AXI_ARVALID(s_axi_cfg_arvalid),
    .S_AXI_ARREADY(s_axi_cfg_arready),
    .S_AXI_RDATA(s_axi_cfg_rdata),
    .S_AXI_RRESP(s_axi_cfg_rresp),
    .S_AXI_RVALID(s_axi_cfg_rvalid),
    .S_AXI_RREADY(s_axi_cfg_rready),

    .cfg_reg0(cfg_valid),
    .cfg_reg1(cfg_cntr_simd),
    .cfg_reg2(cfg_cntr_kw),
    .cfg_reg3(cfg_cntr_kh),
    .cfg_reg4(cfg_cntr_w),
    .cfg_reg5(cfg_cntr_h),
    .cfg_reg6(cfg_incr_head_simd),
    .cfg_reg7(cfg_incr_head_kw),
    .cfg_reg8(cfg_incr_head_kh),
    .cfg_reg9(cfg_incr_head_w),
    .cfg_reg10(cfg_incr_head_h),
    .cfg_reg11(cfg_incr_tail_w),
    .cfg_reg12(cfg_incr_tail_h),
    .cfg_reg13(cfg_incr_tail_last),
    .cfg_reg14(cfg_last_read),
    .cfg_reg15(cfg_last_write)
);

$TOP_MODULE_NAME$_impl
#(
    .BIT_WIDTH(BIT_WIDTH),
    .SIMD(SIMD),
    .MMV_IN(MMV_IN),
    .MMV_OUT(MMV_OUT),
    .CNTR_BITWIDTH(CNTR_BITWIDTH),
    .INCR_BITWIDTH(INCR_BITWIDTH)
)
impl
(
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .in0_V_V_TDATA(in0_V_TDATA),
    .in0_V_V_TVALID(in0_V_TVALID),
    .in0_V_V_TREADY(in0_V_TREADY),
    .out_V_V_TDATA(out_V_TDATA),
    .out_V_V_TVALID(out_V_TVALID),
    .out_V_V_TREADY(out_V_TREADY),

    .cfg_valid(cfg_valid),
    .cfg_cntr_simd(cfg_cntr_simd),
    .cfg_cntr_kw(cfg_cntr_kw),
    .cfg_cntr_kh(cfg_cntr_kh),
    .cfg_cntr_w(cfg_cntr_w),
    .cfg_cntr_h(cfg_cntr_h),
    .cfg_incr_head_simd(cfg_incr_head_simd),
    .cfg_incr_head_kw(cfg_incr_head_kw),
    .cfg_incr_head_kh(cfg_incr_head_kh),
    .cfg_incr_head_w(cfg_incr_head_w),
    .cfg_incr_head_h(cfg_incr_head_h),
    .cfg_incr_tail_w(cfg_incr_tail_w),
    .cfg_incr_tail_h(cfg_incr_tail_h),
    .cfg_incr_tail_last(cfg_incr_tail_last),
    .cfg_last_read(cfg_last_read),
    .cfg_last_write(cfg_last_write)
);

endmodule //TOP_MODULE_NAME

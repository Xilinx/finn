module $LOOP_CONTROL_WRAPPER_NAME$ #(
    parameter N_MAX_LAYERS   = $N_MAX_LAYERS$,
    parameter ADDR_BITS      = 64,
    parameter DATA_BITS      = 256,
    parameter LEN_BITS       = 32,
    parameter CNT_BITS       = 16,
    parameter ILEN_BITS      = $ILEN_BITS$,
    parameter OLEN_BITS      = $OLEN_BITS$,
    parameter M_AXI_HBM_BASE_ADDR = 0, // m_axi_hbm base address
    parameter LAYER_OFFS_INT = $LAYER_OFFS_INT$ // calculate layer offsets in intermediate buffer => 0
) (
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axi_hbm:m_axis_core_in:m_axis_core_in_fw_idx:s_axis_core_out:in0_V:out0_V:idx_fs:idx_se, ASSOCIATED_RESET = ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input   ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input   ap_rst_n,

    // AXI4 master interface for m_axi_hbm
    output [ADDR_BITS-1:0] m_axi_hbm_araddr,
    output [1:0]           m_axi_hbm_arburst,
    output [3:0]           m_axi_hbm_arcache,
    output [1:0]           m_axi_hbm_arid,
    output [7:0]           m_axi_hbm_arlen,
    output                 m_axi_hbm_arlock,
    output [2:0]           m_axi_hbm_arprot,
    output [2:0]           m_axi_hbm_arsize,
    input                  m_axi_hbm_arready,
    output                 m_axi_hbm_arvalid,
    output [ADDR_BITS-1:0] m_axi_hbm_awaddr,
    output [1:0]           m_axi_hbm_awburst,
    output [3:0]           m_axi_hbm_awcache,
    output [1:0]           m_axi_hbm_awid,
    output [7:0]           m_axi_hbm_awlen,
    output                 m_axi_hbm_awlock,
    output [2:0]           m_axi_hbm_awprot,
    output [2:0]           m_axi_hbm_awsize,
    input                  m_axi_hbm_awready,
    output                 m_axi_hbm_awvalid,
    input  [DATA_BITS-1:0] m_axi_hbm_rdata,
    input  [1:0]           m_axi_hbm_rid,
    input                  m_axi_hbm_rlast,
    input  [1:0]           m_axi_hbm_rresp,
    output                 m_axi_hbm_rready,
    input                  m_axi_hbm_rvalid,
    output [DATA_BITS-1:0] m_axi_hbm_wdata,
    output                 m_axi_hbm_wlast,
    output [DATA_BITS/8-1:0] m_axi_hbm_wstrb,
    input                  m_axi_hbm_wready,
    output                 m_axi_hbm_wvalid,
    input  [1:0]           m_axi_hbm_bid,
    input  [1:0]           m_axi_hbm_bresp,
    output                 m_axi_hbm_bready,
    input                  m_axi_hbm_bvalid,

    // AXI4S master interface for core_in
    output [DATA_BITS-1:0] m_axis_core_in_tdata,
    output                 m_axis_core_in_tvalid,
    input                  m_axis_core_in_tready,

    // AXI4S master interface for core_in_fw_idx
    output [DATA_BITS-1:0] m_axis_core_in_fw_idx_tdata,
    output                 m_axis_core_in_fw_idx_tvalid,
    input                  m_axis_core_in_fw_idx_tready,

    // AXI4S slave interface for core_out_fw_idx
    input  [DATA_BITS-1:0] s_axis_core_out_fw_idx_tdata,
    input                  s_axis_core_out_fw_idx_tvalid,
    output                 s_axis_core_out_fw_idx_tready,

    // AXI4S slave interface for core_out
    input  [DATA_BITS-1:0] s_axis_core_out_tdata,
    input                  s_axis_core_out_tvalid,
    output                 s_axis_core_out_tready,

    // activation signals
    input  [DATA_BITS-1:0] in0_V_tdata,
    input                  in0_V_tvalid,
    output                 in0_V_tready,
    output [DATA_BITS-1:0] out0_V_tdata,
    output                 out0_V_tvalid,
    input                  out0_V_tready,

    // control signals
    output wire [1:0]         done_if
);

    loop_control #(
        .N_MAX_LAYERS(N_MAX_LAYERS),
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),
        .ILEN_BITS(ILEN_BITS),
        .OLEN_BITS(OLEN_BITS),
        .M_AXI_HBM_BASE_ADDR(M_AXI_HBM_BASE_ADDR),
        .LAYER_OFFS_INT(LAYER_OFFS_INT)
    ) loop_control_inst (
       .aclk(ap_clk),
       .aresetn(ap_rst_n),

       // AXI4 master interface for m_axi_hbm
       .m_axi_hbm_araddr(m_axi_hbm_araddr),
       .m_axi_hbm_arburst(m_axi_hbm_arburst),
       .m_axi_hbm_arcache(m_axi_hbm_arcache),
       .m_axi_hbm_arid(m_axi_hbm_arid),
       .m_axi_hbm_arlen(m_axi_hbm_arlen),
       .m_axi_hbm_arlock(m_axi_hbm_arlock),
       .m_axi_hbm_arprot(m_axi_hbm_arprot),
       .m_axi_hbm_arsize(m_axi_hbm_arsize),
       .m_axi_hbm_arready(m_axi_hbm_arready),
       .m_axi_hbm_arvalid(m_axi_hbm_arvalid),
       .m_axi_hbm_awaddr(m_axi_hbm_awaddr),
       .m_axi_hbm_awburst(m_axi_hbm_awburst),
       .m_axi_hbm_awcache(m_axi_hbm_awcache),
       .m_axi_hbm_awid(m_axi_hbm_awid),
       .m_axi_hbm_awlen(m_axi_hbm_awlen),
       .m_axi_hbm_awlock(m_axi_hbm_awlock),
       .m_axi_hbm_awprot(m_axi_hbm_awprot),
       .m_axi_hbm_awsize(m_axi_hbm_awsize),
       .m_axi_hbm_awready(m_axi_hbm_awready),
       .m_axi_hbm_awvalid(m_axi_hbm_awvalid),
       .m_axi_hbm_rdata(m_axi_hbm_rdata),
       .m_axi_hbm_rid(m_axi_hbm_rid),
       .m_axi_hbm_rlast(m_axi_hbm_rlast),
       .m_axi_hbm_rresp(m_axi_hbm_rresp),
       .m_axi_hbm_rready(m_axi_hbm_rready),
       .m_axi_hbm_rvalid(m_axi_hbm_rvalid),
       .m_axi_hbm_wdata(m_axi_hbm_wdata),
       .m_axi_hbm_wlast(m_axi_hbm_wlast),
       .m_axi_hbm_wstrb(m_axi_hbm_wstrb),
       .m_axi_hbm_wready(m_axi_hbm_wready),
       .m_axi_hbm_wvalid(m_axi_hbm_wvalid),
       .m_axi_hbm_bid(m_axi_hbm_bid),
       .m_axi_hbm_bresp(m_axi_hbm_bresp),
       .m_axi_hbm_bready(m_axi_hbm_bready),
       .m_axi_hbm_bvalid(m_axi_hbm_bvalid),

       // AXI4S master interface for core_in
       .m_axis_core_in_tdata(m_axis_core_in_tdata),
       .m_axis_core_in_tvalid(m_axis_core_in_tvalid),
       .m_axis_core_in_tready(m_axis_core_in_tready),

       // AXI4S slave interface for core_out
       .s_axis_core_out_tdata(s_axis_core_out_tdata),
       .s_axis_core_out_tvalid(s_axis_core_out_tvalid),
       .s_axis_core_out_tready(s_axis_core_out_tready),

       // AXI4S master interface for core_in_fw_idx
       .m_axis_core_in_fw_idx_tdata(m_axis_core_in_fw_idx_tdata),
       .m_axis_core_in_fw_idx_tvalid(m_axis_core_in_fw_idx_tvalid),
       .m_axis_core_in_fw_idx_tready(m_axis_core_in_fw_idx_tready),

        // AXI4S slave interface for core_out_fw_idx
        .s_axis_core_out_fw_idx_tdata(s_axis_core_out_fw_idx_tdata),
        .s_axis_core_out_fw_idx_tvalid(s_axis_core_out_fw_idx_tvalid),
        .s_axis_core_out_fw_idx_tready(s_axis_core_out_fw_idx_tready),

       .axis_fs_tdata(in0_V_tdata),
       .axis_fs_tvalid(in0_V_tvalid),
       .axis_fs_tready(in0_V_tready),
       .axis_se_tdata(out0_V_tdata),
       .axis_se_tvalid(out0_V_tvalid),
       .axis_se_tready(out0_V_tready),

       // control signals
       .n_layers($N_LAYERS$),
       .done_if(done_if)

    );

endmodule

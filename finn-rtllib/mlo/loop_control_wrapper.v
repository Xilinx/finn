module $MODULE_NAME_AXI_WRAPPER$ #(
    parameter N_MAX_LAYERS = $N_MAX_LAYERS$,
    parameter N_FW_CORES = $N_FW_CORES$,
    parameter ADDR_BITS = $ADDR_BITS$,
    parameter DATA_BITS = $DATA_BITS$,
    parameter LEN_BITS = $LEN_BITS$,
    parameter CNT_BITS = $CNT_BITS$,
    parameter ILEN_BITS = $ILEN_BITS$,
    parameter OLEN_BITS = $OLEN_BITS$,
    parameter ADDR_INT   = $ADDR_INT$,
    parameter LAYER_OFFS_INT = $LAYER_OFFS_INT$
) (
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
    output [DATA_BITS-1:0] core_in_tdata,
    output                 core_in_tvalid,
    input                  core_in_tready,

    // AXI4S master interface for core_in_fw_idx [N_FW_CORES]
    output [N_FW_CORES-1:0][DATA_BITS-1:0] core_in_fw_idx_tdata,
    output [N_FW_CORES-1:0]                core_in_fw_idx_tvalid,
    input  [N_FW_CORES-1:0]                core_in_fw_idx_tready,

    // AXI4S slave interface for core_out
    input  [DATA_BITS-1:0] core_out_tdata,
    input                  core_out_tvalid,
    output                 core_out_tready,

    input  logic           aclk,
    input  logic           aresetn,

    // activation signals
    input  [DATA_BITS-1:0] axis_fs_tdata,
    input                  axis_fs_tvalid,
    output                 axis_fs_tready,
    output [DATA_BITS-1:0] axis_se_tdata,
    output                 axis_se_tvalid,
    input                  axis_se_tready,

    // control signals
    input  logic [CNT_BITS-1:0] n_layers,
    output logic [1:0]         done_if,

    // AXI4S slave interface for idx_fs
    input  [DATA_BITS-1:0] idx_fs_tdata,
    input                  idx_fs_tvalid,
    output                 idx_fs_tready,
    // AXI4S master interface for idx_se
    output [DATA_BITS-1:0] idx_se_tdata,
    output                 idx_se_tvalid,
    input                  idx_se_tready
);

    loop_control #(
        .N_MAX_LAYERS(N_MAX_LAYERS),
        .N_FW_CORES(N_FW_CORES),
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),
        .ILEN_BITS(ILEN_BITS),
        .OLEN_BITS(OLEN_BITS),
        .ADDR_INT(ADDR_INT),
        .LAYER_OFFS_INT(LAYER_OFFS_INT)
    ) loop_control_inst (
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
        .core_in_tdata(core_in_tdata),
        .core_in_tvalid(core_in_tvalid),
        .core_in_tready(core_in_tready),

        // AXI4S master interface for core_in_fw_idx [N_FW_CORES]
        .core_in_fw_idx_tdata(core_in_fw_idx_tdata),
        .core_in_fw_idx_tvalid(core_in_fw_idx_tvalid),
        .core_in_fw_idx_tready(core_in_fw_idx_tready),

        // AXI4S slave interface for core_out
        .core_out_tdata(core_out_tdata),
        .core_out_tvalid(core_out_tvalid),
        .core_out_tready(core_out_tready),

        .aclk(aclk),
        .aresetn(aresetn),

        // activation signals
        .axis_fs_tdata(axis_fs_tdata),
        .axis_fs_tvalid(axis_fs_tvalid),
        .axis_fs_tready(axis_fs_tready),
        .axis_se_tdata(axis_se_tdata),
        .axis_se_tvalid(axis_se_tvalid),
        .axis_se_tready(axis_se_tready),

        // control signals
        .n_layers(n_layers),
        .done_if(done_if),

        // AXI4S slave interface for idx_fs
        .idx_fs_tdata(idx_fs_tdata),
        .idx_fs_tvalid(idx_fs_tvalid),
        .idx_fs_tready(idx_fs_tready),
        // AXI4S master interface for idx_se
        .idx_se_tdata(idx_se_tdata),
        .idx_se_tvalid(idx_se_tvalid),
        .idx_se_tready(idx_se_tready)
    );

endmodule
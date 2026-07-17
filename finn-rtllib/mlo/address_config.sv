module address_config #(
    int unsigned              BASE_ADDR_BITS = 64,
    int unsigned              AXIL_DATA_BITS = 32,
    int unsigned              AXIL_ADDR_BITS = 3
) (
    input  logic                        aclk,
    input  logic                        aresetn,

    input  logic[AXIL_ADDR_BITS-1:0]    s_axilite_awaddr,
    input  logic[2:0]                   s_axilite_awprot,
    input  logic                        s_axilite_awvalid,
    output logic                        s_axilite_awready,
    input  logic[AXIL_DATA_BITS-1:0]    s_axilite_wdata,
    input  logic[AXIL_DATA_BITS/8-1:0]  s_axilite_wstrb,
    input  logic                        s_axilite_wvalid,
    output logic                        s_axilite_wready,
    output logic[1:0]                   s_axilite_bresp,
    output logic                        s_axilite_bvalid,
    input  logic                        s_axilite_bready,
    input  logic[AXIL_ADDR_BITS-1:0]    s_axilite_araddr,
    input  logic[2:0]                   s_axilite_arprot,
    input  logic                        s_axilite_arvalid,
    output logic                        s_axilite_arready,
    output logic[AXIL_DATA_BITS-1:0]    s_axilite_rdata,
    output logic[1:0]                   s_axilite_rresp,
    output logic                        s_axilite_rvalid,
    input  logic                        s_axilite_rready,

    output logic[BASE_ADDR_BITS-1:0]    base_address
);

logic                       cfg_en;
logic                       cfg_we;
logic                       cfg_a;      // address ignored (single register)
logic[BASE_ADDR_BITS-1:0]   cfg_d;
logic                       cfg_rack;
logic[BASE_ADDR_BITS-1:0]   cfg_q;

axilite #(
    .ADDR_WIDTH(AXIL_ADDR_BITS),
    .DATA_WIDTH(AXIL_DATA_BITS),
    .IP_DATA_WIDTH(BASE_ADDR_BITS)
) inst_axilite (
    .aclk(aclk), .aresetn(aresetn),

    .awready(s_axilite_awready), .awvalid(s_axilite_awvalid), .awaddr(s_axilite_awaddr), .awprot(s_axilite_awprot),
    .wready(s_axilite_wready),   .wvalid(s_axilite_wvalid),   .wdata(s_axilite_wdata),   .wstrb(s_axilite_wstrb),
    .bready(s_axilite_bready),   .bvalid(s_axilite_bvalid),   .bresp(s_axilite_bresp),

    .arready(s_axilite_arready), .arvalid(s_axilite_arvalid), .araddr(s_axilite_araddr), .arprot(s_axilite_arprot),
    .rready(s_axilite_rready),   .rvalid(s_axilite_rvalid),   .rresp(s_axilite_rresp),   .rdata(s_axilite_rdata),

    .ip_en(cfg_en), .ip_wen(cfg_we), .ip_addr(cfg_a), .ip_wdata(cfg_d),
    .ip_rack(cfg_rack), .ip_rdata(cfg_q)
);

// Base address holding register
logic[BASE_ADDR_BITS-1:0] base_reg = '0;
always_ff @(posedge aclk) begin
    if(!aresetn) begin
        base_reg <= '0;
    end
    else if(cfg_en && cfg_we) begin
        base_reg <= cfg_d;
    end
end

// Read-back acknowledge: reply one cycle after a read access
logic cfg_rack_reg = 1'b0;
always_ff @(posedge aclk) begin
    if(!aresetn) begin
        cfg_rack_reg <= 1'b0;
    end
    else begin
        cfg_rack_reg <= cfg_en && !cfg_we;
    end
end

assign cfg_q    = base_reg;
assign cfg_rack = cfg_rack_reg;
assign base_address = base_reg;

endmodule

module $MODULE_NAME_AXI_WRAPPER$ #(
    parameter N_MAX_LAYERS = $N_MAX_LAYERS$, // Maximum number of layers in the FINN pipeline
    parameter N_FW_CORES = $N_FW_CORES$, // Number of FETCH_WEIGHTS cores in the FINN pipeline
    parameter ADDR_BITS = $ADDR_BITS$, // Address bits for 
    parameter DATA_BITS = $DATA_BITS$, // Data bits for AXI4
    parameter LEN_BITS = $LEN_BITS$, // Length bits for AXI4
    parameter CNT_BITS = $CNT_BITS$, // Counter bits for AXI4S
    parameter ILEN_BITS = $ILEN_BITS$, // Length bits for AXI4S input
    parameter OLEN_BITS = $OLEN_BITS$, // Length bits for AXI4S output
    parameter ADDR_INT   = $ADDR_INT$, // Start address for intermediate frames
    parameter LAYER_OFFS_INT = $LAYER_OFFS_INT$
) (
    
)

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
        AXI4.master                 m_axi_hbm,

        AXI4S.master                core_in,
        AXI4S.master                core_in_fw_idx [N_FW_CORES],
        AXI4S.slave                 core_out,
    
        input  logic                aclk,
        input  logic                aresetn
    
        // activation signals
        AXI4S.slave                axis_fs;
        AXI4S.master               axis_se;
        
        // control signals
        input  logic [CNT_BITS-1:0] n_layers;
        output logic [1:0]         done_if;
        
        AXI4S.slave                idx_fs;
        AXI4S.master               idx_se;
    );

endmodule 
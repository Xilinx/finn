    // Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
    //
    // This file is subject to the Xilinx Design License Agreement located
    // in the LICENSE.md file in the root directory of this repository.
    //
    // This file contains confidential and proprietary information of Xilinx, Inc.
    // and is protected under U.S. and international copyright and other
    // intellectual property laws.
    //
    // DISCLAIMER
    // This disclaimer is not a license and does not grant any rights to the materials
    // distributed herewith. Except as otherwise provided in a valid license issued to
    // you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
    // MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
    // DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
    // INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
    // FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
    // in contract or tort, including negligence, or under any other theory of
    // liability) for any loss or damage of any kind or nature related to, arising
    // under or in connection with these materials, including for any direct, or any
    // indirect, special, incidental, or consequential loss or damage (including loss
    // of data, profits, goodwill, or any type of loss or damage suffered as a result
    // of any action brought by a third party) even if such damage or loss was
    // reasonably foreseeable or Xilinx had been advised of the possibility of the
    // same.
    //
    // CRITICAL APPLICATIONS
    // Xilinx products are not designed or intended to be fail-safe, or for use in
    // any application requiring failsafe performance, such as life-support or safety
    // devices or systems, Class III medical devices, nuclear facilities, applications
    // related to the deployment of airbags, or any other applications that could lead
    // to death, personal injury, or severe property or environmental damage
    // (individually and collectively, "Critical Applications"). Customer assumes the
    // sole risk and liability of any use of Xilinx products in Critical Applications,
    // subject only to applicable laws and regulations governing limitations on product
    // liability.
    //
    // THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

    import iwTypes::*;

    `include "axi_macros.svh"

    module loop_control (
        AXI4L.slave                 s_axi_ctrl,

        AXI4.master                 m_axi_hbm [N_HBM_PORTS],

        AXI4SF.slave                s_axis_h2c,
        AXI4SF.master               m_axis_c2h,

        AXI4S.master                core_in,
        AXI4S.slave                 core_out,
    
        input  logic                aclk,
        input  logic                aclk_dp,
        input  logic                aresetn
    );

    `AXISF_TIE_OFF_S(s_axis_h2c)
    `AXISF_TIE_OFF_M(m_axis_c2h)

    // ================-----------------------------------------------------------------
    // Params
    // ================-----------------------------------------------------------------
    localparam int unsigned N_DMA_PORTS = 3;
    localparam int unsigned N_FW_CORES = N_HBM_PORTS - N_DMA_PORTS;
    localparam int PE_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 8, 8, 8, 8, 16, 16 }; 

    localparam int SIMD_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 12, 12, 12, 12, 24, 24 }; 

    localparam int MW_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 384, 384, 384, 384, 1536, 384 }; 

    localparam int MH_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 384, 384, 384, 384, 384, 1536 }; 

    localparam logic[ADDR_BITS-1:0] WADDR_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 64'h4180000000, 64'h4200000000, 64'h4280000000, 64'h4300000000, 64'h4380000000, 64'h4400000000 }; 

    localparam logic[ADDR_BITS-1:0] WOFFS_ARRAY[N_HBM_PORTS-N_DMA_PORTS] = '{ 64'h100000, 64'h100000, 64'h100000, 64'h100000, 64'h100000, 64'h100000 }; 


    localparam logic[ADDR_BITS-1:0] ADDR_SRC = 64'h4000000000;
    localparam logic[ADDR_BITS-1:0] ADDR_DST = 64'h4080000000;
    localparam logic[ADDR_BITS-1:0] ADDR_INT = 64'h4100000000;
    localparam logic[ADDR_BITS-1:0] LAYER_OFFS_INT = 64'h10000;
    localparam integer N_REPS = 128;
    
    localparam int unsigned ACTIVATION_WIDTH = 8;

    // ================-----------------------------------------------------------------
    // CTRL
    // ================-----------------------------------------------------------------

    AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS+CNT_BITS+LEN_BITS)) f_ctrl_fs ();
    AXI4S #(.AXI4S_DATA_BITS(ADDR_BITS)) f_ctrl_se ();
    logic done;
    logic [1:0] done_if;
    logic [N_FW_CORES-1:0] done_w;
    logic [CNT_BITS-1:0] n_layers;

    // Slave
    axil_iw_slv_mlo #(
        .LEN_BITS(LEN_BITS),
        .ADDR_BITS(ADDR_BITS),
        .CNT_BITS(CNT_BITS),
        .N_FW_CORES(N_FW_CORES)
    ) inst_axil_iw_core_slv_mm (
        .aclk(aclk),
        .aresetn(aresetn),

        .axi_ctrl(s_axi_ctrl),

        .n_layers(n_layers),

        .f_ctrl_fs(f_ctrl_fs),
        .f_ctrl_se(f_ctrl_se),
        .s_done(done),
        .s_done_if(done_if),
        .s_done_w(done_w)
    );

    // ================-----------------------------------------------------------------
    // Intermediate frames
    // ================-----------------------------------------------------------------
    
    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_if_in ();
    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_if_out ();
    AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) axis_if_in ();
    AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) axis_if_out ();

    intermediate_frames #(
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),

        .ADDR_INT(ADDR_INT),
        .LAYER_OFFS_INT(LAYER_OFFS_INT),
        .N_MAX_LAYERS(N_MAX_LAYERS)
    ) inst_intermediate_frames (
        .aclk(aclk),
        .aresetn(aresetn),

        .m_done(done_if),

        .m_axi_hbm(m_axi_hbm[2]),

        .s_idx(idx_if_in),
        .m_idx(idx_if_out),

        .s_axis(axis_if_in),
        .m_axis(axis_if_out)
    );

    // ================-----------------------------------------------------------------
    // Fetch start
    // ================-----------------------------------------------------------------

    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_fs ();
    AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) axis_fs ();

    fetch_start #(
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),

        .ILEN_BITS(ILEN_BITS),
        .ADDR_SRC(ADDR_SRC)
    ) inst_fetch_start (
        .aclk(aclk),
        .aresetn(aresetn),
        .m_axi_hbm(m_axi_hbm[0]),
        .s_ctrl(f_ctrl_fs),
        .m_idx(idx_fs),
        .m_axis(axis_fs)
    );

    // ================-----------------------------------------------------------------
    // Mux in
    // ================-----------------------------------------------------------------

    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS)) idx_fw [N_FW_CORES] ();
    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_out ();

    mux_in #(
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),

        .ILEN_BITS(ILEN_BITS),
        .N_FW_CORES(N_FW_CORES)
    ) inst_mux_in (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_idx_fs(idx_fs),
        .s_idx_if(idx_if_out),
        .m_idx_fw(idx_fw),
        .m_idx_out(idx_out),

        .s_axis_fs(axis_fs),
        .s_axis_if(axis_if_out),
        .m_axis(core_in)
    );

    // ================-----------------------------------------------------------------
    // Mux out
    // ================-----------------------------------------------------------------

    AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_se ();
    AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) axis_se ();

    mux_out #(
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),

        .OLEN_BITS(OLEN_BITS)
    ) inst_mux_out (
        .aclk(aclk),
        .aresetn(aresetn),

        .n_layers(n_layers),

        .s_idx(idx_out),
        .m_idx_se(idx_se),
        .m_idx_if(idx_if_in),

        .s_axis(core_out),
        .m_axis_se(axis_se),
        .m_axis_if(axis_if_in)
    );

    // ================-----------------------------------------------------------------
    // Store end
    // ================-----------------------------------------------------------------

    store_end #(
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .CNT_BITS(CNT_BITS),

        .OLEN_BITS(OLEN_BITS),
        .ADDR_DST(ADDR_DST)
    ) inst_store_end (
        .aclk(aclk),
        .aresetn(aresetn),
        .m_axi_hbm(m_axi_hbm[1]),
        .s_ctrl(f_ctrl_se),
        .s_idx(idx_se),
        .m_done(done),
        .s_axis(axis_se)
    );


    // ================-----------------------------------------------------------------
    // Weights cores
    // ================-----------------------------------------------------------------

    
    
    endmodule
    
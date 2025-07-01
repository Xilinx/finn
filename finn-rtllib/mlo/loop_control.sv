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

    `include "axi_macros.svh"

    module loop_control #(
        parameter int unsigned N_MAX_LAYERS = 16, // Maximum number of layers in the FINN pipeline
        parameter int unsigned ADDR_BITS = 64, // Address bits for
        parameter int unsigned DATA_BITS = 512, // Data bits for AXI4
        parameter int unsigned LEN_BITS = 32, // Length bits for AXI4
        parameter int unsigned CNT_BITS = 16, // Counter bits for AXI4S
        parameter int unsigned ILEN_BITS = 16, // Length bits for AXI4S input
        parameter int unsigned OLEN_BITS = 16, // Length bits for AXI4S output
        parameter int unsigned M_AXI_HBM_BASE_ADDR   = 64'h4100000000, // Start address for intermediate frames
        parameter int unsigned LAYER_OFFS_INT = 64'h10000
    ) (
        input  logic                aclk,
        input  logic                aresetn,

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

        // AXI4S slave interface for core_out
        input  [DATA_BITS-1:0] s_axis_core_out_tdata,
        input                  s_axis_core_out_tvalid,
        output                 s_axis_core_out_tready,

        // AXI4S master interface for core_in_fw_idx
        output [DATA_BITS-1:0] m_axis_core_in_fw_idx_tdata,
        output                 m_axis_core_in_fw_idx_tvalid,
        input                  m_axis_core_in_fw_idx_tready,

        // activation signals
        input  [DATA_BITS-1:0] axis_fs_tdata,
        input                  axis_fs_tvalid,
        output                 axis_fs_tready,
        output [DATA_BITS-1:0] axis_se_tdata,
        output                 axis_se_tvalid,
        input                  axis_se_tready,

        // control signals
        input  wire [CNT_BITS-1:0] n_layers,
        output wire [1:0]          done_if

    );


    AXI4  #(.AXI4_ADDR_BITS(ADDR_BITS), .AXI4_DATA_BITS(DATA_BITS)) m_axi_hbm_if();
    AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) core_in ();
    AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) core_out ();
    AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) core_in_fw_idx ();
    AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) axis_fs_if ();
    AXI4S #(.AXI4S_DATA_BITS(DATA_BITS)) axis_se_if ();

    `AXI_ASSIGN_I2S(m_axi_hbm_if, m_axi_hbm)
    `AXIS_ASSIGN_I2S(core_in, m_axis_core_in)
    `AXIS_ASSIGN_S2I(s_axis_core_out,  core_out)

    `AXIS_ASSIGN_I2S(core_in_fw_idx, m_axis_core_in_fw_idx)

    `AXIS_ASSIGN_I2S(axis_fs_if, axis_fs)
    `AXIS_ASSIGN_S2I(axis_se, axis_se_if)

  //  ================-----------------------------------------------------------------
  //  Intermediate frames
  //  ================-----------------------------------------------------------------

   AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_if_in ();
   AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_if_out ();
   AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) axis_if_in ();
   AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) axis_if_out ();

   intermediate_frames #(
       .ADDR_BITS(ADDR_BITS),
       .DATA_BITS(DATA_BITS),
       .LEN_BITS(LEN_BITS),
       .CNT_BITS(CNT_BITS),
       .ILEN_BITS(ILEN_BITS),
       .OLEN_BITS(OLEN_BITS),
       .ADDR_INT(M_AXI_HBM_BASE_ADDR),
       .LAYER_OFFS_INT(LAYER_OFFS_INT),
       .N_MAX_LAYERS(N_MAX_LAYERS)
   ) inst_intermediate_frames (
       .aclk(aclk),
       .aresetn(aresetn),

       .m_done(done_if),

       .m_axi_hbm(m_axi_hbm_if),

       .s_idx(idx_if_in),
       .m_idx(idx_if_out),

       .s_axis(axis_if_in),
       .m_axis(axis_if_out)
   );

   // ================-----------------------------------------------------------------
   // Mux in
   // ================-----------------------------------------------------------------

   AXI4S #(.AXI4S_DATA_BITS(2*CNT_BITS+LEN_BITS)) idx_out ();

   mux_in #(
       .ADDR_BITS(ADDR_BITS),
       .DATA_BITS(DATA_BITS),
       .LEN_BITS(LEN_BITS),
       .CNT_BITS(CNT_BITS),

       .ILEN_BITS(ILEN_BITS)
   ) inst_mux_in (
       .aclk(aclk),
       .aresetn(aresetn),
       .s_idx_if(idx_if_out),
       .m_idx_fw(core_in_fw_idx),
       .m_idx_out(idx_out),

       .s_axis_fs(axis_fs_if),
       .s_axis_if(axis_if_out),
       .m_axis(core_in)
   );

   // ================-----------------------------------------------------------------
   // Mux out
   // ================-----------------------------------------------------------------

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
       .m_idx_if(idx_if_in),

       .s_axis(core_out),
       .m_axis_se(axis_se_if),
       .m_axis_if(axis_if_in)
   );

endmodule

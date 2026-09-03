/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

module $LOOP_CONTROL_WRAPPER_NAME$ #(
    parameter N_MAX_LAYERS   = $N_MAX_LAYERS$,
    parameter INPUT_BYTES    = $INPUT_BYTES$, // number of bytes in the input shape
    parameter N_LAYERS       = $N_LAYERS$,

    parameter ADDR_BITS      = 64,
    parameter DATA_BITS      = 256,
    parameter LEN_BITS       = 32,
    parameter IDX_BITS       = 16,
    parameter ELEM_BITS      = $ELEM_BITS$,
    parameter ILEN_BITS      = $ILEN_BITS$,
    parameter OLEN_BITS      = $OLEN_BITS$,

    // Base address offset (added to base_address)
    parameter [ADDR_BITS-1:0] ADDRESS_OFFSET = $ADDRESS_OFFSET$
) (
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axi_intermediate_frame:m_axis_core_in:m_axis_core_in_fw_idx:s_axis_core_out:in0_V:out0_V:s_axis_core_out_fw_idx, ASSOCIATED_RESET = ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input   ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input   ap_rst_n,

    // AXI4 master interface for m_axi_intermediate_frame
    output [ADDR_BITS-1:0] m_axi_intermediate_frame_araddr,
    output [1:0]           m_axi_intermediate_frame_arburst,
    output [3:0]           m_axi_intermediate_frame_arcache,
    output [1:0]           m_axi_intermediate_frame_arid,
    output [7:0]           m_axi_intermediate_frame_arlen,
    output                 m_axi_intermediate_frame_arlock,
    output [2:0]           m_axi_intermediate_frame_arprot,
    output [2:0]           m_axi_intermediate_frame_arsize,
    input                  m_axi_intermediate_frame_arready,
    output                 m_axi_intermediate_frame_arvalid,
    output [ADDR_BITS-1:0] m_axi_intermediate_frame_awaddr,
    output [1:0]           m_axi_intermediate_frame_awburst,
    output [3:0]           m_axi_intermediate_frame_awcache,
    output [1:0]           m_axi_intermediate_frame_awid,
    output [7:0]           m_axi_intermediate_frame_awlen,
    output                 m_axi_intermediate_frame_awlock,
    output [2:0]           m_axi_intermediate_frame_awprot,
    output [2:0]           m_axi_intermediate_frame_awsize,
    input                  m_axi_intermediate_frame_awready,
    output                 m_axi_intermediate_frame_awvalid,
    input  [DATA_BITS-1:0] m_axi_intermediate_frame_rdata,
    input  [1:0]           m_axi_intermediate_frame_rid,
    input                  m_axi_intermediate_frame_rlast,
    input  [1:0]           m_axi_intermediate_frame_rresp,
    output                 m_axi_intermediate_frame_rready,
    input                  m_axi_intermediate_frame_rvalid,
    output [DATA_BITS-1:0] m_axi_intermediate_frame_wdata,
    output                 m_axi_intermediate_frame_wlast,
    output [DATA_BITS/8-1:0] m_axi_intermediate_frame_wstrb,
    input                  m_axi_intermediate_frame_wready,
    output                 m_axi_intermediate_frame_wvalid,
    input  [1:0]           m_axi_intermediate_frame_bid,
    input  [1:0]           m_axi_intermediate_frame_bresp,
    output                 m_axi_intermediate_frame_bready,
    input                  m_axi_intermediate_frame_bvalid,

    // AXI4S master interface for core_in
    output [ILEN_BITS-1:0] m_axis_core_in_tdata,
    output                 m_axis_core_in_tvalid,
    input                  m_axis_core_in_tready,

    // AXI4S master interface for core_in_fw_idx
    output [IDX_BITS-1:0]  m_axis_core_in_fw_idx_tdata,
    output                 m_axis_core_in_fw_idx_tvalid,
    input                  m_axis_core_in_fw_idx_tready,

    // AXI4S slave interface for core_out
    input  [OLEN_BITS-1:0] s_axis_core_out_tdata,
    input                  s_axis_core_out_tvalid,
    output                 s_axis_core_out_tready,

    // AXI4S slave interface for core_out_fw_idx
    input  [IDX_BITS-1:0] s_axis_core_out_fw_idx_tdata,
    input                  s_axis_core_out_fw_idx_tvalid,
    output                 s_axis_core_out_fw_idx_tready,

    // Activation signals
    input  [ILEN_BITS-1:0] in0_V_tdata,
    input                  in0_V_tvalid,
    output                 in0_V_tready,

    output [OLEN_BITS-1:0] out0_V_tdata,
    output                 out0_V_tvalid,
    input                  out0_V_tready,

    // Control signals
    output wire [1:0]      done_if
`ifdef HAS_BASE_ADDRESS
    ,
    // Base Address
    input wire [ADDR_BITS-1:0] base_address
`endif
);

    loop_control #(
        .N_LAYERS(N_LAYERS),
        .FM_SIZE(INPUT_BYTES),
        .ADDR_BITS(ADDR_BITS),
        .DATA_BITS(DATA_BITS),
        .LEN_BITS(LEN_BITS),
        .IDX_BITS(IDX_BITS),
        .ELEM_BITS(ELEM_BITS),
        .ILEN_BITS(ILEN_BITS),
        .OLEN_BITS(OLEN_BITS),
        .ADDRESS_OFFSET(ADDRESS_OFFSET)
    ) loop_control_inst (
       .aclk(ap_clk),
       .aresetn(ap_rst_n),

       // AXI4 master interface for m_axi_intermediate_frame
       .m_axi_intermediate_frame_araddr(m_axi_intermediate_frame_araddr),
       .m_axi_intermediate_frame_arburst(m_axi_intermediate_frame_arburst),
       .m_axi_intermediate_frame_arcache(m_axi_intermediate_frame_arcache),
       .m_axi_intermediate_frame_arid(m_axi_intermediate_frame_arid),
       .m_axi_intermediate_frame_arlen(m_axi_intermediate_frame_arlen),
       .m_axi_intermediate_frame_arlock(m_axi_intermediate_frame_arlock),
       .m_axi_intermediate_frame_arprot(m_axi_intermediate_frame_arprot),
       .m_axi_intermediate_frame_arsize(m_axi_intermediate_frame_arsize),
       .m_axi_intermediate_frame_arready(m_axi_intermediate_frame_arready),
       .m_axi_intermediate_frame_arvalid(m_axi_intermediate_frame_arvalid),
       .m_axi_intermediate_frame_awaddr(m_axi_intermediate_frame_awaddr),
       .m_axi_intermediate_frame_awburst(m_axi_intermediate_frame_awburst),
       .m_axi_intermediate_frame_awcache(m_axi_intermediate_frame_awcache),
       .m_axi_intermediate_frame_awid(m_axi_intermediate_frame_awid),
       .m_axi_intermediate_frame_awlen(m_axi_intermediate_frame_awlen),
       .m_axi_intermediate_frame_awlock(m_axi_intermediate_frame_awlock),
       .m_axi_intermediate_frame_awprot(m_axi_intermediate_frame_awprot),
       .m_axi_intermediate_frame_awsize(m_axi_intermediate_frame_awsize),
       .m_axi_intermediate_frame_awready(m_axi_intermediate_frame_awready),
       .m_axi_intermediate_frame_awvalid(m_axi_intermediate_frame_awvalid),
       .m_axi_intermediate_frame_rdata(m_axi_intermediate_frame_rdata),
       .m_axi_intermediate_frame_rid(m_axi_intermediate_frame_rid),
       .m_axi_intermediate_frame_rlast(m_axi_intermediate_frame_rlast),
       .m_axi_intermediate_frame_rresp(m_axi_intermediate_frame_rresp),
       .m_axi_intermediate_frame_rready(m_axi_intermediate_frame_rready),
       .m_axi_intermediate_frame_rvalid(m_axi_intermediate_frame_rvalid),
       .m_axi_intermediate_frame_wdata(m_axi_intermediate_frame_wdata),
       .m_axi_intermediate_frame_wlast(m_axi_intermediate_frame_wlast),
       .m_axi_intermediate_frame_wstrb(m_axi_intermediate_frame_wstrb),
       .m_axi_intermediate_frame_wready(m_axi_intermediate_frame_wready),
       .m_axi_intermediate_frame_wvalid(m_axi_intermediate_frame_wvalid),
       .m_axi_intermediate_frame_bid(m_axi_intermediate_frame_bid),
       .m_axi_intermediate_frame_bresp(m_axi_intermediate_frame_bresp),
       .m_axi_intermediate_frame_bready(m_axi_intermediate_frame_bready),
       .m_axi_intermediate_frame_bvalid(m_axi_intermediate_frame_bvalid),

       // AXI4S master interface for core_in
       .m_axis_core_tdata(m_axis_core_in_tdata),
       .m_axis_core_tvalid(m_axis_core_in_tvalid),
       .m_axis_core_tready(m_axis_core_in_tready),

       // AXI4S slave interface for core_out
       .s_axis_core_tdata(s_axis_core_out_tdata),
       .s_axis_core_tvalid(s_axis_core_out_tvalid),
       .s_axis_core_tready(s_axis_core_out_tready),

       // AXI4S master interface for core_in_fw_idx
       .m_idx_tdata(m_axis_core_in_fw_idx_tdata),
       .m_idx_tvalid(m_axis_core_in_fw_idx_tvalid),
       .m_idx_tready(m_axis_core_in_fw_idx_tready),

       // AXI4S slave interface for core_out_fw_idx
       .s_idx_tdata(s_axis_core_out_fw_idx_tdata),
       .s_idx_tvalid(s_axis_core_out_fw_idx_tvalid),
       .s_idx_tready(s_axis_core_out_fw_idx_tready),

       // Activation signals
       .s_axis_fs_tdata(in0_V_tdata),
       .s_axis_fs_tvalid(in0_V_tvalid),
       .s_axis_fs_tready(in0_V_tready),

       .m_axis_se_tdata(out0_V_tdata),
       .m_axis_se_tvalid(out0_V_tvalid),
       .m_axis_se_tready(out0_V_tready),

`ifdef HAS_BASE_ADDRESS
       .base_address(base_address)
`else
       .base_address({ADDR_BITS{1'b0}})
`endif
    );

endmodule

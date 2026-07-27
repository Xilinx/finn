// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
/****************************************************************************
 * @brief   Verilog wrapper for IP packaging (decoupled requant).
 *
 * Directly instantiates the SystemVerilog requant_axi_decoupled core. A plain
 * Verilog top module is required because Vivado IP packaging does not allow a
 * SystemVerilog top module. Unlike the embedded variant, no intermediate
 * SystemVerilog wrapper is needed: all parameters are plain integers that
 * Verilog can pass straight through to the SystemVerilog core.
 ***************************************************************************/

module $TOP_MODULE_NAME$ (
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:s_scale_V:s_bias_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  ap_rst_n,

    //- AXI Stream - Data Input ---------
    output  in0_V_TREADY,
    input   in0_V_TVALID,
    input  [$IN_STREAM_WIDTH$-1:0]  in0_V_TDATA,

    //- AXI Stream - Scale Param Input --
    output  s_scale_V_TREADY,
    input   s_scale_V_TVALID,
    input  [$SCALE_STREAM_WIDTH$-1:0]  s_scale_V_TDATA,

    //- AXI Stream - Bias Param Input ---
    output  s_bias_V_TREADY,
    input   s_bias_V_TVALID,
    input  [$BIAS_STREAM_WIDTH$-1:0]  s_bias_V_TDATA,

    //- AXI Stream - Output -------------
    input   out0_V_TREADY,
    output  out0_V_TVALID,
    output [$OUT_STREAM_WIDTH$-1:0]  out0_V_TDATA
);

    requant_axi_decoupled #(
        .VERSION($VERSION$),
        .K($K$), .N($N$), .C($C$), .PE($PE$),
        .TAP_MIN($TAP_MIN$), .TAP_MAX($TAP_MAX$)
    ) core (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_tready(in0_V_TREADY),
        .s_axis_tvalid(in0_V_TVALID),
        .s_axis_tdata(in0_V_TDATA),
        .s_scale_tready(s_scale_V_TREADY),
        .s_scale_tvalid(s_scale_V_TVALID),
        .s_scale_tdata(s_scale_V_TDATA),
        .s_bias_tready(s_bias_V_TREADY),
        .s_bias_tvalid(s_bias_V_TVALID),
        .s_bias_tdata(s_bias_V_TDATA),
        .m_axis_tready(out0_V_TREADY),
        .m_axis_tvalid(out0_V_TVALID),
        .m_axis_tdata(out0_V_TDATA)
    );

endmodule

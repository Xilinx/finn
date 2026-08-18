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
 *
 * Primary parameters are substituted by Python codegen; stream widths are
 * derived as localparams to keep the template in sync with the SV modules.
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
    parameter VERSION = $VERSION$,
    parameter K       = $K$,
    parameter N       = $N$,
    parameter C       = $C$,
    parameter PE      = $PE$,
    parameter TAP_MIN = $TAP_MIN$,
    parameter TAP_MAX = $TAP_MAX$,
    parameter SIGNED_OUT = $SIGNED_OUT$,

    // Derived widths (matching requant_axi_decoupled.sv localparam chain)
    parameter S_WIDTH = (K <= (VERSION == 3 ? 24 : 18)) ? 25 : (VERSION == 3 ? 24 : 18),
    parameter X_WIDTH = (K <= (VERSION == 3 ? 24 : 18)) ? K  : ((VERSION == 1 ? 25 : 27) < K ? (VERSION == 1 ? 25 : 27) : K),
    parameter BIAS_WIDTH  = S_WIDTH + X_WIDTH,
    parameter TAP_RANGE = TAP_MAX - TAP_MIN + 1,
    parameter TAP_WIDTH  = (TAP_RANGE > 1) ? $clog2(TAP_RANGE) : 1,
    parameter PARAMS_LANE_WIDTH = S_WIDTH + TAP_WIDTH + BIAS_WIDTH,
    parameter INPUT_STREAM_WIDTH  = ((PE * K + 7) / 8) * 8,
    parameter OUTPUT_STREAM_WIDTH = ((PE * N + 7) / 8) * 8,
    parameter PARAMS_STREAM_WIDTH = ((PE * PARAMS_LANE_WIDTH + 7) / 8) * 8
)(
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:s_params_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  ap_rst_n,

    //- AXI Stream - Data Input ---------
    output  in0_V_TREADY,
    input   in0_V_TVALID,
    input  [INPUT_STREAM_WIDTH-1:0]  in0_V_TDATA,

    //- AXI Stream - Params Input -------
    output  s_params_V_TREADY,
    input   s_params_V_TVALID,
    input  [PARAMS_STREAM_WIDTH-1:0]  s_params_V_TDATA,

    //- AXI Stream - Output -------------
    input   out0_V_TREADY,
    output  out0_V_TVALID,
    output [OUTPUT_STREAM_WIDTH-1:0]  out0_V_TDATA
);

    requant_axi_decoupled #(
        .VERSION(VERSION),
        .K(K), .N(N), .C(C), .PE(PE),
        .TAP_MIN(TAP_MIN), .TAP_MAX(TAP_MAX),
        .SIGNED_OUT(SIGNED_OUT)
    ) core (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_tready(in0_V_TREADY),
        .s_axis_tvalid(in0_V_TVALID),
        .s_axis_tdata(in0_V_TDATA),
        .s_params_tready(s_params_V_TREADY),
        .s_params_tvalid(s_params_V_TVALID),
        .s_params_tdata(s_params_V_TDATA),
        .m_axis_tready(out0_V_TREADY),
        .m_axis_tvalid(out0_V_TVALID),
        .m_axis_tdata(out0_V_TDATA)
    );

endmodule

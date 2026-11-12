// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
/****************************************************************************
 * @brief   Verilog wrapper for IP packaging (decoupled FP32 requant).
 *
 * Directly instantiates the SystemVerilog requantf_axi_decoupled core. A plain
 * Verilog top module is required because Vivado IP packaging does not allow a
 * SystemVerilog top module. Unlike the embedded variant, no intermediate
 * SystemVerilog wrapper is needed: all parameters are plain integers that
 * Verilog can pass straight through to the SystemVerilog core.
 *
 * Primary parameters are substituted by Python codegen; stream widths are
 * derived as localparams to keep the template in sync with the SV modules.
 * The parameter stream carries one FP32 SCALE/BIAS pair per lane (PE*64 bits);
 * the feeder must pre-adjust the bias (round/sign) before streaming.
 ***************************************************************************/

module $TOP_MODULE_NAME$ #(
    parameter N  = $N$,
    parameter C  = $C$,
    parameter PE = $PE$,
    parameter SIGNED_OUT = $SIGNED_OUT$,

    // Derived widths (matching requantf_axi_decoupled.sv localparam chain)
    parameter INPUT_STREAM_WIDTH  = PE * 32,
    parameter OUTPUT_STREAM_WIDTH = ((PE * N + 7) / 8) * 8,
    parameter PARAMS_STREAM_WIDTH = PE * 64
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

    requantf_axi_decoupled #(
        .N(N), .C(C), .PE(PE),
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

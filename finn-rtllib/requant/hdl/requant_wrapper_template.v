/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief   Verilog stub wrapper for IP packaging.
 *
 * This is a simple Verilog wrapper that instantiates the SystemVerilog
 * implementation module. Required because Vivado IP packaging doesn't
 * allow SystemVerilog as the top module.
 ***************************************************************************/

module $TOP_MODULE_NAME$ (
    //- Global Control ------------------
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  ap_rst_n,

    //- AXI Stream - Input --------------
    output  in0_V_TREADY,
    input   in0_V_TVALID,
    input  [$IN_STREAM_WIDTH$-1:0]  in0_V_TDATA,

    //- AXI Stream - Output -------------
    input   out0_V_TREADY,
    output  out0_V_TVALID,
    output [$OUT_STREAM_WIDTH$-1:0]  out0_V_TDATA
);

    $TOP_MODULE_NAME$_impl impl (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .in0_V_TREADY(in0_V_TREADY),
        .in0_V_TVALID(in0_V_TVALID),
        .in0_V_TDATA(in0_V_TDATA),
        .out0_V_TREADY(out0_V_TREADY),
        .out0_V_TVALID(out0_V_TVALID),
        .out0_V_TDATA(out0_V_TDATA)
    );

endmodule

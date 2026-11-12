/******************************************************************************
 * Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
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
 *****************************************************************************/

module $TOP_MODULE_NAME$ #(
	// Width of the fill-count ports; wide enough to carry fifo_gauge's fixed
	// 32-bit occupancy counter out to FIFO sizing without truncation. fifo.sv's
	// own (narrower) counter zero-extends into these ports.
	parameter COUNT_WIDTH = 32
)(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

output [COUNT_WIDTH-1:0] count,
output [COUNT_WIDTH-1:0] maxcount,

//- AXI Stream - Input --------------
output   in0_V_TREADY,
input   in0_V_TVALID,
input  $IN_RANGE$ in0_V_TDATA,

//- AXI Stream - Output --------------
input   out0_V_TREADY,
output   out0_V_TVALID,
output  $OUT_RANGE$ out0_V_TDATA
);

`ifdef FINN_SIMULATION
	fifo_gauge #(.WIDTH($WIDTH$), .DATA_LOGFILE("$DATA_LOGFILE$"), .LOG_VERBOSE($LOG_VERBOSE$), .LOG_FLUSH_CYCLES($LOG_FLUSH_CYCLES$)) fifo (
		.clk(ap_clk), .rst(!ap_rst_n),
		.idat(in0_V_TDATA), .ivld(in0_V_TVALID), .irdy(in0_V_TREADY),
		.odat(out0_V_TDATA), .ovld(out0_V_TVALID), .ordy(out0_V_TREADY),
		.count(count), .maxcount(maxcount)
	);
`else
	// Store only the $REAL_WIDTH$ payload bits and zero the AXI byte padding on the
	// output. Leaving the constant padding to Vivado's trimming breaks URAM-backed
	// FIFOs in Vivado 2025.1/2025.2: the URAM read pipeline absorption is planned
	// for the untrimmed width, and the trimmed netlist delivers data ahead of valid.
	wire [$REAL_WIDTH$-1:0]  fifo_odat;
	fifo #(.DEPTH($DEPTH$), .DATA_WIDTH($REAL_WIDTH$), .RAM_STYLE("$RAM_STYLE$")) fifo (
		.clk(ap_clk), .rst(!ap_rst_n),
		.idat(in0_V_TDATA[$REAL_WIDTH$-1:0]), .ivld(in0_V_TVALID), .irdy(in0_V_TREADY),
		.odat(fifo_odat), .ovld(out0_V_TVALID), .ordy(out0_V_TREADY),
		.count(count), .maxcount(maxcount)
	);
	assign	out0_V_TDATA = fifo_odat;	// zero-extends into the padding
`endif

endmodule

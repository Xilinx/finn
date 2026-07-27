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

module mvu_dyn_axi #(
    bit IS_MVU = 1,
    parameter COMPUTE_CORE,
    int unsigned MH,
    int unsigned MW,
    int unsigned PE,
	int unsigned SIMD,
    int unsigned SEGMENTLEN = 0,
    // Dyn specific
    int unsigned N_VECTORS,

    int unsigned ACTIVATION_WIDTH = 8,
    int unsigned WEIGHT_WIDTH = ACTIVATION_WIDTH,
    int unsigned ACCU_WIDTH = 2*ACTIVATION_WIDTH+$clog2(MH),
    bit NARROW_WEIGHTS     = 0,
	bit SIGNED_ACTIVATIONS = 1,

    bit PUMPED_COMPUTE = 1,
    bit FORCE_BEHAVIORAL = 0,
	bit M_REG_LUT = 1,

    // Safely deducible parameters
	localparam int unsigned  INPUT_2_STREAM_WIDTH               = PE * WEIGHT_WIDTH,
	localparam int unsigned  INPUT_2_STREAM_WIDTH_BA            = (INPUT_2_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_1_STREAM_WIDTH               = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_1_STREAM_WIDTH_BA            = (INPUT_1_STREAM_WIDTH  + 7)/8 * 8,
    localparam int unsigned  OUTPUT_STREAM_WIDTH                = PE * ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA             = (OUTPUT_STREAM_WIDTH + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN  = SIMD % 2
) (
    // Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	// synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// Matrix stream - input 1
	input	logic [INPUT_1_STREAM_WIDTH_BA-1:0]  s_axis_input_0_tdata,
	input	logic  s_axis_input_0_tvalid,
	output	logic  s_axis_input_0_tready,

    // Matrix stream - input 2
	input	logic [INPUT_2_STREAM_WIDTH_BA-1:0]  s_axis_input_1_tdata,
	input	logic  s_axis_input_1_tvalid,
	output	logic  s_axis_input_1_tready,

	// Matrix stream - output
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready
);

//
// Signals
//

// Input weights
typedef logic [PE-1:0][WEIGHT_WIDTH-1:0] dyn_w_t;
typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] mu_w_t;
uwire mu_w_t axis_input_1_tdata;
logic axis_input_1_tvalid;
logic axis_input_1_tready;

//
// Instantiations
//

// Matrix load
mv_matrix_load #(
    .PE(PE), .SIMD(SIMD),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .MH(MH), .MW(MW),
    .N_REPS(N_VECTORS)
) inst_matrix_load (
    .clk(ap_clk),
    .rst(~ap_rst_n),
    .ivld(s_axis_input_1_tvalid),
    .irdy(s_axis_input_1_tready),
    .idat(dyn_w_t'(s_axis_input_1_tdata)),
    .ovld(axis_input_1_tvalid),
    .ordy(axis_input_1_tready),
    .odat(axis_input_1_tdata)
);

// MVU
mvu_vvu_axi #(
    .IS_MVU(IS_MVU),
    .COMPUTE_CORE(COMPUTE_CORE),
    .MW(MW),
    .MH(MH),
    .PE(PE),
    .SIMD(SIMD),
    .SEGMENTLEN(SEGMENTLEN),

    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .WEIGHT_WIDTH(ACTIVATION_WIDTH),
    .ACCU_WIDTH(ACCU_WIDTH),
    .NARROW_WEIGHTS(NARROW_WEIGHTS),
    .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),

    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL),
    .M_REG_LUT(M_REG_LUT)
) inst_mvu_vvu_axi (
    .ap_clk                     (ap_clk),
    .ap_clk2x                   (ap_clk2x),
    .ap_rst_n                   (ap_rst_n),

    .s_axis_weights_tdata       (axis_input_1_tdata),
    .s_axis_weights_tvalid      (axis_input_1_tvalid),
    .s_axis_weights_tready      (axis_input_1_tready),

    .s_axis_input_tdata         (s_axis_input_0_tdata),
    .s_axis_input_tvalid        (s_axis_input_0_tvalid),
    .s_axis_input_tready        (s_axis_input_0_tready),

    .m_axis_output_tdata        (m_axis_output_tdata),
    .m_axis_output_tvalid       (m_axis_output_tvalid),
    .m_axis_output_tready       (m_axis_output_tready)
);

endmodule

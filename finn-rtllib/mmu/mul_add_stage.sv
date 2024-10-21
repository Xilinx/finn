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
//`define BEHAVIOURAL

module mul_add_stage #(
    parameter                                               SIMD,
    parameter                                               PE,
    parameter                                               ACTIVATION_WIDTH,
    parameter                                               ACCU_WIDTH,
    parameter                                               FORCE_BEHAVIOURAL = 0,
    parameter                                               PUMPED_COMPUTE = 0,
    parameter                                               COMPUTE_CORE = "mvu_vvu_8sx9_dsp58",
    parameter                                               SIGNED_ACTIVATIONS = 1,
    parameter                                               SEGMENTLEN = 2,
    parameter                                               FORCE_BEHAVIORAL = 0,
    parameter                                               NARROW_WEIGHTS = 1,
    parameter                                               SIMD_UNEVEN = SIMD % 2
) (
    input  logic                                            clk,
    input  logic                                            clk2x,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]           amvau_i,
    input  logic [PE-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0]   mvu_w,
    input  logic                                            ival,
    input  logic                                            ilast,

	input  logic [PE-1:0][ACCU_WIDTH-1:0]                   i_acc,
    output logic                                            inc_acc,

    output logic [PE-1:0][ACCU_WIDTH-1:0]                   odat,
    output logic                                            oval,
    output logic                                            olast
);

// DSP
//- Conditionally Pumped DSP Compute ------------------------------------
typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
uwire  ovld_int;
uwire dsp_p_t  odat_int;

if(1) begin : blkDsp
	localparam int IS_MVU = 1;
	localparam int unsigned  EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE ? SIMD+1 : SIMD;
	localparam int unsigned  DSP_SIMD = EFFECTIVE_SIMD/(PUMPED_COMPUTE+1);
	typedef logic [PE    -1:0][DSP_SIMD-1:0][ACTIVATION_WIDTH    -1:0]  dsp_w_t;
	typedef logic [PE-1:0][DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  dsp_a_t;

	uwire  dsp_clk;
	uwire  dsp_en;

	uwire  dsp_last;
	uwire  dsp_zero;
	uwire dsp_w_t  dsp_w;
	uwire dsp_a_t  dsp_a;

	uwire  dsp_vld;
	uwire dsp_p_t  dsp_p;

	if(!PUMPED_COMPUTE) begin : genUnpumpedCompute
		assign	dsp_clk = clk;
		assign	dsp_en  = en;

		assign	dsp_last = ival;
		assign	dsp_zero = !ival;
		assign	dsp_w = mvu_w;
		assign	dsp_a = amvau_i;

		assign	ovld_int = dsp_vld;
		assign	odat_int = dsp_p;
	end : genUnpumpedCompute
	else begin : genPumpedCompute
		assign	dsp_clk = clk2x;

		// Identify second fast cycle just before active slow clock edge
		logic  Active = 0;
					always_ff @(posedge clk2x) begin
			if(rst)  Active <= 0;
			else     Active <= !Active;
		end

		// The input for a slow cycle is split across two fast cycles along the SIMD dimension.
		//	- Both fast cycles are controlled by the same enable state.
		//	- A zero cycle is duplicated across both fast cycles.
		//	- The last flag must be restricted to the second fast cycle.

		dsp_w_t  W = 'x;
		for(genvar  pe = 0; pe < PE; pe++) begin : genPERegW

			uwire [2*DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  w;
			for(genvar  i =    0; i <       SIMD; i++)  assign  w[i] = mvu_w[pe][i];
			for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  w[i] = 0;

			always_ff @(posedge clk2x) begin
				if(rst)      W[pe] <= 'x;
				else if(en)  W[pe] <= w[(Active? DSP_SIMD : 0) +: DSP_SIMD];
			end

		end : genPERegW

		dsp_a_t  A = 'x;
		for(genvar  pe = 0; pe < PE; pe++) begin : genPERegA

			uwire [2*DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  a;
			for(genvar  i =    0; i <       SIMD; i++)  assign  a[i] = amvau_i[pe][i];
			for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  a[i] = 0;

			always_ff @(posedge clk2x) begin
				if(rst)      A[pe] <= 'x;
				else if(en)  A[pe] <= a[(Active? DSP_SIMD : 0) +: DSP_SIMD];
			end

		end : genPERegA

		logic  Zero = 1;
		always_ff @(posedge clk2x) begin
			if(rst) begin
				Zero <= 1;
			end
			else if(en) begin
				Zero <= !ival;
			end
		end

		assign	dsp_en = en;
		assign	dsp_last = ival;
		assign	dsp_zero = Zero;
		assign	dsp_w = W;
		assign	dsp_a = A;

		// Since no two consecutive last cycles will ever be asserted on the input,
		// valid outputs will also always be spaced by, at least, one other cycle.
		// We can always hold a captured output for two cycles to allow the slow
		// clock to pick it up.
		logic    Vld = 0;
		dsp_p_t  P = 'x;
		always_ff @(posedge clk2x) begin
			if(rst) begin
				Vld <= 0;
				P   <= 'x;
			end
			else if(en) begin
				if(dsp_vld)  P <= dsp_p;
				Vld <= dsp_vld || (Vld && !Active);
			end
		end
		assign	ovld_int = dsp_vld;
		assign	odat_int = P;

	end : genPumpedCompute

	case(COMPUTE_CORE)
	"mvu_vvu_8sx9_dsp58":
if(PUMPED_COMPUTE) begin
					mvu_vvu_8sx9_dsp58 #(
							.IS_MVU(IS_MVU),
							.PE(PE), .SIMD(DSP_SIMD),
							.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
							.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN),
							.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
					) core (
			.clk(clk2x), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
end
else begin
					mvu_vvu_8sx9_dsp58 #(
							.IS_MVU(IS_MVU),
							.PE(PE), .SIMD(DSP_SIMD),
							.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
							.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN),
							.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
					) core (
			.clk(clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
end
	"mvu_4sx4u_dsp48e1":
		mvu_4sx4u #(
			.PE(PE), .SIMD(DSP_SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .NARROW_WEIGHTS(NARROW_WEIGHTS),
			.VERSION(1), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	"mvu_4sx4u_dsp48e2":
		mvu_4sx4u #(
			.PE(PE), .SIMD(DSP_SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .NARROW_WEIGHTS(NARROW_WEIGHTS),
			.VERSION(2), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	"mvu_8sx8u_dsp48":
		mvu_8sx8u_dsp48 #(
			.PE(PE), .SIMD(DSP_SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) core (
			.clk(dsp_clk), .rst, .en(dsp_en),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);
	default: initial begin
		$error("Unrecognized COMPUTE_CORE '%s'", COMPUTE_CORE);
		$finish;
	end
	endcase

end : blkDsp

localparam integer DSP_58_LAT = 1 + 4 + $clog2(SIMD);
localparam integer DSP_48_LAT = 1 + 5;
localparam integer BEH_LAT = 2;

localparam integer MUL_LAT = FORCE_BEHAVIOURAL ? BEH_LAT : (COMPUTE_CORE == "mvu_vvu_8sx9_dsp58" ? DSP_58_LAT : DSP_48_LAT);

// REG
always_ff @(posedge clk) begin
    if(rst) begin
        odat <= 0;
		oval <= 1'b0;
    end
    else begin
        if(en) begin
			for(int i = 0; i < PE; i++) begin
            	odat[i] <= $signed(odat_int[i]) + $signed(i_acc[i]);
			end
			oval <= ovld_int;
        end
    end
end

assign inc_acc = en && ovld_int;

logic [MUL_LAT:0] last;

assign last[0] = ilast;

always_ff @(posedge clk) begin
    if(rst) begin
        for(int i = 1; i <= MUL_LAT; i++) begin
            last[i] <= 0;
        end
    end
    else begin
        if(en) begin
            for(int i = 1; i <= MUL_LAT; i++) begin
                last[i] <= last[i-1];
            end
        end 
    end
end

assign olast = last[MUL_LAT];


endmodule
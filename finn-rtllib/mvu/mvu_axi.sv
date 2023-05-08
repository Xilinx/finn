/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	Matrix Vector Unit (MVU) AXI-lite interface wrapper.
 *****************************************************************************/

module mvu_axi #(
	int unsigned MW,
	int unsigned MH,
	int unsigned PE,
	int unsigned SIMD,
	int unsigned ACTIVATION_WIDTH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,
	bit SIGNED_ACTIVATIONS = 0,
	int unsigned SEGMENTLEN = 0,
	bit FORCE_BEHAVIORAL = 0,
	string MVU_IMPL_STYLE,

	localparam int unsigned WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
	localparam int unsigned INPUT_STREAM_WIDTH_BA = (SIMD*ACTIVATION_WIDTH+7)/8 * 8,
	localparam int unsigned WEIGHT_STREAM_WIDTH = PE*SIMD*WEIGHT_WIDTH,
	localparam int unsigned INPUT_STREAM_WIDTH = SIMD*ACTIVATION_WIDTH,
	localparam int unsigned SF = MW/SIMD,
	localparam int unsigned NF = MH/PE,
	localparam int unsigned OUTPUT_LANES = PE,
	localparam int unsigned OUTPUT_STREAM_WIDTH_BA = (OUTPUT_LANES*ACCU_WIDTH + 7)/8 * 8
)
(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_weights_tdata,
	input	logic  s_axis_weights_tvalid,
	output	logic  s_axis_weights_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata,
	input	logic  s_axis_input_tvalid,
	output	logic  s_axis_input_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready
);

//-------------------- Parameter sanity checks --------------------\\
	initial begin
		if (MW % SIMD != 0) begin
			$error("Matrix width (%0d) is not a multiple of SIMD (%0d).", MW, SIMD);
			$finish;
		end
		if (MH % PE != 0) begin
			$error("Matrix height (%0d) is not a multiple of PE (%0d).", MH, PE);
			$finish;
		end
		if (ACTIVATION_WIDTH > 9) begin
			$error("Activation width of %0d-bits exceeds maximum of 9-bits", ACTIVATION_WIDTH);
			$finish;
		end
		if (WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if (SIGNED_ACTIVATIONS == 0 && ACTIVATION_WIDTH==9) begin
			$error("Activation width of %0d-bits exceeds maximum of 8-bits for unsigned numbers", ACTIVATION_WIDTH);
			$finish;
		end
		if (MVU_IMPL_STYLE == "mvu_8sx9") begin
			if (SEGMENTLEN == 0) begin
				$warning("Segment length of %0d defaults to chain length", SEGMENTLEN);
			end
			if (SEGMENTLEN > (SIMD+2)/3) begin
				$error("Segment length of %0d exceeds chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
				$finish;
			end
		end
	end

	uwire clk = ap_clk;
	uwire rst = !ap_rst_n;

	typedef logic [INPUT_STREAM_WIDTH-1 : 0] mvauin_t;

	uwire mvauin_t amvau;
	uwire alast;
	uwire afin;
	uwire avld;
	uwire ardy;

	replay_buffer #(.LEN(SF), .REP(NF), .W($bits(mvauin_t))) activation_replay (
		.clk, .rst,
		.ivld(s_axis_input_tvalid), .irdy(s_axis_input_tready), .idat(mvauin_t'(s_axis_input_tdata)),
		.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast), .ofin(afin)
	);

//-------------------- Input control --------------------\\
	uwire en;
	uwire istb = avld && s_axis_weights_tvalid;
	assign ardy = en && s_axis_weights_tvalid;
	assign s_axis_weights_tready = en && avld;

//-------------------- Core MVU --------------------\\
	uwire ovld;
	uwire [PE-1:0][ACCU_WIDTH-1:0] odat;
	typedef logic [WEIGHT_STREAM_WIDTH-1 : 0] mvauin_weight_t;
	
	if (MVU_IMPL_STYLE == "mvu_8sx9_dsp58") begin : genMVU8sx9
		mvu_8sx9 #(.PE(PE), .SIMD(SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN),
		.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk, .rst, .en,
			.last(alast && avld), .zero(!istb), .w(mvauin_weight_t'(s_axis_weights_tdata)), .a(amvau),
			.vld(ovld), .p(odat)
		);
	end
	else if (MVU_IMPL_STYLE == "mvu_4sx4u") begin : genMVU4sx4u
		mvu_4sx4u #(.PE(PE), .SIMD(SIMD), .ACCU_WIDTH(ACCU_WIDTH), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk, .rst, .en,
			.last(alast && avld), .zero(!istb), .w(mvauin_weight_t'(s_axis_weights_tdata)), .a(amvau),
			.vld(ovld), .p(odat)
		);
	end
	else if (MVU_IMPL_STYLE == "mvu_8sx8u_dsp48") begin : genMVU8sx8u
		mvu_8sx8u_dsp48 #(.PE(PE), .SIMD(SIMD), .ACCU_WIDTH(ACCU_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
		 .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
			.clk, .rst, .en,
			.last(alast && avld), .zero(!istb), .w(mvauin_weight_t'(s_axis_weights_tdata)), .a(amvau),
			.vld(ovld), .p(odat)
		);
	end
	else initial begin
		$error("Unrecognized MVU_IMPL_STYLE!");
		$finish;
	end

//-------------------- Output register slice --------------------\\
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	} A = '{ vld: 0, default: 'x};

	assign en = !A.vld || !ovld;

	uwire  b_load;
	always_ff @(posedge clk) begin
		if(rst)		A <= '{ vld: 0, default: 'x };
		else if(!A.vld || b_load) begin
			A.vld <= ovld && en;
			for(int unsigned  i = 0; i < PE; i++) begin
				// CR-1148862:
				// A.dat[i] <= odat[i];
				automatic logic [ACCU_WIDTH-1:0]  v = odat[i];
				A.dat[i] <= v[ACCU_WIDTH-1:0];
			end
		end
	end
	
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	} B = '{ vld: 0, default: 'x};

	assign	b_load = !B.vld || m_axis_output_tready;
	always_ff @(posedge clk) begin
		if(rst)		B <= '{ default: 'x };
		else begin
			if(b_load)	B <= '{ vld: A.vld, dat: A.dat};
		end	
	end

	assign	m_axis_output_tvalid = B.vld;
	assign	m_axis_output_tdata  = B.dat;

endmodule : mvu_axi
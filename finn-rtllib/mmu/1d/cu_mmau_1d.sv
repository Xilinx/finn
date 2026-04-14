/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
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
 * @brief	Compute unit (DSP grid) - MMAU
 * @author	Dario Korolija <dario.korolija@amd.com>
 *****************************************************************************/

module cu_mmau_1d #(
    int unsigned PE,
    int unsigned CLEN,
    int unsigned CU_SIMD,
    
    int unsigned ACTIVATION_WIDTH,
    int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,

    bit SIGNED_ACTIVATIONS = 1,
    int unsigned FORCE_BEHAVIOURAL = 0
  )  (
    // Global Control
	input  logic clk,
    input  logic rst,

    // Enable
    output logic en,

	// Input
    input  logic ivld,
    input  logic [CLEN-1:0] ilast,
    input  logic [CLEN-1:0][CU_SIMD-1:0][ACTIVATION_WIDTH-1:0] a,
    input  logic [PE-1:0][CU_SIMD-1:0][WEIGHT_WIDTH-1:0] w,

	// Ouput
	output logic m_axis_tvalid,
    input  logic m_axis_tready,
    output logic [PE-1:0][ACCU_WIDTH-1:0] m_axis_tdata
  );
	
// Startup Recovery Watchdog
//  The DSP slice needs 100ns of recovery time after initial startup before
//  being able to ingest input properly. This watchdog discovers violating
//  stimuli during simulation and produces a corresponding warning.
//------------------------------------------------------------------------------------
	if(1) begin : blkRecoveryWatch
		logic  Dirty = 1;
		initial begin
			#100ns;
			Dirty <= 0;
		end

		always_ff @(posedge clk) begin
			assert(!Dirty || rst) else begin
				$warning("%m: Feeding input during DSP startup recovery. Expect functional errors.");
			end
		end
	end : blkRecoveryWatch

// Shifts - activations and weights
//------------------------------------------------------------------------------------
    localparam int unsigned PAD_BITS_ACT = 9 - ACTIVATION_WIDTH;
    localparam int unsigned PAD_BITS_WEIGHT = 8 - WEIGHT_WIDTH;

    logic [CLEN:0][PE-1:0][CU_SIMD*WEIGHT_WIDTH-1:0] Wc;
    logic [CLEN-1:0][PE-1:0][23:0] Wc_int;

    for(genvar i = 0; i < PE; i++) begin
        assign Wc[0][i] = w[i];

        for (genvar k = 0; k < CU_SIMD; k++) begin
            assign Wc_int[0][i][8*k +: 8] =
                PAD_BITS_WEIGHT == 0 ? Wc[0][i][WEIGHT_WIDTH*k+:WEIGHT_WIDTH] : { {PAD_BITS_WEIGHT{Wc[0][i][k*WEIGHT_WIDTH+WEIGHT_WIDTH-1]}}, Wc[0][i][k*WEIGHT_WIDTH+:WEIGHT_WIDTH] };
        end
    end 

    /*
	always_ff @(posedge clk) begin
		if(rst) begin
            for(int i = 1; i < CLEN; i++) begin
                for(int j = 0; j < PE; j++) begin
                    Wc[i][j] <= 'X;
                end
            end
        end
            for(int i = 1; i < CLEN; i++) begin
                for(int j = 0; j < PE; j++) begin
                    if(ivld) begin
                        Wc[i][j] <= Wc[i-1][j];
                    end
                end
            end
	end
    */

// Shifts - per DSP
//------------------------------------------------------------------------------------
	localparam int unsigned DSP_PIPELINE_STAGES = 3;
    logic [CLEN-1:0][DSP_PIPELINE_STAGES:0] Lc;

    for(genvar i = 0; i < CLEN; i++) begin
        assign Lc[i][0] = ilast[i];
    end

    always_ff @(posedge clk) begin
        if(rst) begin
            for(int i = 0; i < CLEN; i++) begin
                for(int k = 1; k <= DSP_PIPELINE_STAGES; k++) begin
                    Lc[i][k] <= 'X;
                end
            end
        end
        else begin
            for(int i = 0; i < CLEN; i++) begin
                for(int k = 1; k <= DSP_PIPELINE_STAGES; k++) begin
                    if(ivld) begin
                        Lc[i][k] <= Lc[i][k-1];
                    end
                end
            end
        end
    end

// Instantiate PE x CLEN DSPs
//------------------------------------------------------------------------------------
    logic [CLEN-1:0][PE-1:0][ACCU_WIDTH-1:0] pout;

  /*  if(FORCE_BEHAVIOURAL == 1) begin
        logic [CLEN-1:0][CU_SIMD*ACTIVATION_WIDTH-1:0] Ac_int;
        logic [CLEN-1:0][PE-1:0][CU_SIMD*WEIGHT_WIDTH-1:0] Wc_int;
        logic [CLEN-1:0][PE-1:0][CU_SIMD-1:0][ACCU_WIDTH-1:0] Mc_int_part;
        logic [CLEN-1:0][PE-1:0][ACCU_WIDTH-1:0] Mc_int_sum;
        logic [CLEN-1:0][PE-1:0][ACCU_WIDTH-1:0] Mc_int;


        for (genvar i = 0; i < CLEN; i++) begin
            always_ff @(posedge clk) begin
                if(rst) begin
                    Ac_int[i] <= 'X;
                end else begin
                    if(ivld) begin
                        Ac_int[i] <= a[i];
                    end
                end
            end

            for (genvar j = 0; j < PE; j++) begin           
                always_comb begin
                    Mc_int_sum[i][j] = 0;

                    for(int k = 0; k < CU_SIMD; k++) begin
                        Mc_int_part[i][j][k] = $signed(Ac_int[i][k*ACTIVATION_WIDTH+:ACTIVATION_WIDTH]) * $signed(Wc_int[i][j][k*WEIGHT_WIDTH+:WEIGHT_WIDTH]);
                        Mc_int_sum[i][j] = $signed(Mc_int_sum[i][j]) + $signed(Mc_int_part[i][j][k]);
                    end
                end

                always_ff @(posedge clk) begin
                    if(rst) begin
                        Wc_int[i][j] <= '0;
                        Mc_int[i][j] <= '0;
                        pout[i][j] <= '0;
                    end else begin
                        if(ivld) begin
                            Wc_int[i][j] <= Wc[i][j];
                            Mc_int[i][j] <= $signed(Mc_int_sum[i][j]);
                            pout[i][j] <= Lc[i][DSP_PIPELINE_STAGES] ? $signed(Mc_int[i][j]) : $signed(Mc_int[i][j]) + $signed(pout[i][j]);
                        end
                    end
                end
            end
        end
    end else begin */
        localparam int INTERNAL_REGS = 1; // 1 : 0
        localparam bit PREG = 1;
        localparam int CC_LEN = CLEN / 4;

        logic [CLEN-1:0][26:0] Ac_int;
        //logic [CLEN-1:0][PE-1:0][23:0] Wc_int;
        logic [CLEN-1:0][PE-1:0][23:0] tmp_cc;

        for (genvar i = 0; i < CLEN; i++) begin
            for (genvar k = 0; k < CU_SIMD; k++) begin
                assign Ac_int[i][9*k +: 9] =
                    SIGNED_ACTIVATIONS ? PAD_BITS_ACT == 0 ? a[i][k] : { {PAD_BITS_ACT{a[i][k][ACTIVATION_WIDTH-1]}}, a[i][k] }
                        : PAD_BITS_ACT == 0 ? a[i][k] : { {PAD_BITS_ACT{1'b0}}, a[i][k] } ;
            end
            

            for (genvar j = 0; j < PE; j++) begin   
               /* for (genvar k = 0; k < CU_SIMD; k++) begin
                    assign Wc_int[i][j][8*k +: 8] =
                        PAD_BITS_WEIGHT == 0 ? Wc[i][j][WEIGHT_WIDTH*k+:WEIGHT_WIDTH] : { {PAD_BITS_WEIGHT{Wc[i][j][k*WEIGHT_WIDTH+WEIGHT_WIDTH-1]}}, Wc[i][j][k*WEIGHT_WIDTH+:WEIGHT_WIDTH] };
                end */


                DSP58 #(
                    // Feature Control Attributes: Data Path Selection
                    .AMULTSEL("A"),                     // Selects A input to multiplier (A, AD)
                    .A_INPUT("DIRECT"),                 // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
                    .BMULTSEL("B"),                     // Selects B input to multiplier (AD, B)
                    .B_INPUT((i % CC_LEN == 0) ? "DIRECT" : "CASCADE"),                 // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
                    .DSP_MODE("INT8"),                  // Configures DSP to a particular mode of operation. Set to INT24 for
                                                        // legacy mode.
                    .PREADDINSEL("A"),                  // Selects input to pre-adder (A, B)
                    .RND(58'h000000000000000),          // Rounding Constant
                    .USE_MULT("MULTIPLY"),              // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
                    .USE_SIMD("ONE58"),                 // SIMD selection (FOUR12, ONE58, TWO24)
                    .USE_WIDEXOR("FALSE"),              // Use the Wide XOR function (FALSE, TRUE)
                    .XORSIMD("XOR24_34_58_116"),        // Mode of operation for the Wide XOR (XOR12_22, XOR24_34_58_116)
                    // Pattern Detector Attributes: Pattern Detection Configuration
                    .AUTORESET_PATDET("NO_RESET"),      // NO_RESET, RESET_MATCH, RESET_NOT_MATCH
                    .AUTORESET_PRIORITY("RESET"),       // Priority of AUTORESET vs. CEP (CEP, RESET).
                    .MASK(58'h0ffffffffffffff),         // 58-bit mask value for pattern detect (1=ignore)
                    .PATTERN(58'h000000000000000),      // 58-bit pattern match for pattern detect
                    .SEL_MASK("MASK"),                  // C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
                    .SEL_PATTERN("PATTERN"),            // Select pattern value (C, PATTERN)
                    .USE_PATTERN_DETECT("NO_PATDET"),   // Enable pattern detect (NO_PATDET, PATDET)
                    // Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
                    .IS_ALUMODE_INVERTED(4'b0000),      // Optional inversion for ALUMODE
                    .IS_CARRYIN_INVERTED(1'b0),         // Optional inversion for CARRYIN
                    .IS_CLK_INVERTED(1'b0),             // Optional inversion for CLK
                    .IS_INMODE_INVERTED(5'b00000),      // Optional inversion for INMODE
                    .IS_NEGATE_INVERTED(3'b000),        // Optional inversion for NEGATE
                    .IS_OPMODE_INVERTED({2'b00, // W: LAST ? 0 : P
                                        3'b000, // Z: 0
                                        2'b01, // Y : M
                                        2'b01  // X: M
                    }), // Optional inversion for OPMODE
                    .IS_RSTALLCARRYIN_INVERTED(1'b0),   // Optional inversion for RSTALLCARRYIN
                    .IS_RSTALUMODE_INVERTED(1'b0),      // Optional inversion for RSTALUMODE
                    .IS_RSTA_INVERTED(1'b0),            // Optional inversion for RSTA
                    .IS_RSTB_INVERTED(1'b0),            // Optional inversion for RSTB
                    .IS_RSTCTRL_INVERTED(1'b0),         // Optional inversion for STCONJUGATE_A
                    .IS_RSTC_INVERTED(1'b0),            // Optional inversion for RSTC
                    .IS_RSTD_INVERTED(1'b0),            // Optional inversion for RSTD
                    .IS_RSTINMODE_INVERTED(1'b0),       // Optional inversion for RSTINMODE
                    .IS_RSTM_INVERTED(1'b0),            // Optional inversion for RSTM
                    .IS_RSTP_INVERTED(1'b0),            // Optional inversion for RSTP
                    // Register Control Attributes: Pipeline Register Configuration
                    .ACASCREG(INTERNAL_REGS),          // Number of pipeline stages between A/ACIN and ACOUT (0-2)
                    .ADREG(0),                          // Pipeline stages for pre-adder (0-1)
                    .ALUMODEREG(0),                     // Pipeline stages for ALUMODE (0-1)
                    .AREG(INTERNAL_REGS),              // Pipeline stages for A (0-2)
                    .BCASCREG(INTERNAL_REGS),          // Number of pipeline stages between B/BCIN and BCOUT (0-2)
                    .BREG(INTERNAL_REGS),              // Pipeline stages for B (0-2)
                    .CARRYINREG(0),                     // Pipeline stages for CARRYIN (0-1)
                    .CARRYINSELREG(0),                  // Pipeline stages for CARRYINSEL (0-1)
                    .CREG(0),                           // Pipeline stages for C (0-1)
                    .DREG(0),                           // Pipeline stages for D (0-1)
                    .INMODEREG(1),                      // Pipeline stages for INMODE (0-1)
                    .MREG(1),                           // Multiplier pipeline stages (0-1)
                    .OPMODEREG(1),                      // Pipeline stages for OPMODE (0-1)
                    .PREG(PREG),                        // Number of pipeline stages for P (0-1)
                    .RESET_MODE("SYNC")                 // Selection of synchronous or asynchronous reset. (ASYNC, SYNC).
                )
                DSP58_inst (
                    // Cascade outputs: Cascade Ports
                    .ACOUT(),                           // 34-bit output: A port cascade
                    .BCOUT((i % CC_LEN == CC_LEN-1) ? tmp_cc[i+1][j] : Wc_int[i+1][j]),             // 24-bit output: B cascade
                    .CARRYCASCOUT(),                    // 1-bit output: Cascade carry
                    .MULTSIGNOUT(),                     // 1-bit output: Multiplier sign cascade
                    .PCOUT()    ,                           // 58-bit output: Cascade output
                    // Control outputs: Control Inputs/Status Bits
                    .OVERFLOW(),                        // 1-bit output: Overflow in add/acc
                    .PATTERNBDETECT(),                  // 1-bit output: Pattern bar detect
                    .PATTERNDETECT(),                   // 1-bit output: Pattern detect
                    .UNDERFLOW(),                       // 1-bit output: Underflow in add/acc
                    // Data outputs: Data Ports
                    .CARRYOUT(),                        // 4-bit output: Carry
                    .P(pout[i][j]),                     // 58-bit output: Primary data
                    .XOROUT(),                          // 8-bit output: XOR data
                    // Cascade inputs: Cascade Ports
                    .ACIN('x),                          // 34-bit input: A cascade data
                    .BCIN((i % CC_LEN == 0) ? 'x : Wc_int[i][j]),                // 24-bit input: B cascade
                    .CARRYCASCIN('x),                   // 1-bit input: Cascade carry
                    .MULTSIGNIN('x),                    // 1-bit input: Multiplier sign cascade
                    .PCIN('x),                          // 58-bit input: P cascade
                    // Control inputs: Control Inputs/Status Bits
                    .ALUMODE(4'h0),                     // 4-bit input: ALU control
                    .CARRYINSEL('0),                    // 3-bit input: Carry select
                    .CLK(clk),                          // 1-bit input: Clock
                    .INMODE({5'b10001}),     // 5-bit input: INMODE control
                    .NEGATE('0),                        // 3-bit input: Negates the input of the multiplier
                    .OPMODE({
                            Lc[i][DSP_PIPELINE_STAGES-1] ? 2'b00 : 2'b01,
                            7'b000_0000
                    }), // 9-bit input: Operation mode
                    // Data inputs: Data Ports
                    .A({ 7'b0, Ac_int[i] }),             // 34-bit input: A data
                    .B((i % CC_LEN == 0) ? Wc_int[i][j] : 'x),                             // 24-bit input: B data
                    .C('x),                             // 58-bit input: C data
                    .CARRYIN('0),                       // 1-bit input: Carry-in
                    .D('x),                             // 27-bit input: D data
                    // Reset/Clock Enable inputs: Reset/Clock Enable Inputs
                    .ASYNC_RST('0),                     // 1-bit input: Asynchronous reset for all registers.
                    .CEA1(ivld),                          // 1-bit input: Clock enable for 1st stage AREG
                    .CEA2('0),                          // 1-bit input: Clock enable for 2nd stage AREG
                    .CEAD('0),                          // 1-bit input: Clock enable for ADREG
                    .CEALUMODE('0),                     // 1-bit input: Clock enable for ALUMODE
                    .CEB1(ivld),                          // 1-bit input: Clock enable for 1st stage BREG
                    .CEB2('0),                          // 1-bit input: Clock enable for 2nd stage BREG
                    .CEC('0),                           // 1-bit input: Clock enable for CREG
                    .CECARRYIN('0),                     // 1-bit input: Clock enable for CARRYINREG
                    .CECTRL(ivld),                        // 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
                    .CED('0),                           // 1-bit input: Clock enable for DREG
                    .CEINMODE('1),                      // 1-bit input: Clock enable for INMODEREG
                    .CEM(ivld),                           // 1-bit input: Clock enable for MREG
                    .CEP(ivld),                           // 1-bit input: Clock enable for PREG
                    .RSTA(rst),                          // 1-bit input: Reset for AREG
                    .RSTALLCARRYIN('0),                 // 1-bit input: Reset for CARRYINREG
                    .RSTALUMODE('0),                    // 1-bit input: Reset for ALUMODEREG
                    .RSTB(rst),                          // 1-bit input: Reset for BREG
                    .RSTC('0),                          // 1-bit input: Reset for CREG
                    .RSTCTRL(rst),                      // 1-bit input: Reset for OPMODEREG and CARRYINSELREG
                    .RSTD('0),                          // 1-bit input: Reset for DREG and ADREG
                    .RSTINMODE(rst),                    // 1-bit input: Reset for INMODE register
                    .RSTM(rst),                          // 1-bit input: Reset for MREG
                    .RSTP(rst)                          // 1-bit input: Reset for PREG
                );

                
                if(i % CC_LEN == CC_LEN-1) begin
                    sft_reg #(
                        .N(CC_LEN)
                    ) inst_sft_reg (
                        .clk(clk),
                        .ivld(ivld),
                        .din(Wc_int[i-(CC_LEN-1)][j]),
                        .dout(Wc_int[i+1][j])
                    );
                end

            end
        end
 //   end

// Collect
//------------------------------------------------------------------------------------
    logic [CLEN-1:0][PE-1:0][ACCU_WIDTH-1:0] Pc;
    logic [CLEN-1:0] Pc_vld;

    always_ff @(posedge clk) begin
        if(rst) begin
            for(int i = 0; i < CLEN; i++) begin
                Pc[i] <= '0;
            end
        end else begin
            for(int i = 0; i < CLEN; i++) begin
                if(ivld) begin
                    if(i == CLEN-1) begin
                        Pc[i] <= pout[i];
                        Pc_vld[i] <= Lc[i][DSP_PIPELINE_STAGES];
                    end else begin
                        Pc[i] <= Lc[i][DSP_PIPELINE_STAGES] ?  pout[i] : Pc[i+1];
                        Pc_vld[i] <= Lc[i][DSP_PIPELINE_STAGES] ? 1'b1 : Pc_vld[i+1];
                    end 
                end
            end
        end
    end

    logic ovld;
    logic [PE-1:0][ACCU_WIDTH-1:0] p;

    assign ovld = Pc_vld[0];
    assign p = Pc[0];

    collect_out_1d #(
        .PE(PE), 
        .ACCU_WIDTH(ACCU_WIDTH)
    ) inst_collect_out (
        .clk(clk), .rst(rst),
        .en(en),
        .p_tdata(p), .p_tvalid(ovld),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready)
    );

endmodule
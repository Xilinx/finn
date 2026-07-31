/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Combinational integer to IEEE 754 float32 converter (round to zero).
 * @author	Shane Fleming <shane.fleming@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module int_to_fp32 #(
	int unsigned  WIDTH,	// Input width
	bit           SIGNED
)(
	input	logic [WIDTH-1:0]  ival,
	output	logic      [31:0]  fval
);

	//=== Sign & Magnitude ================================================
	uwire              sign = SIGNED ? ival[WIDTH-1] : 0;
	uwire [WIDTH-1:0]  mag  = SIGNED ? (sign ? -ival : ival) : ival;

	//=== Leading-Zero Count via Bit-Reversal + Rightmost-Bit Isolation ===
	uwire [WIDTH-1:0]  rev = {<<{mag}};        // bit reversal for first-one detection
	uwire [WIDTH-1:0]  oh  = (~rev + 1) & rev; // one-hot isolation of rightmost one in reversed input

	//=== Exponent & Mantissa Tap Position Calculation ====================
	// OR Reduction from activating hot one positions
	//  - a zero magnitude yields the required zero exponent
	logic [7:0]  exp; // 127, ..., WIDTH +126
	logic [(WIDTH < 2? 0 : $clog2(WIDTH)-1):0]  tap; //   0, ..., WIDTH -  1
	always_comb begin
		exp = 0;
		tap = 0;
		for(int unsigned  i = 0; i < WIDTH; i++) begin
			if(oh[i]) begin
				exp |= (WIDTH + 126) - i;
				tap |= (WIDTH -   1) - i;
			end
		end
	end

	//=== Mantissa Extraction =============================================
	uwire [WIDTH+22:0]  mag_ext = {mag, 23'b0};
	uwire [      22:0]  man = mag_ext[tap+:23];

	//=== Output Assembly =================================================
	assign	fval = {sign, exp, man};

endmodule : int_to_fp32

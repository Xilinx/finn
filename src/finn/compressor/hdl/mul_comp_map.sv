/**
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * @brief
 *	Broadcasts multiplication inputs to feed a bit product matrix for compression.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	This interface component broadcasts multiplication inputs to produce a bit
 *	product matrix like the one below. The output is flattened for the
 *	ingestion by a compressor with the indicated indices:
 *
 *	                                   [6]a3.b0  [3]a2.b0  [1]a1.b0  [0]a0.b0
 *	                        [10]a3.b1  [7]a2.b1  [4]a1.b1  [2]a0.b1
 *	             [13]a3.b2  [11]a2.b2  [8]a1.b2  [5]a0.b2
 *	  [15]a3.b3  [14]a2.b3  [12]a1.b3  [9]a0.b3
 *
 *	Functions designated to informing about the produced shape are provided:
 *	  - columns()   - the number of columns in the matrix shape.
 *	  - height(col) - the height of the specified column.
 *	Additionally, the bit product operator is identified for each index by:
 *	  - gate_op(idx) - the assumed bit product operator as hex LUT code.
 *
 *	In the case of unsigned operands, all bit products require to be computed
 *	as AND gates (8), i.e. m[i] = oa[i] & ob[i].
 *
 * The operands can be specified to be signed, which will effect these changes
 * to produce the correct funtionality:
 *
 * SIGNED_A
 * --------
 *	The sign extensions of the multiples of input a are not materialized.
 *	Instead, this identity with s := a_{NA-1} & b_i is applied:
 *		s ... s  s
 *		----------
 *		        !s
 *		        -1
 *	In consequence:
 *	  - The `gate_op()` for the left matrix boundary is identified as NAND (7).
 *	  - The `absolute_term()` function returns a valu of
 *	       (-2^NB + 1) * 2^{NA-1}
 *	    that must be added to the matrix sum for the correct product value.
 *
 * SIGNED_B
 * --------
 *	The sign extension of input b is not materialized.
 *	Instead, the multiple of a by the sign bit of b is weighted negatively,
 *	which expands the produced matrix as follows:
 *
 *	                                             [ 6]a3.b0  [3]a2.b0  [1]a1.b0  [0]a0.b0
 *	                                  [11]a3.b1  [ 7]a2.b1  [4]a1.b1  [2]a0.b1
 *	                       [14]a3.b2  [12]a2.b2  [ 8]a1.b2  [5]a0.b2
 *	  [17]0!b3! [16]a3!b3  [15]a2!b3  [13]a1!b3  [ 9]a0!b3
 *	        -1                                   [10]   b3
 *	-----------------------------------------------------------------------------------
 *	                                  [10]a0!b3  [ 6]a3.b0  [3]a2.b0  [1]a1.b0  [0]a0.b0
 *	                                  [11]a3.b1  [ 7]a2.b1  [4]a1.b1  [2]a0.b1
 *	                       [14]a3.b2  [12]a2.b2  [ 8]a1.b2  [5]a0.b2
 *	  [17]0!b3! [16]a3!b3  [15]a2!b3  [13]a1!b3  [ 9]a0.b3
 *	        -1
 *
 *	using:
 *		- a.b  :=    a & b
 *		- a!b  :=   !a & b
 *		- a!b! := !(!a & b)
 *
 *	In consequence:
 *	  - The bit sizes of the outputs are wider and the `columns()` count is larger.
 *	  - The `gate_op()` at the shown indeces is identified as 2 or D.
 *	Note that the height of the matrix grows to NB+1 if NA > NB.
 *
 * SIGNED_A & SIGNED_B
 * -------------------
 *	Both approaches are combined for a purely signed multiplication:
 *
 *	                        [10]a0!b3  [ 6]a3.b0! [3]a2.b0  [1]a1.b0  [0]a0.b0
 *	                        [11]a3.b1! [ 7]a2.b1  [4]a1.b1  [2]a0.b1
 *	             [14]a3.b2! [12]a2.b2  [ 8]a1.b2  [5]a0.b2
 *	  [16]a3!b3! [15]a2!b3  [13]a1!b3  [ 9]a0.b3
 *	         -1         -1         -1         -1
 *
 *	using:
 *		- a.b  :=    a & b
 *		- a!b  :=   !a & b
 *		- a.b! := !( a & b)
 *		- a!b! := !(!a & b)
 *	In consequence:
 *	  - The bit sizes of the outputs are wider.
 *	  - The `gate_op()` at the shown indeces is properly identified.
 *	  - The `absolute_term()` function returns a value of
 *	       (-2^NB + 1) * 2^{NA-1}
 *	    that must be added to the matrix sum for the correct product value.
 *	Note that the height of the matrix grows to NB+1 if NA > NB.
 */

interface mul_comp_map #(
	int unsigned  NA,	// bit width of multiplicand
	int unsigned  NB,	// bit width of multiplier
	bit  SIGNED_A,		// signed multiplicand
	bit  SIGNED_B,		// signed multiplier

	// Extra bits due to sign handling and total output size
	localparam int unsigned  NX = (NA == 1) || !SIGNED_B? 0 : SIGNED_A? 1 : 2,
	localparam int unsigned  NM = NA*NB + NX
)(
	// Input Operands
	input	logic [NA-1:0]  ia,  // Multiplicand
	input	logic [NB-1:0]  ib   // Multiplier
);
	// Bit Matrix Broadcasts
	logic [NM-1:0]  oa;
	logic [NM-1:0]  ob;


	// Operand length support is not symmetrical.
	initial begin
		if(NA < NB) begin
			$error("%m: Switch multiplication operands.");
			$finish;
		end
	end

	function int unsigned columns();
		return  NA == 1? 1 : NB + NA - (!SIGNED_B || SIGNED_A);
	endfunction : columns

	function int unsigned height(input int unsigned  col);
		if(NA == 1)  return  col < 1;
		else begin
			automatic int unsigned  ret =
				(col <  NB)?      col + 1 :
				(col <  NA)?      NB :
				(col <  NB+NA-1)? NB+NA-1 - col :
				(col == NB+NA-1)? SIGNED_B && !SIGNED_A :
				/* else */        0;
			if(SIGNED_B && (col == NB))  ret++;
			return  ret;
		end
	endfunction : height

	function bit signed [NA+NB-1:0] absolute_term();
		if(NA == 1)  return  SIGNED_A ^^ SIGNED_B? -1 : 0;
		else begin
			automatic bit signed [NA+NB-1:0]  ret = '{
				NA+NB-1: SIGNED_A || SIGNED_B,
				NA-1:    SIGNED_A,
				default: 0
			};
			return  ret;
		end
	endfunction : absolute_term


	// Beyond the tip of left triangle at column of height 1
	localparam int unsigned  HIGH = NM - (SIGNED_B && !SIGNED_A);

	function bit [3:0] gate_op(input int unsigned  idx);
		if(NA == 1)  return  SIGNED_A ^^ SIGNED_B? 7 : 8;
		else begin
			automatic bit [3:0]  op = 8; // AND

			if(SIGNED_B) begin
				automatic bit  inv = 0;
				// Negative weight for sign-bit row
				for(int unsigned  col = 0; col < NB; col++) begin
					if(idx == HIGH-1 - col*(col+1)/2)  inv = 1;
				end
				if(idx == HIGH)  inv = 1;
				if(inv)  op = { op[1:0], op[3:2] };
				if((idx == HIGH) && !SIGNED_A)  op = ~op;
			end

			if(SIGNED_A) begin
				automatic bit  inv = 0;
				// NAND along left matrix boundary
				for(int unsigned  col = 0; col < NB; col++) begin
					if(idx == HIGH - (col+1)*(col+2)/2 + (SIGNED_B && (col < NB-1)))  inv = 1;
				end
				if(inv)  op = ~op;
			end

			return  op;
		end
	endfunction : gate_op

	//-----------------------------------------------------------------------
	// Broadcast Wiring
	if(NA == 1) begin : genTrivial
		assign	oa[0] = ia[0];
		assign	ob[0] = ib[0];
	end : genTrivial
	begin : genMatrix

		// Feed right triangle going right to left until first full-height column
		for(genvar  col = 0; col < NB; col++) begin
			localparam int unsigned  TOP = col*(col+1)/2;
			for(genvar  row = 0; row <= col; row++) begin
				assign	oa[TOP+row] = ia[col-row];
				assign	ob[TOP+row] = ib[row];
			end
		end

		// Feed central full-height rectangle for NA > NB
		for(genvar  col = 0; col < NA-NB; col++) begin
			localparam int unsigned  TOP = NB*(NB+1)/2 + col*NB + SIGNED_B;
			for(genvar  row = 0; row < NB; row++) begin
				assign	oa[TOP + col*NB + row] = ia[NB+col - row];
				assign	ob[TOP + col*NB + row] = ib[row];
			end
		end

		// Feed left triangle going left to right up to last column with a receeded height
		for(genvar  col = 0; col < NB-1; col++) begin
			localparam int unsigned  BOT = HIGH - col*(col+1)/2 - 1;
			for(genvar  row = 0; row <= col; row++) begin
				assign	oa[BOT-row] = ia[NA-1-col+row];
				assign	ob[BOT-row] = ib[NB-1-row];
			end
		end

		// Feed extra elements created for sign handling
		if(SIGNED_B) begin
			assign	oa[NB*(NB+1)/2] = ia[0];
			assign	ob[NB*(NB+1)/2] = ib[NB-1];
			if(!SIGNED_A) begin
				assign	oa[HIGH] = 0;
				assign	ob[HIGH] = ib[NB-1];
			end
		end

	end : genMatrix

endinterface : mul_comp_map

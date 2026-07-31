/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module queue #(
	int unsigned  DATA_WIDTH,
	int unsigned  ELASTICITY
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);

	typedef logic [DATA_WIDTH-1:0]  dat_t;
	initial begin
		if(ELASTICITY < 2) begin
			$error("%m: ELASTICITY of %0d must be made 2 or above.", ELASTICITY);
			$finish;
		end
	end

	logic signed [$clog2(ELASTICITY):0]  Ptr = '1;	// -1, 0, 1, ..., ELASTICITY-1
	logic  Rdy = 1;
	dat_t  A[ELASTICITY];
	assign	irdy = Rdy;

	logic  Vld = 0;
	dat_t  B = 'x;
	assign	odat = B;
	assign	ovld = Vld;

	uwire  bload = !Vld || ordy;
	uwire  push = Rdy && ivld;
	uwire  pop = !Ptr[$left(Ptr)] && bload;

	always_ff @(posedge clk) begin
		if(push)  A <= { idat, A[0:ELASTICITY-2] };
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Ptr <= '1;
			Rdy <= 1;
			Vld <= 0;
			B <= 'x;
		end
		else begin
			// Make sure Rdy encodes what it's supposed to: space available in queue
			assert(Rdy == (Ptr < signed'(ELASTICITY-1))) else begin
				$error("%m: Broken Rdy computation.");
				$stop;
			end

			Ptr <= Ptr + ((push == pop)? 0 : push? 1 : -1);
			//  pop ==  push: no change
			//  pop && !push: new space
			// !pop &&  push: remaining space if not yet Ptr == ELASTICITY-2
			Rdy <= (pop == push)? Rdy : pop? 1 : Ptr[$left(Ptr)] || (((ELASTICITY-2) & ~Ptr[$left(Ptr)-1:0]) != 0);
			if(bload) begin
				Vld <= !Ptr[$left(Ptr)];
				B <= A[Ptr[$left(Ptr)-1:0]];
			end
		end
	end

endmodule : queue

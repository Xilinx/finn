/**
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief   Quick visualizer for compressor input broadcasting.
 */

module mul_comp_map_tb;
	localparam int unsigned  NA = 5;
	localparam int unsigned  NB = 4;
	localparam bit  SIGNED_A = 1;
	localparam bit  SIGNED_B = 1;
	logic [NA-1:0]  a;
	logic [NB-1:0]  b;
	mul_comp_map #(.NA(NA), .NB(NB), .SIGNED_A(SIGNED_A), .SIGNED_B(SIGNED_B)) map (.ia(a), .ib(b));

	initial begin
		automatic int unsigned  col = 0;
		automatic int unsigned  row = 0;
		a = '0;
		b = '1;

		#5ns;
		for(int unsigned  i = 0; i < $bits(map.oa); i++) begin
			$write("\t%0b.%0d.%0b", map.oa[i], map.gate_op(i), map.ob[i]);
			if(++row == map.height(col)) begin
				$display();
				col++;
				row = 0;
			end
		end
		$display("\t%0b", map.absolute_term());
	end

endmodule : mul_comp_map_tb

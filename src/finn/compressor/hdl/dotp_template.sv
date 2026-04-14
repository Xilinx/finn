module dotp_{n}x{sa}{na}{sb}{nb} #(
	localparam int unsigned  N = {n},
	localparam int unsigned  NA = {na},
	localparam int unsigned  NB = {nb},
	localparam bit  SIGNED_A = {signed_a},
	localparam bit  SIGNED_B = {signed_b},
	localparam int unsigned  NP = NA > 1?
		$clog2(N) + (!SIGNED_B && (NB == 1)? NA : NA+NB) :
		SIGNED_A ^^ SIGNED_B? 1 + $clog2(N) /*[-N:0]*/ : $clog2(N+1) /*[0:N]*/
)(
	input	logic  clk,

	input	logic [N-1:0][NA-1:0]  a,
	input	logic [N-1:0][NB-1:0]  b,
	output	logic [NP-1:0]  p
);

	// Input to Matrix Broadcasting
	mul_comp_map #(.NA(NA), .NB(NB), .SIGNED_A(SIGNED_A), .SIGNED_B(SIGNED_B)) map0 (.ia(a[0]), .ib(b[0]));
	localparam int unsigned  NM = $bits(map0.oa);
	uwire [NM-1:0]  oa[N];
	uwire [NM-1:0]  ob[N];
	assign	oa[0] = map0.oa;
	assign	ob[0] = map0.ob;
	for(genvar  i = 1; i < N; i++) begin
		mul_comp_map #(.NA(NA), .NB(NB), .SIGNED_A(SIGNED_A), .SIGNED_B(SIGNED_B)) map_i (.ia(a[i]), .ib(b[i]));
		assign	oa[i] = map_i.oa;
		assign	ob[i] = map_i.ob;
	end

	// Flatten all Matrices Column by Column
	logic [N*NM-1:0]  comp_a;
	logic [N*NM-1:0]  comp_b;
	always_comb begin
		automatic int unsigned  src_idx[N] = '{ default: 0 };
		automatic int unsigned  dst_idx = 0;
		for(int unsigned  col = 0; col < map0.columns(); col++) begin
			for(int unsigned  i = 0; i < N; i++) begin
				for(int unsigned  row = 0; row < map0.height(col); row++) begin
					comp_a[dst_idx] = oa[i][src_idx[i]];
					comp_b[dst_idx] = ob[i][src_idx[i]];
					src_idx[i]++;
					dst_idx++;
				end
			end
		end
	end

	uwire signed [NP-1:0]  comp_p;
	uwire signed [NP-1:0]  abs_p = {abs_term};
	comp_{n}x{sa}{na}{sb}{nb} comp (
		.clk,
		.in(comp_b), .in_2(comp_a),
		.out(comp_p)
	);
	assign	p = comp_p + abs_p;

endmodule : dotp_{n}x{sa}{na}{sb}{nb}

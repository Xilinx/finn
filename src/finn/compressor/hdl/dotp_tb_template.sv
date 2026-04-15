module dotp_{n}x{sa}{na}{sb}{nb}_tb #(
	localparam int unsigned  N = {n},
	localparam int unsigned  NA = {na},
	localparam int unsigned  NB = {nb},
	localparam bit  SIGNED_A = {signed_a},
	localparam bit  SIGNED_B = {signed_b},

	localparam int unsigned  NP = NA > 1?
		$clog2(N) + (!SIGNED_B && (NB == 1)? NA : NA+NB) :
		SIGNED_A ^^ SIGNED_B? 1 + $clog2(N) /*[-N:0]*/ : $clog2(N+1) /*[0:N]*/,
	localparam bit  SIGNED_P = NA == 1? SIGNED_A ^^ SIGNED_B : SIGNED_A || SIGNED_B
)();
	uwire  clk = 'z;

	logic [N-1:0][NA-1:0]  a;
	logic [N-1:0][NB-1:0]  b;
	uwire [NP-1:0]  p;

	dotp_{n}x{sa}{na}{sb}{nb} dut (
		.clk,
		.a, .b, .p
	);

	initial begin
		repeat(137) begin
			automatic type(a)  aa;
			automatic type(b)  bb;
			automatic int  pp = 0;
			automatic int  px;
			void'(std::randomize(aa, bb));
			for(int unsigned  i = 0; i < N; i++) begin
				automatic logic  sa = SIGNED_A && aa[i][NA-1];
				automatic logic  sb = SIGNED_B && bb[i][NB-1];
				pp += $signed({sa, aa[i]}) * $signed({sb, bb[i]});
			end

			a <= aa;
			b <= bb;
			#10ns;
			px = $signed({ SIGNED_P && p[NP-1], p });
			assert((^p !== 1'bx) && (px == pp)) else begin
				$error("Received %0d [0x%0x] instead of %0d.", px, p, pp);
				$stop;
			end
		end

		$display("Test completed.");
		$finish;
	end

endmodule : dotp_{n}x{sa}{na}{sb}{nb}_tb

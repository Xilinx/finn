module mux
#(
    parameter NINPUTS = 1,
	parameter WIDTH = 16
)
(
	input [NINPUTS*WIDTH-1:0] in,
	output [WIDTH-1:0] out,
	input [$clog2(NINPUTS)-1:0] sel
);

assign out = in >> (sel*WIDTH);

endmodule
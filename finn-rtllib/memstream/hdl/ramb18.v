module ramb18_wf_dualport
#(
    parameter ID = 0,
	parameter DWIDTH = 18,
	parameter AWIDTH = 10,
	parameter MEM_INIT = ""
)
(
	input clk,
	
	input wea,
	input [AWIDTH-1:0] addra,
	input [DWIDTH-1:0] wdataa,
	output reg [DWIDTH-1:0] rdqa,

	input web,
	input [AWIDTH-1:0] addrb,
	input [DWIDTH-1:0] wdatab,
	output reg [DWIDTH-1:0] rdqb
);

(* ram_style = "block" *) reg [DWIDTH-1:0] mem[0:2**AWIDTH-1];
reg [DWIDTH-1:0] rdataa;
reg [DWIDTH-1:0] rdatab;

reg [7:0] idx = ID;
//initialize memory
initial begin
    //note the hacky way of adding a filename memblock_ID.dat to the path provided in MEM_INIT
	//ID can go up to 99
	if (ID < 0 && ID > 99) begin
	    $display("ID out of range [0-99]");
	    $finish();
    end
	//MEM_INIT path must be terminated by /
	if (ID < 10)
		$readmemh({MEM_INIT,"memblock_",idx+8'd48,".dat"}, mem, 0, 1023);
	else
		$readmemh({MEM_INIT,"memblock_",(idx/10)+8'd48,(idx%10)+8'd48,".dat"}, mem, 0, 1023);
end

//memory ports, with output pipeline register
always @(posedge clk) begin
    if(wea)
        mem[addra] <= wdataa;
    rdataa <= mem[addra];
    rdqa <= rdataa;
end
always @(posedge clk) begin
    if(web)
        mem[addrb] <= wdatab;
    rdatab <= mem[addrb];
    rdqb <= rdatab;
end

endmodule
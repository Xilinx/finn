`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/22/2023 02:19:08 PM
// Design Name: 
// Module Name: harness_sink
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module vector_xor #(
    parameter WIDTH=8
)(
    input [WIDTH-1:0] in_data,
    output out_data
);

assign out_data = ^in_data;

endmodule

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


module harness_sink #(
    parameter STREAM_WIDTH=8
)(
    input enable,
    output valid,
    output checksum,
    input [STREAM_WIDTH-1:0] s_axis_0_tdata,
    input s_axis_0_tvalid,
    output s_axis_0_tready
);

assign s_axis_0_tready = enable;

assign valid = s_axis_0_tvalid;
assign checksum = ^s_axis_0_tdata;

endmodule

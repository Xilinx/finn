/*
 Copyright (c) 2020, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of FINN nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

module axi4lite_if
#(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,//AXI4 spec requires this to be strictly 32 or 64
    parameter IP_DATA_WIDTH = 64//can be any power-of-2 multiple of DATA_WIDTH
)
(
//system signals
input aclk,
input aresetn,//active low, asynchronous assertion and synchronous deassertion

//Write channels
//write address
output reg                  awready,
input                       awvalid,
input [ADDR_WIDTH-1:0]      awaddr,
input [2:0]                 awprot,
//write data
output reg                  wready,
input                       wvalid,
input [DATA_WIDTH-1:0]      wdata,
input [(DATA_WIDTH/8)-1:0]  wstrb,
//burst response
input                       bready,
output reg                  bvalid,
output reg [1:0]            bresp,//NOTE: 00 = OKAY, 10 = SLVERR (write error)

//Read channels
//read address
output reg                  arready,
input                       arvalid,
input [ADDR_WIDTH-1:0]      araddr,
input [2:0]                 arprot,
//read data
input                       rready,
output reg                  rvalid,
output reg [1:0]            rresp,//NOTE: 00 = OKAY, 10 = SLVERR (read error)
output reg [DATA_WIDTH-1:0] rdata,

//IP-side interface
output reg                  ip_en,
output reg                  ip_wen,
output reg [ADDR_WIDTH-1:0] ip_addr,
output [IP_DATA_WIDTH-1:0]  ip_wdata,
input                       ip_rack,
input [IP_DATA_WIDTH-1:0]      ip_rdata
);

localparam RESP_OKAY = 2'b00;
localparam RESP_SLVERR = 2'b10;
//get ceil(log2(ceil(IP_DATA_WIDTH/DATA_WIDTH)))
localparam NFOLDS_LOG = $clog2((IP_DATA_WIDTH + DATA_WIDTH - 1) / DATA_WIDTH);

reg                      internal_ren;
reg                      internal_wen;
reg                      internal_wack;
reg [ADDR_WIDTH-1:0]     internal_raddr;
reg [ADDR_WIDTH-1:0]     internal_waddr;
reg [DATA_WIDTH-1:0]     internal_wdata;
wire [DATA_WIDTH-1:0]    internal_rdata;
reg                      internal_error = 0;

//check DATA_WIDTH
initial begin
    if(DATA_WIDTH != 32 & DATA_WIDTH != 64) begin
        $display("AXI4Lite DATA_WIDTH must be 32 or 64");
        $finish;
    end
end

//transaction state machine
localparam  STATE_IDLE  = 0,
            STATE_READ  = 1,
            STATE_WRITE = 2;

reg [1:0] state;

always @(posedge aclk or negedge aresetn)
    if(~aresetn)
        state <= STATE_IDLE;
    else case(state)
        STATE_IDLE:
            if(awvalid & wvalid)
                state <= STATE_WRITE;
            else if(arvalid)
                state <= STATE_READ;
        STATE_READ:
            if(rvalid & rready)
                state <= STATE_IDLE;
        STATE_WRITE:
            if(bvalid & bready)
                state <= STATE_IDLE;
        default: state <= STATE_IDLE;
    endcase

//write-related internal signals
always @(*) begin
    internal_waddr = awaddr >> $clog2(DATA_WIDTH/8);
    internal_wdata = wdata;
    internal_wen = (state == STATE_IDLE) & awvalid & wvalid; 
end

always @(posedge aclk) begin
    awready <= internal_wen;
    wready <= internal_wen;
end

//read-related internal signals
always @(*) begin
    internal_raddr = araddr >> $clog2(DATA_WIDTH/8);
    internal_ren = (state == STATE_IDLE) & ~internal_wen & arvalid;
end

always @(posedge aclk)
    arready <= internal_ren;

wire write_to_last_fold;

always @(posedge aclk) begin
    ip_wen <= write_to_last_fold;
    ip_en <= internal_ren | write_to_last_fold;
    if(internal_ren | write_to_last_fold)
        ip_addr <= internal_ren ? (internal_raddr >> NFOLDS_LOG) : (internal_waddr >> NFOLDS_LOG);
    internal_wack <= internal_wen;
end

genvar i;
reg [(1<<NFOLDS_LOG)*DATA_WIDTH-1:0] ip_wdata_wide;
generate
if(NFOLDS_LOG == 0) begin: no_fold
    assign write_to_last_fold = internal_wen;
    assign internal_rdata = ip_rdata;
    always @(posedge aclk)
        ip_wdata_wide <= internal_wdata;
end else begin: fold
    reg [NFOLDS_LOG-1:0] internal_rfold;
    assign write_to_last_fold = internal_wen & (internal_waddr[NFOLDS_LOG-1:0] == {(NFOLDS_LOG){1'b1}});
    assign internal_rdata = ip_rdata >> (internal_rfold*DATA_WIDTH);
    for(i=0; i<(1<<NFOLDS_LOG); i = i+1) begin: gen_wdata
        always @(posedge aclk) begin
            if(internal_waddr[NFOLDS_LOG-1:0] == i)
                ip_wdata_wide[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] <= internal_wdata;
            if(internal_ren)
                internal_rfold <= internal_raddr[NFOLDS_LOG-1:0];
        end
    end
end
endgenerate
assign ip_wdata = ip_wdata_wide[IP_DATA_WIDTH-1:0];

//write response on AXI4L bus
always @(posedge aclk or negedge aresetn)
    if(~aresetn) begin
        bvalid <= 0;//AXI4 spec requires BVALID pulled LOW during reset
        bresp <= RESP_OKAY;
    end else if(internal_wack) begin
        bvalid <= 1;
        bresp <= internal_error ? RESP_SLVERR : RESP_OKAY;
    end else if(bready) begin
        bvalid <= 0;
        bresp <= RESP_OKAY;
    end

//read response on AXI4L bus
always @(posedge aclk or negedge aresetn)
    if(~aresetn) begin
        rvalid <= 0;//AXI4 spec requires RVALID pulled LOW during reset
        rdata <= 0;
        rresp <= RESP_OKAY;
    end else if(ip_rack) begin
        rvalid <= 1;
        rdata <= internal_rdata;
        rresp <= internal_error ? RESP_SLVERR : RESP_OKAY;
    end else if(rready) begin
        rvalid <= 0;
        rdata <= 0;
        rresp <= RESP_OKAY;
    end

endmodule


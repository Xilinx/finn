// Copyright (c) 2023 Advanced Micro Devices, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of AMD nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


module tb #(
	// sampling period (in cycles) for reading instrumentation wrapper registers
	int unsigned  INSTR_READ_PERIOD = 1000,
    // 16-bit LFSR seed for generating fixed random data
    int unsigned LFSR_SEED = 1
)();


// Clock & Reset
logic  ap_clk = 0;
always #5ns ap_clk = !ap_clk;
logic  ap_rst_n = 0;
uwire  ap_rst = !ap_rst_n;

// wires for instrumentation wrapper AXI lite interface
logic [31:0] axilite_ctrl_araddr = 'x;
uwire axilite_ctrl_arready;
logic axilite_ctrl_arvalid = 0;
logic [31:0]  axilite_ctrl_awaddr = 'x;
uwire axilite_ctrl_awready;
logic axilite_ctrl_awvalid = 0;
uwire axilite_ctrl_bready = 1;
uwire [1:0]axilite_ctrl_bresp;
uwire axilite_ctrl_bvalid;
uwire [31:0]axilite_ctrl_rdata;
logic axilite_ctrl_rready = 1;
uwire [1:0]axilite_ctrl_rresp;
uwire axilite_ctrl_rvalid;
logic [31:0]  axilite_ctrl_wdata = 'x;
uwire axilite_ctrl_wready;
uwire [3:0]axilite_ctrl_wstrb = 4'b1111;
logic  axilite_ctrl_wvalid = 0;




dut_wrapper dut_wrapper_inst (
	.ap_clk_0(ap_clk), .ap_rst_n_0(ap_rst_n),
    .s_axi_ctrl_0_araddr(axilite_ctrl_araddr),
    .s_axi_ctrl_0_arready(axilite_ctrl_arready),
    .s_axi_ctrl_0_arvalid(axilite_ctrl_arvalid),
    .s_axi_ctrl_0_awaddr(axilite_ctrl_awaddr),
    .s_axi_ctrl_0_awready(axilite_ctrl_awready),
    .s_axi_ctrl_0_awvalid(axilite_ctrl_awvalid),
    .s_axi_ctrl_0_bready(axilite_ctrl_bready),
    .s_axi_ctrl_0_bresp(axilite_ctrl_bresp),
    .s_axi_ctrl_0_bvalid(axilite_ctrl_bvalid),
    .s_axi_ctrl_0_rdata(axilite_ctrl_rdata),
    .s_axi_ctrl_0_rready(axilite_ctrl_rready),
    .s_axi_ctrl_0_rresp(axilite_ctrl_rresp),
    .s_axi_ctrl_0_rvalid(axilite_ctrl_rvalid),
    .s_axi_ctrl_0_wdata(axilite_ctrl_wdata),
    .s_axi_ctrl_0_wready(axilite_ctrl_wready),
    .s_axi_ctrl_0_wstrb(axilite_ctrl_wstrb),
    .s_axi_ctrl_0_wvalid(axilite_ctrl_wvalid)
);

//---------------------------------------------------------------------------

initial begin
	$timeformat(-9, 2, " ns");
	// perform reset
	repeat(100)  @(posedge ap_clk);
	ap_rst_n <= 1;
	$display("Reset complete");
    repeat(100) @(posedge ap_clk);
    // instrumentation wrapper configuration:
    // set up LFSR seed + start data generation + output sink
    axilite_ctrl_awaddr  <= 'h10;
    axilite_ctrl_awvalid <= 1;
    axilite_ctrl_wdata   <= (LFSR_SEED << 16) | 'b11;
    axilite_ctrl_wvalid  <= 1;
    repeat(8) begin
        @(posedge ap_clk);
        if(axilite_ctrl_wready && axilite_ctrl_awready)  break;
    end
    axilite_ctrl_wvalid  <= 0;
    axilite_ctrl_awvalid <= 0;
    axilite_ctrl_awaddr  <= 'x;
    axilite_ctrl_wdata   <= 'x;
    while(1) begin
        axilite_ctrl_araddr  <= 'h18;
        axilite_ctrl_arvalid <= 1;
        repeat(8) begin
            @(posedge ap_clk);
            if(axilite_ctrl_rvalid) begin
                $display("[t=%0t] STATUS_I = %0d", $time, axilite_ctrl_rdata);
                break;
            end
        end
        axilite_ctrl_araddr  <= 'h20;
        axilite_ctrl_arvalid <= 1;
        repeat(8) begin
            @(posedge ap_clk);
            if(axilite_ctrl_rvalid) begin
                $display("[t=%0t] STATUS_O = %0d", $time, axilite_ctrl_rdata);
                break;
            end
        end
        axilite_ctrl_araddr  <= 'h28;
        axilite_ctrl_arvalid <= 1;
        repeat(8) begin
            @(posedge ap_clk);
            if(axilite_ctrl_rvalid) begin
                $display("[t=%0t] LATENCY = %0d", $time, axilite_ctrl_rdata);
                break;
            end
        end
        axilite_ctrl_araddr  <= 'h38;
        axilite_ctrl_arvalid <= 1;
        repeat(8) begin
            @(posedge ap_clk);
            if(axilite_ctrl_rvalid) begin
                $display("[t=%0t] INTERVAL = %0d", $time, axilite_ctrl_rdata);
                break;
            end
        end
        axilite_ctrl_araddr  <= 'h48;
        axilite_ctrl_arvalid <= 1;
        repeat(8) begin
            @(posedge ap_clk);
            if(axilite_ctrl_rvalid) begin
                $display("[t=%0t] CHECKSUM = %0x", $time, axilite_ctrl_rdata);
                if(axilite_ctrl_rdata) begin
                    $display("Nonzero checksum detected, stopping simulation");
                    $finish;
                end
                break;
            end
        end
        axilite_ctrl_arvalid <= 0;
        repeat(INSTR_READ_PERIOD)  @(posedge ap_clk);
    end
end


endmodule : tb

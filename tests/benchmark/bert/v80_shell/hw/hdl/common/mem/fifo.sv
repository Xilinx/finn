/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF    SUCH DAMAGE.
  */


`timescale 1ns / 1ps

module fifo #(
	parameter integer DATA_BITS = 64,
	parameter integer FIFO_SIZE = 8
) (
	input  logic 					aclk,
	input  logic 					aresetn,

	input  logic 					rd,
	input  logic 					wr,

	output logic					ready_rd,
	output logic 					ready_wr,

	input  logic [DATA_BITS-1:0] 	data_in,
	output logic [DATA_BITS-1:0]	data_out
);

// Constants
localparam integer PNTR_BITS = $clog2(FIFO_SIZE);

// Internal registers
logic [PNTR_BITS-1:0] wr_pntr;
logic [PNTR_BITS-1:0] rd_pntr;
logic [PNTR_BITS:0] n_entries;

logic isFull;
logic isEmpty;

logic [FIFO_SIZE-1:0][DATA_BITS-1:0] data;

// FIFO flags
assign isFull = (n_entries == FIFO_SIZE);
assign isEmpty = (n_entries == 0);

genvar i;

always_ff @(posedge aclk) begin
	if(aresetn == 1'b0) begin
		 n_entries <= 0;
		 data <= 0;
	end else begin
		 // Number of entries
		 if (rd && !isEmpty && (!wr || isFull))
		 	n_entries <= n_entries - 1;
		 else if (wr && !isFull && (!rd || isEmpty))
		 	n_entries <= n_entries + 1;
		 // Data
		 if(wr && !isFull)
		 	data[wr_pntr] <= data_in;
	end
end

always_ff @(posedge aclk) begin
	if(aresetn == 1'b0) begin
		rd_pntr <= 0;
		wr_pntr <= 0;
	end else begin
		// Write pointer
		if(wr && !isFull) begin
			if(wr_pntr == (FIFO_SIZE-1))
				wr_pntr <= 0;
			else
				wr_pntr <= wr_pntr + 1;
		end
		// Read pointer
		if(rd && !isEmpty) begin
			if(rd_pntr == (FIFO_SIZE-1))
				rd_pntr <= 0;
			else 
				rd_pntr <= rd_pntr + 1;
		end
	end
end

// Output
assign ready_rd = ~isEmpty;
assign ready_wr = ~isFull;

assign data_out = data[rd_pntr];

endmodule // fifo
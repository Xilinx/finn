/******************************************************************************
 * Copyright (C) 2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	AXI Stream Adapter for Data Width Converter.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/
module dwc_axi #(
	int unsigned  IBITS,
	int unsigned  OBITS,

	localparam int unsigned  AXI_IBITS = (IBITS+7)/8 * 8,
	localparam int unsigned  AXI_OBITS = (OBITS+7)/8 * 8
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [AXI_IBITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [AXI_OBITS-1:0]  m_axis_tdata
);

	dwc #(.IBITS(IBITS), .OBITS(OBITS)) core (
		.clk(ap_clk), .rst(!ap_rst_n),
		.irdy(s_axis_tready), .ivld(s_axis_tvalid), .idat(s_axis_tdata[IBITS-1:0]),
		.ordy(m_axis_tready), .ovld(m_axis_tvalid), .odat(m_axis_tdata[OBITS-1:0])
	);
	if(OBITS < AXI_OBITS) begin
		assign	m_axis_tdata[AXI_OBITS-1:OBITS] = '0;
	end

endmodule : dwc_axi

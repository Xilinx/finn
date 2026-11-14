/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Wiring-only pass-thru AXI-Stream connector.
 *
 * Connects an AXI-Stream input to an AXI-Stream output without any logic or
 * registers. It gives operators that only re-interpret the shape of a stream
 * (reshape, flatten, squeeze, unsqueeze) a hardware node of their own at no
 * resource or latency cost.
 ***************************************************************************/

module passthru_axi #(
	int unsigned  DATA_WIDTH
)(
	input	logic [DATA_WIDTH-1:0]  s_axis_tdata,
	input	logic  s_axis_tvalid,
	output	logic  s_axis_tready,

	output	logic [DATA_WIDTH-1:0]  m_axis_tdata,
	output	logic  m_axis_tvalid,
	input	logic  m_axis_tready
);
`default_nettype none

	assign	m_axis_tdata  = s_axis_tdata;
	assign	m_axis_tvalid = s_axis_tvalid;
	assign	s_axis_tready = m_axis_tready;

`default_nettype wire
endmodule : passthru_axi

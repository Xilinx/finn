// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (inclu	ding loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

import iwTypes::*;

`include "axi_macros.svh"

module hbm_offsets (
    input  logic                                        aclk,
    input  logic                                        aresetn,

    AXI4.slave                                          s_axi_hbm [N_HBM_PORTS],
    AXI4.master                                         m_axi_hbm [N_HBM_PORTS]
);

typedef logic [HBM_ADDR_BITS-1:0] addr_t;
addr_t ch_base [N_HBM_PORTS];
addr_t rng_mask = addr_t'(HBM_RNG) - addr_t'(1);

for (genvar gi = 0; gi < N_HBM_PORTS; gi++) begin
	assign ch_base[gi] = addr_t'(HBM_OFFS) + addr_t'(gi) * addr_t'(HBM_RNG);
end

for (genvar gj = 0; gj < N_HBM_PORTS; gj++) begin
	addr_t ar_off = addr_t'(s_axi_hbm[gj].araddr) & rng_mask;
	addr_t aw_off = addr_t'(s_axi_hbm[gj].awaddr) & rng_mask;

	assign m_axi_hbm[gj].araddr  = ch_base[gj] | ar_off;
	assign m_axi_hbm[gj].awaddr  = ch_base[gj] | aw_off;

	assign m_axi_hbm[gj].arburst = s_axi_hbm[gj].arburst;
	assign m_axi_hbm[gj].arcache = s_axi_hbm[gj].arcache;
	assign m_axi_hbm[gj].arid    = s_axi_hbm[gj].arid;
	assign m_axi_hbm[gj].arlen   = s_axi_hbm[gj].arlen;
	assign m_axi_hbm[gj].arlock  = s_axi_hbm[gj].arlock;
	assign m_axi_hbm[gj].arprot  = s_axi_hbm[gj].arprot;
	assign m_axi_hbm[gj].arsize  = s_axi_hbm[gj].arsize;
	assign m_axi_hbm[gj].arvalid = s_axi_hbm[gj].arvalid;
	assign s_axi_hbm[gj].arready = m_axi_hbm[gj].arready;

	assign m_axi_hbm[gj].awburst = s_axi_hbm[gj].awburst;
	assign m_axi_hbm[gj].awcache = s_axi_hbm[gj].awcache;
	assign m_axi_hbm[gj].awid    = s_axi_hbm[gj].awid;
	assign m_axi_hbm[gj].awlen   = s_axi_hbm[gj].awlen;
	assign m_axi_hbm[gj].awlock  = s_axi_hbm[gj].awlock;
	assign m_axi_hbm[gj].awprot  = s_axi_hbm[gj].awprot;
	assign m_axi_hbm[gj].awsize  = s_axi_hbm[gj].awsize;
	assign m_axi_hbm[gj].awvalid = s_axi_hbm[gj].awvalid;
	assign s_axi_hbm[gj].awready = m_axi_hbm[gj].awready;

	assign s_axi_hbm[gj].rdata   = m_axi_hbm[gj].rdata;
	assign s_axi_hbm[gj].rid     = m_axi_hbm[gj].rid;
	assign s_axi_hbm[gj].rlast   = m_axi_hbm[gj].rlast;
	assign s_axi_hbm[gj].rresp   = m_axi_hbm[gj].rresp;
	assign m_axi_hbm[gj].rready  = s_axi_hbm[gj].rready;
	assign s_axi_hbm[gj].rvalid  = m_axi_hbm[gj].rvalid;

	assign m_axi_hbm[gj].wdata   = s_axi_hbm[gj].wdata;
	assign m_axi_hbm[gj].wlast   = s_axi_hbm[gj].wlast;
	assign m_axi_hbm[gj].wstrb   = s_axi_hbm[gj].wstrb;
	assign s_axi_hbm[gj].wready  = m_axi_hbm[gj].wready;
	assign m_axi_hbm[gj].wvalid  = s_axi_hbm[gj].wvalid;

	assign s_axi_hbm[gj].bid     = m_axi_hbm[gj].bid;
	assign s_axi_hbm[gj].bresp   = m_axi_hbm[gj].bresp;
	assign m_axi_hbm[gj].bready  = s_axi_hbm[gj].bready;
	assign s_axi_hbm[gj].bvalid  = m_axi_hbm[gj].bvalid;
end
    

endmodule

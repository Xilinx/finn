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
  
import iwTypes::*;

`include "axi_macros.svh"

module axi_hbm_reg (
	input logic 			          aclk,
	input logic 			          aresetn,
	
	AXI4.slave  		            s_axi,
	AXI4.master			            m_axi
);

  axi_hbm_register_slice_256 (
      .aclk(aclk),
      .aresetn(aresetn),
      .s_axi_awid(s_axi.awid),
      .s_axi_awaddr(s_axi.awaddr),
      .s_axi_awlen(s_axi.awlen),
      .s_axi_awsize(s_axi.awsize),
      .s_axi_awburst(s_axi.awburst),
      .s_axi_awlock(s_axi.awlock),
      .s_axi_awcache(s_axi.awcache),
      .s_axi_awprot(s_axi.awprot),
      .s_axi_awvalid(s_axi.awvalid),
      .s_axi_awready(s_axi.awready),
      .s_axi_arid(s_axi.arid),
      .s_axi_araddr(s_axi.araddr),
      .s_axi_arlen(s_axi.arlen),
      .s_axi_arsize(s_axi.arsize),
      .s_axi_arburst(s_axi.arburst),
      .s_axi_arlock(s_axi.arlock),
      .s_axi_arcache(s_axi.arcache),
      .s_axi_arprot(s_axi.arprot),
      .s_axi_arvalid(s_axi.arvalid),
      .s_axi_arready(s_axi.arready),
      .s_axi_wdata(s_axi.wdata),
      .s_axi_wstrb(s_axi.wstrb),
      .s_axi_wlast(s_axi.wlast),
      .s_axi_wvalid(s_axi.wvalid),
      .s_axi_wready(s_axi.wready),
      .s_axi_bid(s_axi.bid),
      .s_axi_bresp(s_axi.bresp),
      .s_axi_bvalid(s_axi.bvalid),
      .s_axi_bready(s_axi.bready),
      .s_axi_rid(s_axi.rid),
      .s_axi_rdata(s_axi.rdata),
      .s_axi_rresp(s_axi.rresp),
      .s_axi_rlast(s_axi.rlast),
      .s_axi_rvalid(s_axi.rvalid),
      .s_axi_rready(s_axi.rready),
      .m_axi_awid(m_axi.awid),
      .m_axi_awaddr(m_axi.awaddr),
      .m_axi_awlen(m_axi.awlen),
      .m_axi_awsize(m_axi.awsize),
      .m_axi_awburst(m_axi.awburst),
      .m_axi_awlock(m_axi.awlock),
      .m_axi_awcache(m_axi.awcache),
      .m_axi_awprot(m_axi.awprot),
      .m_axi_awvalid(m_axi.awvalid),
      .m_axi_awready(m_axi.awready),
      .m_axi_arid(m_axi.arid),
      .m_axi_araddr(m_axi.araddr),
      .m_axi_arlen(m_axi.arlen),
      .m_axi_arsize(m_axi.arsize),
      .m_axi_arburst(m_axi.arburst),
      .m_axi_arlock(m_axi.arlock),
      .m_axi_arcache(m_axi.arcache),
      .m_axi_arprot(m_axi.arprot),
      .m_axi_arvalid(m_axi.arvalid),
      .m_axi_arready(m_axi.arready),
      .m_axi_wdata(m_axi.wdata),
      .m_axi_wstrb(m_axi.wstrb),
      .m_axi_wlast(m_axi.wlast),
      .m_axi_wvalid(m_axi.wvalid),
      .m_axi_wready(m_axi.wready),
      .m_axi_bid(m_axi.bid),
      .m_axi_bresp(m_axi.bresp),
      .m_axi_bvalid(m_axi.bvalid),
      .m_axi_bready(m_axi.bready),
      .m_axi_rid(m_axi.rid),
      .m_axi_rdata(m_axi.rdata),
      .m_axi_rresp(m_axi.rresp),
      .m_axi_rlast(m_axi.rlast),
      .m_axi_rvalid(m_axi.rvalid),
      .m_axi_rready(m_axi.rready)
  );
  
endmodule
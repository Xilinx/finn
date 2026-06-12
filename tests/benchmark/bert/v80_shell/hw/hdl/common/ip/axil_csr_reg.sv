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

module axil_csr_reg (
	input logic 			    aclk,
	input logic 			    aresetn,
	
	AXI4L.slave 		      s_axi,
	AXI4L.master          m_axi
);

  axil_csr_register_slice_64 (
      .aclk(aclk),
      .aresetn(aresetn),
      .s_axi_awaddr(s_axi.awaddr),
      .s_axi_awvalid(s_axi.awvalid),
      .s_axi_awready(s_axi.awready),
      .s_axi_araddr(s_axi.araddr),
      .s_axi_arvalid(s_axi.arvalid),
      .s_axi_arready(s_axi.arready),
      .s_axi_wdata(s_axi.wdata),
      .s_axi_wstrb(s_axi.wstrb),
      .s_axi_wvalid(s_axi.wvalid),
      .s_axi_wready(s_axi.wready),
      .s_axi_bresp(s_axi.bresp),
      .s_axi_bvalid(s_axi.bvalid),
      .s_axi_bready(s_axi.bready),
      .s_axi_rdata(s_axi.rdata),
      .s_axi_rresp(s_axi.rresp),
      .s_axi_rvalid(s_axi.rvalid),
      .s_axi_rready(s_axi.rready),
      .m_axi_awaddr(m_axi.awaddr),
      .m_axi_awvalid(m_axi.awvalid),
      .m_axi_awready(m_axi.awready),
      .m_axi_araddr(m_axi.araddr),
      .m_axi_arvalid(m_axi.arvalid),
      .m_axi_arready(m_axi.arready),
      .m_axi_wdata(m_axi.wdata),
      .m_axi_wstrb(m_axi.wstrb),
      .m_axi_wvalid(m_axi.wvalid),
      .m_axi_wready(m_axi.wready),
      .m_axi_bresp(m_axi.bresp),
      .m_axi_bvalid(m_axi.bvalid),
      .m_axi_bready(m_axi.bready),
      .m_axi_rdata(m_axi.rdata),
      .m_axi_rresp(m_axi.rresp),
      .m_axi_rvalid(m_axi.rvalid),
      .m_axi_rready(m_axi.rready)
  );

endmodule
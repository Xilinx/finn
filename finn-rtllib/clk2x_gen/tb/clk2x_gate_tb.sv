/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench of the interface gates of the internal clock generator.
 * @details
 *	Drives the stitched IP finn_design_wrapper built by the FINN test
 *	test_clk2x_gen_gates_axi_lite_and_mm (tests/fpgadataflow/
 *	test_fpgadataflow_clk2x_gen.py): an input and an output IODMA with
 *	their AXI-Lite slaves s_axi_control_0/1 and AXI-MM masters
 *	m_axi_gmem0/1, and the signature registers s_axilite_info. The
 *	stitched IP derives its clocks from ap_clk by the generator and
 *	releases its design reset only after the generator has locked.
 *
 *	While the design is in reset, the testbench offers transfers on all
 *	AXI-Lite slaves and readiness plus spurious responses on the AXI-MM
 *	masters. It checks that no handshake is accepted or issued: no READY
 *	or response VALID on the slaves, no VALID or READY on the masters.
 *	The offered transfers stall and complete once the design is out of
 *	reset (OKAY, read back). The check is repeated for a second reset.
 ***************************************************************************/

interface axil_if (input logic  clk);
	logic [31:0]  awaddr;
	logic  awvalid;
	logic  awready;
	logic [31:0]  wdata;
	logic [ 3:0]  wstrb;
	logic  wvalid;
	logic  wready;
	logic [ 1:0]  bresp;
	logic  bvalid;
	logic  bready;
	logic [31:0]  araddr;
	logic  arvalid;
	logic  arready;
	logic [31:0]  rdata;
	logic [ 1:0]  rresp;
	logic  rvalid;
	logic  rready;
endinterface : axil_if

interface axim_if (input logic  clk);
	logic  arvalid;
	logic  arready;
	logic  rvalid;
	logic  rready;
	logic  awvalid;
	logic  awready;
	logic  wvalid;
	logic  wready;
	logic  bvalid;
	logic  bready;
endinterface : axim_if

module clk2x_gate_tb;

	localparam time  T = 10ns;	// period the generator is configured for

	logic  clk = 0;
	always #(T/2)  clk = !clk;
	logic  ap_rst_n = 0;

	axil_if  ctl0(clk);
	axil_if  ctl1(clk);
	axil_if  info(clk);
	axim_if  gmem0(clk);
	axim_if  gmem1(clk);

	//-----------------------------------------------------------------------
	// DUT
`define AXIL(PORT, IF) \
		.PORT``_awaddr(IF.awaddr), .PORT``_awvalid(IF.awvalid), .PORT``_awready(IF.awready), \
		.PORT``_wdata(IF.wdata), .PORT``_wstrb(IF.wstrb), .PORT``_wvalid(IF.wvalid), \
		.PORT``_wready(IF.wready), .PORT``_bresp(IF.bresp), .PORT``_bvalid(IF.bvalid), \
		.PORT``_bready(IF.bready), .PORT``_araddr(IF.araddr), .PORT``_arvalid(IF.arvalid), \
		.PORT``_arready(IF.arready), .PORT``_rdata(IF.rdata), .PORT``_rresp(IF.rresp), \
		.PORT``_rvalid(IF.rvalid), .PORT``_rready(IF.rready)
`define AXIM(PORT, IF) \
		.PORT``_arvalid(IF.arvalid), .PORT``_arready(IF.arready), .PORT``_rvalid(IF.rvalid), \
		.PORT``_rready(IF.rready), .PORT``_rdata('0), .PORT``_rresp(2'b00), .PORT``_rlast(1'b1), \
		.PORT``_awvalid(IF.awvalid), .PORT``_awready(IF.awready), .PORT``_wvalid(IF.wvalid), \
		.PORT``_wready(IF.wready), .PORT``_bvalid(IF.bvalid), .PORT``_bready(IF.bready), \
		.PORT``_bresp(2'b00)

	finn_design_wrapper dut (
		.ap_clk(clk), .ap_rst_n(ap_rst_n), .sim_finish(1'b0),
		`AXIL(s_axi_control_0, ctl0), `AXIL(s_axi_control_1, ctl1), `AXIL(s_axilite_info, info),
		`AXIM(m_axi_gmem0, gmem0), `AXIM(m_axi_gmem1, gmem1)
	);

	// the design reset behind the gates
	uwire  design_rst_n = dut.finn_design_i.clk2x_domain.rst_n;

	//-----------------------------------------------------------------------
	// Checker: no handshake while the design is in reset
	int unsigned  ErrCnt = 0;
	int unsigned  GatedCycles = 0;
	initial forever @(posedge clk) begin
		if(!design_rst_n) begin
			if(ap_rst_n)  GatedCycles++;
			if({
				ctl0.awready, ctl0.wready, ctl0.arready, ctl0.bvalid, ctl0.rvalid,
				ctl1.awready, ctl1.wready, ctl1.arready, ctl1.bvalid, ctl1.rvalid,
				info.awready, info.wready, info.arready, info.bvalid, info.rvalid,
				gmem0.arvalid, gmem0.rready, gmem0.awvalid, gmem0.wvalid, gmem0.bready,
				gmem1.arvalid, gmem1.rready, gmem1.awvalid, gmem1.wvalid, gmem1.bready
			} !== '0) begin
				$error("Handshake signal asserted while the design is in reset.");
				ErrCnt++;
			end
			if({
				dut.finn_design_i.clk2x_gate_s_axi_control_0.inner_awvalid,
				dut.finn_design_i.clk2x_gate_s_axi_control_0.inner_arvalid,
				dut.finn_design_i.clk2x_gate_m_axi_gmem0.inner_rvalid,
				dut.finn_design_i.clk2x_gate_m_axi_gmem1.inner_bvalid
			} !== '0) begin
				$error("Transfer reaches the design while it is in reset.");
				ErrCnt++;
			end
		end
	end

	//-----------------------------------------------------------------------
	// AXI-Lite master transfers, may be started before the design is ready
	task automatic axil_write(virtual axil_if  m, input logic [31:0]  addr, input logic [31:0]  data);
		bit  aw_done = 0;
		bit  w_done = 0;
		m.awaddr  <= addr;
		m.awvalid <= 1;
		m.wdata   <= data;
		m.wstrb   <= '1;
		m.wvalid  <= 1;
		while(!(aw_done && w_done)) begin
			@(posedge clk);
			if(m.awvalid && m.awready) begin
				aw_done = 1;
				m.awvalid <= 0;
			end
			if(m.wvalid && m.wready) begin
				w_done = 1;
				m.wvalid <= 0;
			end
		end
		do @(posedge clk); while(!m.bvalid);
		assert(m.bresp == 2'b00) else begin
			$error("Write to %0h: response %0d.", addr, m.bresp);
			ErrCnt++;
		end
	endtask : axil_write

	task automatic axil_read(virtual axil_if  m, input logic [31:0]  addr, output logic [31:0]  data);
		m.araddr  <= addr;
		m.arvalid <= 1;
		do @(posedge clk); while(!m.arready);
		m.arvalid <= 0;
		while(!m.rvalid) @(posedge clk);
		data = m.rdata;
		assert(m.rresp == 2'b00) else begin
			$error("Read from %0h: response %0d.", addr, m.rresp);
			ErrCnt++;
		end
		@(posedge clk);
	endtask : axil_read

	task automatic check_readback(virtual axil_if  m, input logic [31:0]  addr, input logic [31:0]  expected);
		logic [31:0]  data;
		axil_read(m, addr, data);
		assert(data == expected) else begin
			$error("Read back %0h from %0h, expected %0h.", data, addr, expected);
			ErrCnt++;
		end
	endtask : check_readback

	//-----------------------------------------------------------------------
	// Stimulus
	localparam logic [31:0]  ADDR = 'h10;	// IODMA: low word of the buffer address

	initial begin
		{ctl0.awvalid, ctl0.wvalid, ctl0.arvalid} = '0;
		{ctl1.awvalid, ctl1.wvalid, ctl1.arvalid} = '0;
		{info.awvalid, info.wvalid, info.arvalid} = '0;
		{ctl0.bready, ctl0.rready, ctl1.bready, ctl1.rready, info.bready, info.rready} = '1;
		// memory side: always ready, spurious responses during reset
		{gmem0.arready, gmem0.awready, gmem0.wready} = '1;
		{gmem1.arready, gmem1.awready, gmem1.wready} = '1;
		{gmem0.rvalid, gmem0.bvalid, gmem1.rvalid, gmem1.bvalid} = '1;

		// First reset and lock: transfers offered while ap_rst_n is still asserted
		repeat(8) @(posedge clk);
		fork
			axil_write(ctl0, ADDR, 32'hCAFE_0000);
			begin
				logic [31:0]  sig;
				axil_read(info, 0, sig);
			end
			begin
				repeat(8) @(posedge clk);
				ap_rst_n <= 1;
				// stop the spurious responses before the design can leave reset
				repeat(4) @(posedge clk);
				{gmem0.rvalid, gmem0.bvalid, gmem1.rvalid, gmem1.bvalid} <= '0;
			end
		join
		assert(design_rst_n) else begin
			$error("Transfers completed before the design left reset.");
			ErrCnt++;
		end
		check_readback(ctl0, ADDR, 32'hCAFE_0000);

		// Second reset: a write in flight across it
		repeat(4) @(posedge clk);
		fork
			axil_write(ctl1, ADDR, 32'hBEEF_0000);
			begin
				ap_rst_n <= 0;
				repeat(6) @(posedge clk);
				ap_rst_n <= 1;
			end
		join
		check_readback(ctl1, ADDR, 32'hBEEF_0000);

		assert(GatedCycles >= 2*16) else begin
			$error("Only %0d cycles observed with the design in reset.", GatedCycles);
			ErrCnt++;
		end
		$display("Observed %0d cycles with gated interfaces.", GatedCycles);
		if(ErrCnt == 0)  $display("clk2x_gate_tb PASSED");
		else             $display("clk2x_gate_tb FAILED: %0d errors", ErrCnt);
		$finish;
	end

	initial begin
		#(10000*T);
		$display("clk2x_gate_tb FAILED: timeout");
		$finish;
	end

endmodule : clk2x_gate_tb

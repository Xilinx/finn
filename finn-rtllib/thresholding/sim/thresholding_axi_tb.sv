/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *	 this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *	 contributors may be used to endorse or promote products derived from
 *	 this software without specific prior written permission.
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
 * @brief	Testbench for thresholding_axi.
 * @author	Monica Chiosa <monica.chiosa@amd.com>
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *****************************************************************************/

module thresholding_axi_tb #(
	int unsigned  N  = 14,	// number of thresholds
	int unsigned  C  =  6,	// number of channels
	int unsigned  PE =  2,
	real  M0 = 7.3,			// slope of the uniform thresholding line
	real  B0 = 3.1,			// offset of the uniform thresholding line
	bit  THROTTLED = 1,

	localparam int unsigned  CF = C/PE,	// Channel Fold
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(PE) + $clog2(N) + 2
);

	//-----------------------------------------------------------------------
	// Design Geometry

	// For each channel = [0,channel):
	//	 M_channel = M0 + CX*channel
	//	 B_channel = B0 + CX*channel
	// Input/threshold precision computed according with the maximum posible value
	localparam real  CX = 1.375;
	localparam int unsigned K = $clog2(N*(M0+C*CX) + (B0+C*CX)); // unused sign + magnitude
	localparam int unsigned C_BITS = C < 2? 1 : $clog2(C);

	localparam int unsigned MST_STRM_WROUNDS = 503;

	typedef int unsigned  threshs_t[C][N];
	function threshs_t init_thresholds();
		automatic threshs_t  res;
		for(int unsigned  c = 0; c < C; c++) begin
			automatic real  m = M0 + c*CX;
			automatic real  b = B0 + c*CX;
			foreach(res[c][i]) begin
				res[c][i] = int'($ceil(m*i + b));
			end
		end
		return  res;
	endfunction : init_thresholds
	localparam threshs_t  THRESHS = init_thresholds();

	//-----------------------------------------------------------------------
	// Clock and Reset Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		#10ns;
		@(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// DUT
	logic                  s_axilite_AWVALID;
	uwire                  s_axilite_AWREADY;
	logic [ADDR_BITS-1:0]  s_axilite_AWADDR;	// lowest 2 bits (byte selectors) are ignored
	logic                  s_axilite_WVALID;
	uwire                  s_axilite_WREADY;
	logic [         31:0]  s_axilite_WDATA;
	uwire                  s_axilite_BVALID;
	logic                  s_axilite_BREADY;
	uwire [          1:0]  s_axilite_BRESP;
	logic                  s_axilite_ARVALID;
	uwire                  s_axilite_ARREADY;
	logic [ADDR_BITS-1:0]  s_axilite_ARADDR;
	uwire                  s_axilite_RVALID;
	uwire                  s_axilite_RREADY = 1;
	uwire [         31:0]  s_axilite_RDATA;
	uwire [          1:0]  s_axilite_RRESP;

	uwire  irdy;
	logic  ivld;
	logic [PE-1:0][K-1:0]  idat;

	logic  ordy = 0;
	uwire  ovld;
	uwire [PE-1:0][$clog2(N+1)-1:0]  odat;

	thresholding_axi #(.N(N), .WI(K), .WT(K), .C(C), .PE(PE), .SIGNED(0), .USE_AXILITE(1)) dut (
		.ap_clk(clk), .ap_rst_n(!rst),

		// Configuration
		.s_axilite_AWVALID, .s_axilite_AWREADY, .s_axilite_AWADDR,
		.s_axilite_WVALID,  .s_axilite_WREADY,  .s_axilite_WDATA, .s_axilite_WSTRB('1),
		.s_axilite_BVALID,  .s_axilite_BREADY,  .s_axilite_BRESP,
		.s_axilite_ARVALID, .s_axilite_ARREADY, .s_axilite_ARADDR,
		.s_axilite_RVALID,  .s_axilite_RREADY,  .s_axilite_RDATA, .s_axilite_RRESP,

		// Stream Processing
		.s_axis_tready(irdy), .s_axis_tvalid(ivld), .s_axis_tdata(idat),
		.m_axis_tready(ordy), .m_axis_tvalid(ovld), .m_axis_tdata(odat)
	);

	//-----------------------------------------------------------------------
	// Input Stimuli
	typedef logic [PE-1:0][K-1:0]  input_t;
	typedef logic [$clog2(CF)+$clog2(PE)+$clog2(N)-1:0]  addr_t;
	input_t  QW[$];  // Input Feed Tracing
	addr_t   QC[$];

	int unsigned  error_cnt = 0;
	bit  done = 0;
	initial begin
		// Report testbench details
		$display("Testbench - thresholding K=%0d -> N=%0d", K, N);
		for(int unsigned  c = 0; c < C; c++) begin
			$write("Channel #%0d: Thresholds = {", c);
			for(int unsigned  i = 0; i < N; i++)  $write(" %0d", THRESHS[c][i]);
			$display(" }");
		end

		// Config
		s_axilite_AWVALID = 0;
		s_axilite_AWADDR  = 'x;
		s_axilite_WVALID  = 0;
		s_axilite_WDATA   = 'x;
		s_axilite_BREADY  = 0;
		s_axilite_ARVALID = 0;
		s_axilite_ARADDR  = 'x;

		// Stream Input
		ivld = 0;
		idat = 'x;

		@(posedge clk iff !rst);

		// Threshold Configuration
		for(int unsigned  c = 0; c < C; c+=PE) begin
			automatic addr_t  addr = 0;
			if(CF > 1)  addr[$clog2(N)+$clog2(PE)+:$clog2(CF)] = c/PE;
			for(int unsigned  pe = 0; pe < PE; pe++) begin
				if(PE > 1)  addr[$clog2(N)+:$clog2(PE)] = pe;
				for(int unsigned  t = 0; t < N; t++) begin
					addr[0+:$clog2(N)] = t;
					fork
						begin
							s_axilite_AWVALID <= 1;
							s_axilite_AWADDR  <= { addr, 2'b00 };
							@(posedge clk iff s_axilite_AWREADY);
							s_axilite_AWVALID <= 0;
							s_axilite_AWADDR  <= 'x;
						end
						begin
							s_axilite_WVALID <= 1;
							s_axilite_WDATA  <= THRESHS[c+pe][t];
							@(posedge clk iff s_axilite_WREADY);
							s_axilite_WVALID <= 0;
							s_axilite_WDATA  <= 'x;
						end
						begin
							s_axilite_BREADY <= 1;
							@(posedge clk iff s_axilite_BVALID);
							assert(s_axilite_BRESP == '0) else begin
								$error("Error on parameter write.");
								$stop;
							end
							s_axilite_BREADY <= 0;
						end
					join
				end
			end
		end

		fork
			// Intermittent configuration readback
			while(!done) begin
				if(($urandom()%37) != 0) begin
					s_axilite_ARVALID <= 0;
					s_axilite_ARADDR  <= 'x;
					@(posedge clk);
				end
				else begin
					automatic addr_t  addr = $urandom()%N;
					if(PE > 1)  addr[$clog2(N)+:$clog2(PE)] = $urandom()%PE;
					if(CF > 1)  addr[$clog2(N)+$clog2(PE)+:$clog2(CF)] = $urandom()%CF;

					s_axilite_ARVALID <= 1;
					s_axilite_ARADDR  <= { addr, 2'b00 };
					@(posedge clk iff s_axilite_ARREADY);

					QC.push_back(addr);
				end
			end

			// AXI4Stream MST Writes input values
			repeat(MST_STRM_WROUNDS) begin
				automatic input_t  dat;

				while(THROTTLED && ($urandom()%7 == 0)) @(posedge clk);

				std::randomize(dat);
				ivld <= 1;
				idat <= dat;
				@(posedge clk iff irdy);
				ivld <=  0;
				idat <= 'x;
				QW.push_back(dat);
			end
		join_any
		done <= 1;
		repeat(N+16)  @(posedge clk);

		assert(QW.size() == 0) else begin
			$error("Missing %0d outputs.", QW.size());
			$stop;
		end
		assert(QC.size() == 0) else begin
			$error("Missing %0d readback replies.", QC.size());
			$stop;
		end

		$display("Test completed: %0d errors in %0d tests.", error_cnt, MST_STRM_WROUNDS);
		$display("=========================================");
		$finish;
	end

	// Output Checker -------------------------------------------------------

	// Configuration Readback
	always_ff @(posedge clk iff s_axilite_RVALID) begin
		assert(s_axilite_RRESP == '0) else begin
			$error("Read back error.");
			$stop;
		end
		assert(QC.size()) begin
			automatic addr_t  addr = QC.pop_front();
			automatic int unsigned  cnl =
				(CF == 1? 0 : addr[$clog2(N)+$clog2(PE)+:$clog2(CF)] * PE) +
				(PE == 1? 0 : addr[$clog2(N)+:$clog2(PE)]);
			automatic logic [K-1:0]  exp = THRESHS[cnl][addr[0+:$clog2(N)]];
			assert(s_axilite_RDATA == exp) else begin
				$error("Readback mismatch on #%0d.%0d: %0d instead of %0d", cnl, addr[0+:$clog2(N)], s_axilite_RDATA, exp);
				$stop;
			end
		end
		else begin
			$error("Spurious readback output.");
			$stop;
		end
	end

	// Stream Output
	int unsigned  OCnl = 0;
	always @(posedge clk) begin
		if(rst) begin
			OCnl <= 0;
			ordy <= 1'b0;
		end
		else begin
			if(!ordy || ovld)  ordy <= ($urandom()%5 != 0) || !THROTTLED;

			if(ordy && ovld) begin
				assert(QW.size()) begin
					automatic input_t  x = QW.pop_front();

					for(int unsigned  pe = 0; pe < PE; pe++) begin
						automatic int unsigned  cnl = OCnl + pe;

						$display("Mapped CNL=%0d DAT=%3d -> #%2d", cnl, x[pe], odat[pe]);
						assert(
							((odat[pe] == 0) || (THRESHS[cnl][odat[pe]-1] <= x[pe])) &&
							((odat[pe] == N) || (x[pe] < THRESHS[cnl][odat[pe]]))
						) else begin
							$error("Output error on presumed input CNL=%0d DAT=0x%0x -> #%0d", cnl, x[pe], odat[pe]);
							error_cnt++;
							$stop;
						end
					end
				end
				else begin
					$error("Spurious output.");
					$stop;
				end

				OCnl <= (OCnl + PE)%C;
			end
		end
	end

endmodule: thresholding_axi_tb

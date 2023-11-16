/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 *
 */

module thresholding_tb #(
	int unsigned  K  = 10,	// input precision
	int unsigned  N  =  4,	// output precision
	int unsigned  C  =  6,	// number of channels
	int unsigned  PE =  2,

	localparam int unsigned  CF = C/PE	// Channel Fold
);
	localparam int unsigned  MST_STRM_WROUNDS = 507;
	localparam bit  THROTTLED = 1;

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
	// Parallel Instances differing in Data Type
	typedef logic [K -1:0]  val_t;
	typedef val_t  threshs_t[C][2**N-1];
	typedef val_t [PE-1:0]  input_t;
	typedef logic [$clog2(CF)+$clog2(PE)+N-1:0]  addr_t;
	logic [0:2]  term = '0;
	always_comb begin
		if(&term)  $finish;
	end
	for(genvar  i = 0; i < 3; i++) begin : genTypes
		localparam bit  SIGNED = i>0;
		localparam bit  FPARG  = i>1;

		//- DUT -------------------------
		logic  cfg_en;
		logic  cfg_we;
		logic [$clog2(C)+N-1:0]  cfg_a;
		logic [K-1:0]  cfg_d;
		uwire  cfg_rack;
		uwire [K-1:0]  cfg_q;

		uwire  irdy;
		logic  ivld;
		logic [PE-1:0][K-1:0]  idat;

		logic  ordy = 0;
		uwire  ovld;
		uwire [PE-1:0][N-1:0]  odat;

		thresholding #(.N(N), .K(K), .C(C), .PE(PE), .SIGNED(SIGNED), .FPARG(FPARG)) dut (
			.clk, .rst,

			// Configuration
			.cfg_en, .cfg_we, .cfg_a, .cfg_d,
			.cfg_rack, .cfg_q,

			// Stream Processing
			.irdy, .ivld, .idat,
			.ordy, .ovld, .odat
		);

		//- Stimulus Driver -------------
		threshs_t  THRESHS;
		function val_t sigord(input val_t  x);
			automatic val_t  res = x;
			if(SIGNED) begin
				if(FPARG && x[K-1])  res[K-2:0] = ~x[K-2:0];
				res[K-1] = !x[K-1];
			end
			return  res;
		endfunction : sigord

		input_t  QW[$];  // Input tracing
		addr_t   QC[$];  // Readback tracking
		int unsigned  error_cnt = 0;
		bit  done = 0;
		initial begin

			// Generate thresholds
			std::randomize(THRESHS);
			foreach(THRESHS[c]) begin
				val_t  row[2**N-1] = THRESHS[c];
				row.sort with (sigord(item));
				THRESHS[c] = row;
			end

			// Report test case details
			$display("[%0d] Thresholding %s%s%0d -> uint%0d", i, SIGNED? "s" : "u", FPARG? "fp" : "int", K, N);
			for(int unsigned  c = 0; c < C; c++) begin
				$write("[%0d] Channel #%0d: Thresholds = {", i, c);
				for(int unsigned  i = 0; i < 2**N-1; i++)  $write(" %0X", THRESHS[c][i]);
				$display(" }");
			end

			// Config
			cfg_en = 0;
			cfg_we = 'x;
			cfg_a  = 'x;
			cfg_d  = 'x;

			// Stream Input
			ivld = 0;
			idat = 'x;

			@(posedge clk iff !rst);

			// Threshold Configuratin
			cfg_en <= 1;
			cfg_we <= 1;
			for(int unsigned  c = 0; c < C; c+=PE) begin
				if(CF > 1)  cfg_a[N+$clog2(PE)+:$clog2(CF)] <= c/PE;
				for(int unsigned  pe = 0; pe < PE; pe++) begin
					if(PE > 1)  cfg_a[N+:$clog2(PE)] = pe;
					for(int unsigned  t = 0; t < 2**N-1; t++) begin
						cfg_a[0+:N] <= t;
						cfg_d <= THRESHS[c+pe][t];
						@(posedge clk);
					end
				end
			end
			cfg_d <= 'x;

			fork
				// Intermittent configuration readback
				while(!done) begin
					cfg_en <= 0;
					cfg_we <= 'x;
					cfg_a  <= 'x;
					@(posedge clk);
					if(($urandom()%37) == 0) begin
						automatic addr_t  addr = $urandom()%(N-1);
						if(PE > 1)  addr[N+:$clog2(PE)] = $urandom()%PE;
						if(CF > 1)  addr[N+$clog2(PE)+:$clog2(CF)] = $urandom()%CF;

						cfg_en <= 1;
						cfg_we <= 0;
						cfg_a  <= addr;
						@(posedge clk);
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
			repeat(N+6)  @(posedge clk);

			assert(QW.size() == 0) else begin
				$error("[%0d] Missing %0d outputs.", i, QW.size());
				$stop;
			end
			assert(QC.size() == 0) else begin
				$error("[%0d] Missing %0d readback replies.", i, QC.size());
				$stop;
			end

			$display("[%0d] Test completed: %0d errors in %0d tests.", i, error_cnt, MST_STRM_WROUNDS);
			$display("=============================================");
			term[i] <= 1;
		end

		//- Readback Checker --------------
		always_ff @(posedge clk iff cfg_rack) begin
			assert(QC.size()) begin
				automatic addr_t  addr = QC.pop_front();
				automatic int unsigned  cnl =
					(CF == 1? 0 : addr[N+$clog2(PE)+:$clog2(CF)] * PE) +
					(PE == 1? 0 : addr[N+:$clog2(PE)]);
				automatic logic [K-1:0]  exp = THRESHS[cnl][addr[0+:N]];
				assert(cfg_q == exp) else begin
					$error("[%0d] Readback mismatch on #%0d.%0d: %0d instead of %0d", i, cnl, addr[0+:N], cfg_q, exp);
					$stop;
				end
			end
			else begin
				$error("[%0d] Spurious readback output.", i);
				$stop;
			end
		end

		// Output Checker
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

							$display("[%0d] Mapped CNL=%0d DAT=%3x -> #%2d", i, cnl, x[pe], odat[pe]);
							assert(
								((odat[pe] == 0) || (sigord(THRESHS[cnl][odat[pe]-1]) <= sigord(x[pe]))) &&
								((odat[pe] == 2**N-1) || (sigord(x[pe]) < sigord(THRESHS[cnl][odat[pe]])))
							) else begin
								$error("[%0d] Output error on presumed input CNL=%0d DAT=0x%0x -> #%0d", i, cnl, x[pe], odat[pe]);
								error_cnt++;
								$stop;
							end
						end
					end
					else begin
						$error("[%0d] Spurious output.", i);
						$stop;
					end

					OCnl <= (OCnl + PE)%C;
				end
			end
		end

	end : genTypes

endmodule: thresholding_tb

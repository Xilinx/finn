module mvu_dsp58_tb;

	localparam int unsigned  N = 1000;

	localparam int unsigned  MW = 12;
	localparam int unsigned  MH = 4;
	localparam int unsigned  PE = 2;
	localparam int unsigned  SIMD = 6;
	localparam int unsigned  ACTIVATION_WIDTH = 8;
	localparam int unsigned  WEIGHT_WIDTH = 8;
	localparam int unsigned  ACCU_WIDTH = 24;

	//- Global Control ------------------
	logic  clk = 1;
	logic  clk2x = 1;
	always #5ns clk = !clk;
	always #2.5ns clk2x = !clk2x;

	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	//- DUTs ----------------------------

	// Weight Stream
	logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  s_axis_weights_tdata;
	logic  s_axis_weights_tvalid[2];
	uwire  s_axis_weights_tready[2];

	// Input Stream
	logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  s_axis_input_tdata;
	logic  s_axis_input_tvalid[2];
	uwire  s_axis_input_tready[2];

	// Output Stream
	uwire [PE-1:0][ACCU_WIDTH-1:0]  m_axis_output_tdata[2];
	uwire  m_axis_output_tvalid[2];
	logic  m_axis_output_tready[2];

	for(genvar  i = 0; i < 2; i++) begin : genDUTs
		mvu_vvu_axi #(
			.IS_MVU(1),
			.COMPUTE_CORE("mvu_vvu_8sx9_dsp58"),
			.MW(MW), .MH(MH),
			.PE(PE), .SIMD(SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
			.WEIGHT_WIDTH(WEIGHT_WIDTH),
			.ACCU_WIDTH(ACCU_WIDTH),
			.PUMPED_COMPUTE(i)
		) dut (
			.ap_clk(clk), .ap_clk2x(clk2x), .ap_rst_n(!rst),
			.s_axis_weights_tdata,                        .s_axis_weights_tvalid(s_axis_weights_tvalid[i]), .s_axis_weights_tready(s_axis_weights_tready[i]),
			.s_axis_input_tdata,                          .s_axis_input_tvalid  (s_axis_input_tvalid  [i]), .s_axis_input_tready  (s_axis_input_tready  [i]),
			.m_axis_output_tdata(m_axis_output_tdata[i]), .m_axis_output_tvalid (m_axis_output_tvalid [i]), .m_axis_output_tready (m_axis_output_tready [i])
		);
	end : genDUTs


	//- Stimuli -------------------------

	// Weight Feed
	initial begin
		s_axis_weights_tvalid = '{ default: 0 };
		s_axis_weights_tdata  = 'x;
		@(posedge clk iff !rst);

		repeat(N * (MH/PE)*(MW/SIMD)) begin
			automatic type(s_axis_weights_tdata)  weights;
			std::randomize(weights);
			s_axis_weights_tdata <= weights;
			s_axis_weights_tvalid <= '{ default: 1 };
			fork
				begin
					@(posedge clk iff s_axis_weights_tready[0]);
					s_axis_weights_tvalid[0] <= 0;
				end
				begin
					@(posedge clk iff s_axis_weights_tready[1]);
					s_axis_weights_tvalid[1] <= 0;
				end
			join
		end
	end

	// Input Feed
	initial begin
		s_axis_input_tvalid = '{ default: 0 };
		s_axis_input_tdata  = 'x;
		@(posedge clk iff !rst);

		repeat(N * (MW/SIMD)) begin
			automatic type(s_axis_input_tdata)  in;
			std::randomize(in);
			s_axis_input_tdata <= in;
			s_axis_input_tvalid <= '{ default: 1 };
			fork
				begin
					@(posedge clk iff s_axis_input_tready[0]);
					s_axis_input_tvalid[0] <= 0;
				end
				begin
					@(posedge clk iff s_axis_input_tready[1]);
					s_axis_input_tvalid[1] <= 0;
				end
			join
		end
	end

	// Output Capture and Comparison
	initial begin
		m_axis_output_tready = '{ default: 0 };
		@(posedge clk iff !rst);

		repeat(N * (MH/PE)) begin
			automatic type(m_axis_output_tdata)  res;
			m_axis_output_tready <= '{ default: 1 };
			fork
				begin
					@(posedge clk iff m_axis_output_tvalid[0]);
					m_axis_output_tready[0] <= 0;
					res[0] = m_axis_output_tdata[0];
				end
				begin
					@(posedge clk iff m_axis_output_tvalid[1]);
					m_axis_output_tready[1] <= 0;
					res[1] = m_axis_output_tdata[1];
				end
			join
			assert(res[0] == res[1]) else begin
				$error("Output mismatch: %0x <=> %0x", res[0], res[1]);
				$stop;
			end
			while($urandom()%7 < MW/SIMD) @(posedge clk);	// Occassional backpressure
		end

		$display("Test completed.");
		$finish;
	end

endmodule : mvu_dsp58_tb

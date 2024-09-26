module mvu_3sx3u_tb;

	localparam int unsigned  ROUNDS = 157;

	localparam int unsigned  MH = 32;
	localparam int unsigned  MW = 60;
	localparam int unsigned  PE = 1;
	localparam int unsigned  SIMD = 1;

	localparam int unsigned  ACTIVATION_WIDTH = 3;
	localparam int unsigned  WEIGHT_WIDTH = 3;
	localparam int unsigned  ACCU_WIDTH = 16;


	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 1;
	always #5ns clk = !clk;

	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// DUT
	logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  s_axis_weights_tdata;
	logic  s_axis_weights_tvalid;
	uwire  s_axis_weights_tready;

	logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  s_axis_input_tdata;
	logic  s_axis_input_tvalid;
	uwire  s_axis_input_tready;

	uwire [PE-1:0][ACCU_WIDTH-1:0]  m_axis_output_tdata;
	uwire  m_axis_output_tvalid;
	logic  m_axis_output_tready;

	mvu_vvu_axi #(
		.IS_MVU(1),
		.COMPUTE_CORE("mvu_4sx4u_dsp48e2"),
		.MH(MH), .MW(MW),
		.PE(PE), .SIMD(SIMD),

		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.WEIGHT_WIDTH(WEIGHT_WIDTH),
		.ACCU_WIDTH(ACCU_WIDTH)
		//int unsigned SEGMENTLEN = 0,
		//bit FORCE_BEHAVIORAL = 0,
	) dut (
		.ap_clk(clk), .ap_clk2x('x), .ap_rst_n(!rst),
		.s_axis_weights_tdata, .s_axis_weights_tvalid, .s_axis_weights_tready,
		.s_axis_input_tdata,   .s_axis_input_tvalid,   .s_axis_input_tready,
		.m_axis_output_tdata,  .m_axis_output_tvalid,  .m_axis_output_tready
	);

	//-----------------------------------------------------------------------
	// Stimuli

	//- Infinite Weight Feed ------------
	typedef logic signed [WEIGHT_WIDTH-1:0]  weights_t[MH][MW];
	function weights_t calc_WEIGHTS();
		automatic weights_t  ret;
		std::randomize(ret);
		return  ret;
	endfunction : calc_WEIGHTS
	weights_t  WEIGHTS = calc_WEIGHTS();

	initial begin
		s_axis_weights_tdata  = 'x;
		s_axis_weights_tvalid =  0;
		@(posedge clk iff !rst);

		forever begin
			for(int unsigned  h = 0; h < MH; h+=PE) begin
				for(int unsigned  w = 0; w < MW; w+=SIMD) begin
					for(int unsigned  pe = 0; pe < PE; pe++) begin
						for(int unsigned  simd = 0; simd < SIMD; simd++) begin
							s_axis_weights_tdata[pe][simd] <= WEIGHTS[h+pe][w+simd];
						end
					end
					s_axis_weights_tvalid <= 1;
					@(posedge clk iff s_axis_weights_tready);
					s_axis_weights_tvalid <=  0;
					s_axis_weights_tdata  <= 'x;
				end
			end
		end
	end

	//- Input Feed and Reference Computation
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  outvec_t;
	outvec_t  Q_ref[$] = {};

	initial begin
		s_axis_input_tdata  = 'x;
		s_axis_input_tvalid =  0;
		@(posedge clk iff !rst);

		repeat(ROUNDS) begin : blkRounds
			automatic logic [MH-1:0][ACCU_WIDTH-1:0]  accus = '{ default: 0 };

			for(int unsigned  w = 0; w < MW; w+=SIMD) begin : blkSF
				for(int unsigned  simd = 0; simd < SIMD; simd++) begin : blkSIMD
					automatic logic [ACTIVATION_WIDTH-1:0]  act = $urandom();
					for(int unsigned  h = 0; h < MH; h++) begin : blkMH
						automatic logic signed [ACCU_WIDTH-1:0]  prod = WEIGHTS[h][w+simd] * $signed({1'b0, act});
						accus[h] += prod;
					end : blkMH
					s_axis_input_tdata[simd] <= act;
				end : blkSIMD
				s_axis_input_tvalid <= 1;
				@(posedge clk iff s_axis_input_tready);
				s_axis_input_tvalid <=  0;
				s_axis_input_tdata  <= 'x;
			end : blkSF

			for(int unsigned  h = 0; h < MH; h+=PE) begin
				Q_ref.push_back(accus[h+:PE]);
			end

		end : blkRounds
	end

	//- Output Checker
	initial begin
		automatic int  timeout = 0;

		m_axis_output_tready = 0;
		@(posedge clk iff !rst);

		m_axis_output_tready <= 1;
		while(timeout < MW/SIMD+16) begin
			@(posedge clk);
			if(!m_axis_output_tvalid)  timeout++;
			else begin
				automatic outvec_t  exp;

				assert(Q_ref.size()) else begin
					$error("Spurious output.");
					$stop;
				end

				exp = Q_ref.pop_front();
				assert(m_axis_output_tdata === exp) else begin
					$error("Mismatched output %p instead of %p.", m_axis_output_tdata, exp);
					$stop;
				end

				timeout = 0;
			end
		end
		m_axis_output_tready <= 0;

		assert(Q_ref.size() == 0) else begin
			$error("Missing output.");
			$stop;
		end

		$display("Test completed.");
		$finish;
	end

endmodule : mvu_3sx3u_tb

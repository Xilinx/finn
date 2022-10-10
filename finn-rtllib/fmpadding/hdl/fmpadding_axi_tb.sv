
module fmpadding_axi_tb #(
	int unsigned  XCOUNTER_BITS = 8,
	int unsigned  YCOUNTER_BITS = 8,
	int unsigned  NUM_CHANNELS  = 4,
	int unsigned  SIMD          = 2,
	int unsigned  ELEM_BITS     = 4
)();
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8);

	//- Global Control ------------------
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst;

	// Parameter Configuration ----------
	logic         we;
	logic [ 2:0]  wa;
	logic [31:0]  wd;

	//- AXI Stream - Input --------------
	uwire  s_axis_tready;
	logic  s_axis_tvalid;
	logic [STREAM_BITS-1:0]  s_axis_tdata;

	//- AXI Stream - Output -------------
	logic  m_axis_tready;
	uwire  m_axis_tvalid;
	uwire [STREAM_BITS-1:0]  m_axis_tdata;


	// DUT
	fmpadding_axi #(
		.XCOUNTER_BITS(XCOUNTER_BITS),
		.YCOUNTER_BITS(YCOUNTER_BITS),
		.NUM_CHANNELS(NUM_CHANNELS),
		.SIMD(SIMD),
		.ELEM_BITS(ELEM_BITS)
	) dut (
		.ap_clk(clk), .ap_rst_n(!rst),
		.we, .wa, .wd,
		.s_axis_tready, .s_axis_tvalid, .s_axis_tdata,
		.m_axis_tready, .m_axis_tvalid, .m_axis_tdata
	);

	// Stimuli
	localparam int unsigned  IMAGES = 2;
	localparam int unsigned  XSIZE = 10;
	localparam int unsigned  YSIZE =  7;
	localparam int unsigned  PAD_LEFT   = 2;
	localparam int unsigned  PAD_RIGHT  = 3;
	localparam int unsigned  PAD_TOP    = 1;
	localparam int unsigned  PAD_BOTTOM = 2;
	initial begin
		we =  0;
		wa = 'x;
		wd = 'x;

		s_axis_tvalid =  0;
		s_axis_tdata  = 'x;

		// Configure Parameters
		rst = 1;
		@(posedge clk);
		we <= 1;
		/* XOn  */	wa <= 0; wd <= PAD_LEFT;           @(posedge clk);
		/* XOff */	wa <= 1; wd <= XSIZE - PAD_RIGHT;  @(posedge clk);
		/* XEnd */	wa <= 2; wd <= XSIZE - 1;          @(posedge clk);
		/* YOn  */	wa <= 4; wd <= PAD_TOP;            @(posedge clk);
		/* YOff */	wa <= 5; wd <= YSIZE - PAD_BOTTOM; @(posedge clk);
		/* YEnd */	wa <= 6; wd <= YSIZE - 1;          @(posedge clk);
		we <=  0;
		wa <= 'x;
		wd <= 'x;
		@(posedge clk);
		rst <= 0;

		// Feed data input
		s_axis_tvalid <= 1;
		for(int unsigned  i = 0; i < IMAGES * (XSIZE-PAD_LEFT-PAD_RIGHT) * (YSIZE-PAD_TOP-PAD_BOTTOM) * (NUM_CHANNELS/SIMD); i++) begin
			s_axis_tdata  <= i;
			@(posedge clk iff s_axis_tready);
			if($urandom()%5 == 0) begin
				s_axis_tvalid <=  0;
				s_axis_tdata  <= 'x;
				@(posedge clk);
				s_axis_tvalid <=  1;
			end
		end
		s_axis_tvalid <=  0;
		s_axis_tdata  <= 'x;
	end

	// Ouput Throttler
	initial begin
		m_axis_tready =  0;
		@(posedge clk iff !rst);
		m_axis_tready <= 1;
		forever @(posedge clk iff m_axis_tvalid) begin
			m_axis_tready <= 0;
			repeat(4-$clog2(1+$urandom()%15)) @(posedge clk);
			m_axis_tready <= 1;
		end
	end

	// Output logger
	initial begin
		repeat(IMAGES) begin
			for(int unsigned  y = 0; y < YSIZE; y++) begin
				for(int unsigned  x = 0; x < XSIZE; x++) begin
					automatic string  delim = " ";
					for(int unsigned  s = 0; s < NUM_CHANNELS/SIMD; s++) begin
						@(posedge clk iff m_axis_tvalid && m_axis_tready);
						$write("%s%02X", delim, m_axis_tdata);
						delim = ":";
					end
				end
				$display();
			end
			$display("----");
		end
		$finish;
	end

endmodule : fmpadding_axi_tb

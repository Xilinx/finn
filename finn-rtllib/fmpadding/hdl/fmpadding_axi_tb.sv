
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

	// AXI-Light for Parameter Configuration
	logic	       s_axilite_AWVALID;
	uwire	       s_axilite_AWREADY;
	logic	[2:0]  s_axilite_AWADDR;

	logic	        s_axilite_WVALID;
	uwire	        s_axilite_WREADY;
	logic	[31:0]  s_axilite_WDATA;

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
		.INIT_XON(0), .INIT_XOFF(0), .INIT_XEND(0),
		.INIT_YON(0), .INIT_YOFF(0), .INIT_YEND(0),
		.ELEM_BITS(ELEM_BITS)
	) dut (
		.ap_clk(clk), .ap_rst_n(!rst),

		.s_axilite_AWVALID, .s_axilite_AWREADY, .s_axilite_AWADDR,
		.s_axilite_WVALID, .s_axilite_WREADY, .s_axilite_WDATA, .s_axilite_WSTRB('1),
		.s_axilite_BVALID(), .s_axilite_BREADY('1),	.s_axilite_BRESP(),
		.s_axilite_ARVALID('0), .s_axilite_ARREADY(), .s_axilite_ARADDR('x),
		.s_axilite_RVALID(), .s_axilite_RREADY('0), .s_axilite_RDATA(), .s_axilite_RRESP(),

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

	task axi_write(input logic [2:0]  wa, input logic [31:0]  wd);
		s_axilite_AWVALID <= 1;
		s_axilite_AWADDR <= wa;
		@(posedge clk iff s_axilite_AWREADY);
		s_axilite_AWVALID <= 0;
		s_axilite_AWADDR <= 'x;

		s_axilite_WVALID <= 1;
		s_axilite_WDATA <= wd;
		@(posedge clk iff s_axilite_WREADY);
		s_axilite_WVALID <= 0;
		s_axilite_WDATA <= 'x;
	endtask : axi_write


	initial begin
		s_axilite_AWVALID = 0;
		s_axilite_AWADDR = 'x;
		s_axilite_WVALID = 0;
		s_axilite_WDATA = 'x;

		s_axis_tvalid =  0;
		s_axis_tdata  = 'x;

		// Configure Parameters
		rst = 0;
		@(posedge clk);
		/* XOn  */	axi_write(0, PAD_LEFT);
		/* XOff */	axi_write(1, XSIZE - PAD_RIGHT);
		/* XEnd */	axi_write(2, XSIZE - 1);
		/* YOn  */	axi_write(4, PAD_TOP);
		/* YOff */	axi_write(5, YSIZE - PAD_BOTTOM);
		/* YEnd */	axi_write(6, YSIZE - 1);
		@(posedge clk);
		rst <= 1;
		@(posedge clk);
		rst <= 0;
		@(posedge clk);

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

	// Output Throttler
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
		@(negedge rst);
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

/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	DMA engine for fetching weight matrices from external memory.
 *
 * @description
 *  Reads weight matrices from external memory (DDR/HBM) via AXI-MM and
 *  delivers them as a narrow stream to the MVU/VVU compute path.
 *
 *  Expected memory layout (produced by make_weight_file in the FINN compiler):
 *
 *    base_address + ADDRESS_OFFSET
 *    ├── Layer 0   @ offset 0
 *    ├── Layer 1   @ offset LAYER_OFFS
 *    ├── Layer 2   @ offset 2 * LAYER_OFFS
 *    │   ...
 *    └── Layer N-1 @ offset (N-1) * LAYER_OFFS
 *
 *  Within each layer, MH*MW weight elements are stored as MH*MW/IWSIMD
 *  consecutive IWSIMD-groups, each byte-aligned to ceil(IWSIMD*WEIGHT_WIDTH, 8)
 *  bits (= DS_BITS_BA).  The tight per-layer size is:
 *
 *    dma_len = (MH*MW / IWSIMD) * ceil(IWSIMD * WEIGHT_WIDTH, 8) / 8  bytes
 *
 *  LAYER_OFFS rounds dma_len up to the AXI bus width (DATA_BITS/8) so that
 *  each layer's start address is bus-aligned.  The gap between dma_len and
 *  LAYER_OFFS is padding (zeros); it is never fetched.
 *
 *  Because dma_len is generally not a multiple of the bus width, the last AXI
 *  beat of each DMA transfer carries a partial payload.  The external VPC
 *  width converter (instantiated in the wrapper with N=MH*MW) crops the
 *  partial beat, preventing stale data from bleeding across consecutive
 *  transfers.
 *
 *  IWSIMD depends on the operating mode:
 *    TH > 1 (tiled): IWSIMD = (PE * SIMD) / TH — one tile's worth of weights
 *    TH = 1 (direct): IWSIMD = SIMD — one dot-product group
 *
 *  Alignment assumptions:
 *    - LAYER_OFFS is a multiple of DATA_BITS/8 (AXI bus width in bytes),
 *      so every layer starts at a bus-aligned address.
 *    - Partial last AXI beats occur when dma_len does not divide evenly by
 *      the bus width; the external VPC (N=MH*MW) crops them.
 *    - MH*MW must be divisible by IWSIMD (guaranteed by the FINN compiler's
 *      folding constraints: MH is divisible by PE, MW by SIMD).
 *    - WEIGHT_WIDTH * IWSIMD need not be a multiple of 8; per-group byte
 *      padding is applied in both the memory image and dma_len calculation.
 *
 *****************************************************************************/

module fetch_weights #(
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  TH = 1,
	int unsigned  MH,
	int unsigned  MW,
	int unsigned  N_REPS,
	int unsigned  WEIGHT_WIDTH = 8,

	int unsigned  ADDR_BITS = 64,
	int unsigned  DATA_BITS = 256,
	int unsigned  LEN_BITS = 32,
	int unsigned  IDX_BITS = 16,

	int unsigned  N_LAYERS,

	int unsigned  QDEPTH = 8,
	int unsigned  EN_OREG = 1,
	int unsigned  N_DCPL_STGS = 1,
	int unsigned  DBG = 0,

	bit [ADDR_BITS-1:0]  ADDRESS_OFFSET = 0,

	// Safely deducible parameters
	// In external memory (DDR, HBM, ...) weights are stored per IWSIMD group, each
	// padded to roundup(IWSIMD*WEIGHT_WIDTH, 8) bits (= DS_BITS_BA, the DWC output
	// width). The per-layer stride must reflect that per-group padding (not tight
	// bit-packing); reduces to the tight value when IWSIMD*WEIGHT_WIDTH is byte-aligned.
	localparam int unsigned  IWSIMD = (TH > 1)? ((PE*SIMD)/TH) : SIMD,
	localparam int unsigned  OWSIMD = (PE * SIMD) / TH,
	localparam int unsigned  DS_BITS_BA = (IWSIMD*WEIGHT_WIDTH+7)/8 * 8,
	localparam int unsigned  WS_BITS_BA = (OWSIMD*WEIGHT_WIDTH+7)/8 * 8
)(
	input  logic  aclk,
	input  logic  aresetn,

	output logic  m_done,

	// AXI
	output logic[ADDR_BITS-1:0]      m_axi_ddr_araddr,
	output logic[1:0]                m_axi_ddr_arburst,
	output logic[3:0]                m_axi_ddr_arcache,
	output logic[1:0]                m_axi_ddr_arid,
	output logic[7:0]                m_axi_ddr_arlen,
	output logic[0:0]                m_axi_ddr_arlock,
	output logic[2:0]                m_axi_ddr_arprot,
	output logic[2:0]                m_axi_ddr_arsize,
	input  logic                     m_axi_ddr_arready,
	output logic                     m_axi_ddr_arvalid,
	output logic[ADDR_BITS-1:0]      m_axi_ddr_awaddr,
	output logic[1:0]                m_axi_ddr_awburst,
	output logic[3:0]                m_axi_ddr_awcache,
	output logic[1:0]                m_axi_ddr_awid,
	output logic[7:0]                m_axi_ddr_awlen,
	output logic[0:0]                m_axi_ddr_awlock,
	output logic[2:0]                m_axi_ddr_awprot,
	output logic[2:0]                m_axi_ddr_awsize,
	input  logic                     m_axi_ddr_awready,
	output logic                     m_axi_ddr_awvalid,
	input  logic[DATA_BITS-1:0]      m_axi_ddr_rdata,
	input  logic[1:0]                m_axi_ddr_rid,
	input  logic                     m_axi_ddr_rlast,
	input  logic[1:0]                m_axi_ddr_rresp,
	output logic                     m_axi_ddr_rready,
	input  logic                     m_axi_ddr_rvalid,
	output logic[DATA_BITS-1:0]      m_axi_ddr_wdata,
	output logic                     m_axi_ddr_wlast,
	output logic[DATA_BITS/8-1:0]    m_axi_ddr_wstrb,
	input  logic                     m_axi_ddr_wready,
	output logic                     m_axi_ddr_wvalid,
	input  logic[1:0]                m_axi_ddr_bid,
	input  logic[1:0]                m_axi_ddr_bresp,
	output logic                     m_axi_ddr_bready,
	input  logic                     m_axi_ddr_bvalid,

	// Index
	input  logic                     s_idx_tvalid,
	output logic                     s_idx_tready,
	input  logic[IDX_BITS-1:0]       s_idx_tdata,

	// DMA stream out (to external width converter)
	output logic                     axis_dma_tvalid,
	input  logic                     axis_dma_tready,
	output logic[DATA_BITS-1:0]      axis_dma_tdata,

	// DWC stream in (from external width converter)
	input  logic                     axis_dwc_tvalid,
	output logic                     axis_dwc_tready,
	input  logic[DS_BITS_BA-1:0]     axis_dwc_tdata,

	// Stream
	output logic                     m_axis_tvalid,
	input  logic                     m_axis_tready,
	output logic[WS_BITS_BA-1:0]     m_axis_tdata,

	// Base Address
	input logic [ADDR_BITS-1:0]      base_address
);

	//=== Parameter Assertions ==============================================
	initial begin
		if(PE == 0 || SIMD == 0 || TH == 0) begin
			$error("%m: PE (%0d), SIMD (%0d), and TH (%0d) must be non-zero.", PE, SIMD, TH);
			$finish;
		end
		if(MH == 0 || MW == 0) begin
			$error("%m: MH (%0d) and MW (%0d) must be non-zero.", MH, MW);
			$finish;
		end
		if(WEIGHT_WIDTH == 0) begin
			$error("%m: WEIGHT_WIDTH must be non-zero.");
			$finish;
		end
		if((PE * SIMD) % TH != 0) begin
			$error("%m: PE*SIMD (%0d) must be divisible by TH (%0d).", PE * SIMD, TH);
			$finish;
		end
		if((MH * MW) % IWSIMD != 0) begin
			$error("%m: MH*MW (%0d) must be divisible by IWSIMD (%0d).", MH * MW, IWSIMD);
			$finish;
		end
		if(DATA_BITS % WEIGHT_WIDTH != 0) begin
			$error("%m: DATA_BITS (%0d) must be divisible by WEIGHT_WIDTH (%0d).", DATA_BITS, WEIGHT_WIDTH);
			$finish;
		end
		if(DS_BITS_BA % WEIGHT_WIDTH != 0) begin
			$error("%m: DS_BITS_BA (%0d) must be divisible by WEIGHT_WIDTH (%0d).", DS_BITS_BA, WEIGHT_WIDTH);
			$finish;
		end
		if(N_REPS == 0) begin
			$error("%m: N_REPS must be non-zero.");
			$finish;
		end
		if(DATA_BITS == 0 || (DATA_BITS & (DATA_BITS - 1)) != 0) begin
			$error("%m: DATA_BITS (%0d) must be a power of two.", DATA_BITS);
			$finish;
		end
	end

	//=== Layer Offsets =====================================================
	localparam int unsigned  LAYER_OFFS = ((MH*MW/IWSIMD)*((IWSIMD*WEIGHT_WIDTH+7)/8) + (DATA_BITS/8-1)) & ~(DATA_BITS/8-1); // AXI bus-width aligned
	logic [N_LAYERS-1:0][ADDR_BITS-1:0]  l_offsets;
	for(genvar i = 0; i < N_LAYERS; i++) begin : genOffs
		assign	l_offsets[i] = i * LAYER_OFFS;
	end : genOffs

	//=== Index Handling & DMA Control ======================================
	logic  dma_tvalid;
	logic  dma_tready;
	logic [ADDR_BITS-1:0]  dma_addr;
	logic [ LEN_BITS-1:0]  dma_len;

	if(TH > 1) begin : genTiled

		localparam int unsigned  REPS_BITS = (N_REPS == 1)? 1 : $clog2(N_REPS);

		typedef enum logic [0:0] {ST_IDLE, ST_DMA}  state_e;

		//--- Registers -----------------------------------------------------
		state_e  State = ST_IDLE;
		state_e  state_n;

		logic [REPS_BITS-1:0]  CntDma = '0;
		logic [REPS_BITS-1:0]  cnt_dma_n;

		logic [IDX_BITS-1:0]  Idx = '0;
		logic [IDX_BITS-1:0]  idx_n;

		//--- Index Queue ---------------------------------------------------
		uwire  q_idx_vld;
		logic  q_idx_rdy;
		uwire [IDX_BITS-1:0]  q_idx_dat;

		fifo #(.DEPTH(QDEPTH), .DATA_WIDTH(IDX_BITS)) inst_queue_in (
			.clk(aclk), .rst(!aresetn),
			.count(), .maxcount(),
			.idat(s_idx_tdata), .ivld(s_idx_tvalid), .irdy(s_idx_tready),
			.odat(q_idx_dat), .ovld(q_idx_vld), .ordy(q_idx_rdy)
		);

		assign	dma_addr = base_address + ADDRESS_OFFSET + l_offsets[Idx];
		// External memory (DDR, HBM, ...) stores weights as byte-aligned per-IWSIMD
		// packets: each group of IWSIMD weights occupies roundup(IWSIMD*WEIGHT_WIDTH, 8)
		// bits (= DS_BITS_BA). The total fetch length must reflect that per-group
		// padding (not tight bit-packing), otherwise sub-byte weights under-fetch.
		// Reduces to the tight value whenever IWSIMD*WEIGHT_WIDTH is already byte-aligned.
		assign dma_len = (MH*MW/IWSIMD) * ((IWSIMD*WEIGHT_WIDTH+7)/8);

		//--- Sequential ----------------------------------------------------
		always_ff @(posedge aclk) begin
			if(~aresetn) begin
				State  <= ST_IDLE;
				CntDma <= '0;
				Idx    <= 'x;
			end
			else begin
				State  <= state_n;
				CntDma <= cnt_dma_n;
				Idx    <= idx_n;
			end
		end

		//--- Next State ----------------------------------------------------
		always_comb begin
			state_n = State;

			case(State)
			ST_IDLE:
				state_n = q_idx_vld? ST_DMA : ST_IDLE;

			ST_DMA:
				state_n = ((CntDma == N_REPS-1) && dma_tready)? ST_IDLE : ST_DMA;
			endcase
		end

		//--- Datapath ------------------------------------------------------
		always_comb begin
			cnt_dma_n = CntDma;
			idx_n     = Idx;

			q_idx_rdy  = 0;
			dma_tvalid = 0;

			case(State)
			ST_IDLE: begin
				q_idx_rdy  = 1;
				cnt_dma_n  = 0;
				if(q_idx_vld)
					idx_n  = q_idx_dat;
			end

			ST_DMA: begin
				dma_tvalid = 1;
				if(dma_tready)
					cnt_dma_n = CntDma + 1;
			end
			endcase
		end

	end : genTiled
	else begin : genDirect

		uwire [IDX_BITS-1:0]  q_idx_dat;

		fifo #(.DEPTH(QDEPTH), .DATA_WIDTH(IDX_BITS)) inst_idx_queue (
			.clk(aclk), .rst(!aresetn),
			.count(), .maxcount(),
			.idat(s_idx_tdata), .ivld(s_idx_tvalid), .irdy(s_idx_tready),
			.odat(q_idx_dat), .ovld(dma_tvalid), .ordy(dma_tready)
		);

		assign	dma_addr = base_address + ADDRESS_OFFSET + l_offsets[q_idx_dat];
		// Same byte-aligned per-IWSIMD-group packing as the tiled path (see above):
		// each of the MH*MW/IWSIMD groups occupies roundup(IWSIMD*WEIGHT_WIDTH, 8)
		// bits (= DS_BITS_BA) in external memory. Using tight bit-packing here would
		// under-fetch whenever IWSIMD*WEIGHT_WIDTH is not byte-aligned (e.g. SIMD=1,
		// sub-byte weights). Reduces to the tight value when it is byte-aligned.
		assign dma_len = (MH*MW/IWSIMD) * ((IWSIMD*WEIGHT_WIDTH+7)/8);

	end : genDirect

	//=== Write Channel Tie-off (read-only DMA) =============================
	assign	m_axi_ddr_awaddr  = '0;
	assign	m_axi_ddr_awburst = '0;
	assign	m_axi_ddr_awcache = '0;
	assign	m_axi_ddr_awid    = '0;
	assign	m_axi_ddr_awlen   = '0;
	assign	m_axi_ddr_awlock  = '0;
	assign	m_axi_ddr_awprot  = '0;
	assign	m_axi_ddr_awsize  = '0;
	assign	m_axi_ddr_awvalid = 0;
	assign	m_axi_ddr_wdata   = '0;
	assign	m_axi_ddr_wlast   = 0;
	assign	m_axi_ddr_wstrb   = '0;
	assign	m_axi_ddr_wvalid  = 0;
	assign	m_axi_ddr_bready  = 0;

	//=== DMA Engine ========================================================
	cdma_u_rd #(
		.DATA_BITS(DATA_BITS),
		.ADDR_BITS(ADDR_BITS),
		.LEN_BITS(LEN_BITS)
	) inst_dma (
		.aclk(aclk), .aresetn(aresetn),

		.rd_valid(dma_tvalid), .rd_ready(dma_tready),
		.rd_paddr(dma_addr), .rd_len(dma_len),
		.rd_done(m_done),

		.m_axi_ddr_arvalid(m_axi_ddr_arvalid),
		.m_axi_ddr_arready(m_axi_ddr_arready),
		.m_axi_ddr_araddr(m_axi_ddr_araddr),
		.m_axi_ddr_arid(m_axi_ddr_arid),
		.m_axi_ddr_arlen(m_axi_ddr_arlen),
		.m_axi_ddr_arsize(m_axi_ddr_arsize),
		.m_axi_ddr_arburst(m_axi_ddr_arburst),
		.m_axi_ddr_arlock(m_axi_ddr_arlock),
		.m_axi_ddr_arcache(m_axi_ddr_arcache),
		.m_axi_ddr_arprot(m_axi_ddr_arprot),
		.m_axi_ddr_rvalid(m_axi_ddr_rvalid),
		.m_axi_ddr_rready(m_axi_ddr_rready),
		.m_axi_ddr_rdata(m_axi_ddr_rdata),
		.m_axi_ddr_rlast(m_axi_ddr_rlast),
		.m_axi_ddr_rid(m_axi_ddr_rid),
		.m_axi_ddr_rresp(m_axi_ddr_rresp),

		.m_axis_ddr_tvalid(axis_dma_tvalid),
		.m_axis_ddr_tready(axis_dma_tready),
		.m_axis_ddr_tdata(axis_dma_tdata),
		.m_axis_ddr_tkeep(),
		.m_axis_ddr_tlast()
	);

	//=== Local Weight Buffer ===============================================
	logic  axis_lwb_tvalid;
	logic  axis_lwb_tready;
	logic [WS_BITS_BA-1:0]  axis_lwb_tdata;

	if(TH == 1) begin : genLwb
		local_weight_buffer #(
			.PE(PE), .SIMD(SIMD), .MH(MH), .MW(MW),
			.N_REPS(N_REPS), .WEIGHT_WIDTH(WEIGHT_WIDTH), .DBG(DBG)
		) inst_weight_buff (
			.clk(aclk), .rst(~aresetn),
			.ivld(axis_dwc_tvalid), .irdy(axis_dwc_tready), .idat(axis_dwc_tdata),
			.ovld(axis_lwb_tvalid), .ordy(axis_lwb_tready), .odat(axis_lwb_tdata)
		);
	end : genLwb
	else begin : genLwbPassthru
		assign	axis_lwb_tvalid = axis_dwc_tvalid;
		assign	axis_dwc_tready = axis_lwb_tready;
		assign	axis_lwb_tdata  = axis_dwc_tdata;
	end : genLwbPassthru

	//=== Output Register Slice =============================================
	if(EN_OREG) begin : genOreg
		skid #(
			.DATA_WIDTH(WS_BITS_BA), .FEED_STAGES(N_DCPL_STGS)
		) inst_oreg (
			.clk(aclk), .rst(!aresetn),
			.ivld(axis_lwb_tvalid), .irdy(axis_lwb_tready), .idat(axis_lwb_tdata),
			.ovld(m_axis_tvalid), .ordy(m_axis_tready), .odat(m_axis_tdata)
		);
	end : genOreg
	else begin : genOregPassthru
		assign	m_axis_tvalid  = axis_lwb_tvalid;
		assign	axis_lwb_tready = m_axis_tready;
		assign	m_axis_tdata   = axis_lwb_tdata;
	end : genOregPassthru

endmodule : fetch_weights

/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Consolidated FIFO with auto-selecting storage implementation.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *  Unified FIFO selecting between an SRL shift-register implementation
 *  for shallow depths, a distributed-RAM (LUTRAM) pointer-based
 *  implementation for moderate depths at wide data widths, and a
 *  memory-backed implementation for deep FIFOs. The RAM_STYLE parameter
 *  ("auto", "shift", "distributed", "block", "ultra") can force a
 *  specific backing; "auto" selects by DEPTH and DATA_WIDTH.
 *
 *  The SRL path uses a dynamic shift register with an output register,
 *  discounting one slot from the required SRL depth (minimum 4).
 *
 *  The distributed-RAM path uses LUTRAM with explicit read/write
 *  pointers and asynchronous read. It eliminates the cascade output mux
 *  that SRL needs at depths beyond 32 (2x SRLC32E + 1 LUT3/bit),
 *  using fewer LUTs per bit at the cost of pointer overhead (~27 LUTs).
 *  Auto-selection activates it for DEPTH 34-257 with DATA_WIDTH >= 12
 *  (34-64 unconditionally); the upper bound 257 = 256-entry LUTRAM +
 *  1 output register matches the natural 4x RAM64M8 capacity per byte.
 *  Explicit RAM_STYLE="distributed" forces it at any depth >= 2.
 *
 *  The memory path serves both BRAM and URAM, parameterized by an
 *  IS_URAM flag. Non-power-of-two depths are manually decomposed into
 *  a primary (lo) and a secondary (hi) memory space to control primitive
 *  usage. The read pipeline is stallable for BRAM and free-running for
 *  URAM (enabling output register absorption). URAM output is buffered
 *  through a recursive SRL FIFO instance.
 *****************************************************************************/

module fifo #(
	int unsigned  DEPTH,
	int unsigned  DATA_WIDTH,
	parameter  RAM_STYLE = "auto"	// "auto", "shift", "distributed", "block", "ultra"
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy,

	// Occupancy Monitoring
	output	logic [$clog2(DEPTH+1):0]  count,	// items currently held
	output	logic [$clog2(DEPTH+1):0]  maxcount	// maximum of count since reset
);
`default_nettype none

	typedef logic [DATA_WIDTH-1:0]  dat_t;

	initial begin
		if(DEPTH < 2) begin
			$error("%m: DEPTH of %0d must be 2 or above.", DEPTH);
			$finish;
		end
	end

	//-----------------------------------------------------------------------
	// Storage Implementation Selection
	//
	// SRL (shift):        1 LUT/bit up to 32-deep (SRLC32E), but cascading
	//                     beyond 32 adds 1 LUT3/bit for the cascade output
	//                     mux — effectively ~3 LUTs/bit at DEPTH 33-64.
	// LUTRAM (distributed): pointer-based with async read + output register.
	//                     1 LUT/bit via RAM64M8, no cascade mux. Pointer
	//                     overhead (~27 LUTs) amortized at DATA_WIDTH >= 12.
	//                     DEPTH 257 = 256-entry LUTRAM + 1 output register,
	//                     the natural capacity of 4x RAM64M8 per byte.
	// Crossover: DEPTH 34-64, DATA_WIDTH >= 12 — LUTRAM saves ~DATA_WIDTH
	//            LUTs by eliminating the SRL cascade mux. Beyond 64, both
	//            SRL and LUTRAM cascade, but LUTRAM still uses fewer LUTs
	//            per bit. BRAM (0.5 tile + ~30 logic LUTs) is cheaper in
	//            LUTs but consumes scarce block RAM; auto-select favors
	//            LUTRAM up to the natural RAM64M8 x4 boundary.
	localparam  RAM_STYLE_EFF =
		DEPTH <= 33?                      "shift" :
		RAM_STYLE != "auto"?              RAM_STYLE :
		DEPTH <=   64 && DATA_WIDTH < 12? "shift" :
		DEPTH <=  257?                    "distributed" :
		DEPTH <= 2028?                    "block" :
		/* else */                        "ultra";
	initial begin
		if(DEPTH <= 33) begin
			case(RAM_STYLE)
			"auto", "shift", "distributed": begin end
			default:
				$warning(
					"%m: Implementing shift FIFO (instead of %s) for shallow DEPTH of %0d.",
					RAM_STYLE, DEPTH
				);
			endcase
		end
	end

	//-----------------------------------------------------------------------
	// Memory-Backed Geometry
	//	Hoisted out of the memory implementation below so that the actually
	//	implemented capacity is known for sizing the occupancy monitor.
	localparam bit           IS_URAM   = (RAM_STYLE_EFF == "ultra");
	localparam int unsigned  MIN_ABITS = IS_URAM? 12 : 9;  // smallest primitive: URAM 4096, BRAM 512
	localparam int unsigned  QDEPTH    = IS_URAM? 16 : 0;
	localparam int unsigned  DEPTH_REQ = DEPTH - QDEPTH - 1; // actual required memory depth discounting capacity in output queue

	// Memory Space Decomposition
	typedef struct {
		int unsigned  lo;
		int unsigned  hi;
	} abits_t;
	function abits_t INIT_ABITS();
		automatic abits_t  abits = '{ lo: $clog2(DEPTH_REQ), hi: 0 };
		if(abits.lo > MIN_ABITS) begin
			// Consider Decomposition into lo + hi
			automatic int unsigned  hi_abits = $clog2(DEPTH_REQ - 2**(abits.lo-1));
			if(hi_abits < abits.lo - 1) begin
				abits.lo--;
				abits.hi = hi_abits < 1? 1 : hi_abits;
			end
		end
		return  abits;
	endfunction : INIT_ABITS
	localparam abits_t       ABITS      = INIT_ABITS();
	localparam int unsigned  MEM_IMPL   = 2**ABITS.lo + (ABITS.hi? 2**ABITS.hi : 0);
	localparam int unsigned  PIPE_DEPTH = IS_URAM? 3 + (2**ABITS.lo - 1) / 8192 : 2;

	//-----------------------------------------------------------------------
	// Occupancy Monitoring
	//	The implemented capacity may exceed the requested DEPTH: the shift
	//	path never shrinks below four SRL stages, and the memory path rounds
	//	its address space up to whole primitives. The counter is sized for
	//	the actual capacity so that it can never wrap.
	localparam int unsigned  DEPTH_ACTUAL =
		RAM_STYLE_EFF == "shift"?       (DEPTH > 4? DEPTH : 5) :
		RAM_STYLE_EFF == "distributed"? DEPTH :
		/* "block", "ultra" */          MEM_IMPL + PIPE_DEPTH + QDEPTH + 1;
	localparam int unsigned  COUNT_WIDTH = $clog2(DEPTH+1) + 1;
	initial begin
		if(DEPTH_ACTUAL > 2**COUNT_WIDTH - 1) begin
			$error(
				"%m: Occupancy counter of %0d bits too narrow for capacity of %0d.",
				COUNT_WIDTH, DEPTH_ACTUAL
			);
			$finish;
		end
	end

	//	Implementation-independent: counts the items accepted on the input
	//	interface that have not yet been delivered on the output interface.
	uwire  cnt_push = irdy && ivld;
	uwire  cnt_pop  = ovld && ordy;
	logic [COUNT_WIDTH-1:0]  Count    = 0;
	logic [COUNT_WIDTH-1:0]  MaxCount = 0;
	uwire [COUNT_WIDTH-1:0]  count_nxt = Count + ((cnt_push == cnt_pop)? 0 : cnt_push? 1 : -1);
	always_ff @(posedge clk) begin
		if(rst) begin
			Count    <= 0;
			MaxCount <= 0;
		end
		else begin
			Count <= count_nxt;
			if(count_nxt > MaxCount)  MaxCount <= count_nxt;
		end
	end
	assign	count    = Count;
	assign	maxcount = MaxCount;

	case(RAM_STYLE_EFF)
	// SRL Shift-Register Backing
	"shift": begin : genSRL
		localparam int unsigned  DEPTH_IMPL = DEPTH > 4? DEPTH-1 : 4;  // output register B provides one slot

		logic signed [$clog2(DEPTH_IMPL):0]  Ptr = '1;	// -1, 0, 1, ..., DEPTH_IMPL-1
		logic  Rdy = 1;
		(* SHREG_EXTRACT = "yes" *) dat_t  A[DEPTH_IMPL];
		assign	irdy = Rdy;

		logic  Vld = 0;
		dat_t  B = 'x;
		assign	odat = B;
		assign	ovld = Vld;

		uwire  bload = !Vld || ordy;
		uwire  push = Rdy && ivld;
		uwire  pop = !Ptr[$left(Ptr)] && bload;

		always_ff @(posedge clk) begin
			if(push)  A <= { idat, A[0:DEPTH_IMPL-2] };
		end

		always_ff @(posedge clk) begin
			if(rst) begin
				Ptr <= '1;
				Rdy <= 1;
				Vld <= 0;
				B <= 'x;
			end
			else begin
				assert(Rdy == (Ptr < signed'(DEPTH_IMPL-1))) else begin
					$error("%m: Broken Rdy computation.");
					$stop;
				end
				Ptr <= Ptr + ((push == pop)? 0 : push? 1 : -1);
				Rdy <= (pop == push)? Rdy : pop? 1 : Ptr[$left(Ptr)] || (((DEPTH_IMPL-2) & ~Ptr[$left(Ptr)-1:0]) != 0);
				if(bload) begin
					Vld <= !Ptr[$left(Ptr)];
					B <= A[Ptr[$left(Ptr)-1:0]];
				end
			end
		end

	end : genSRL

	// Distributed-RAM (LUTRAM) Pointer-Based FIFO
	"distributed": begin : genDist
		localparam int unsigned  MEM_DEPTH = DEPTH - 1;  // output register B provides one slot
		localparam int unsigned  ABITS     = $clog2(MEM_DEPTH);
		localparam int unsigned  MEM_SIZE  = 2**ABITS;

		//=== Pointer and Counter Declarations ==============================
		typedef logic [ABITS-1:0]  ptr_t;
		ptr_t  WrPtr = 0;
		ptr_t  RdPtr = 0;

		typedef logic signed [$clog2(MEM_DEPTH):0]  cap_t;
		cap_t  Credit = -MEM_DEPTH;  // -MEM_DEPTH (empty), ..., 0 (full)

		cap_t  Fill = 0;              // 0 (empty), ..., -MEM_DEPTH (full)

		assign	irdy = Credit[$left(Credit)];
		uwire  we = irdy && ivld;

		//=== Output Register ===============================================
		logic  Vld = 0;
		dat_t  B = 'x;
		assign	odat = B;
		assign	ovld = Vld;

		uwire  bload = !Vld || ordy;
		uwire  drain = Fill[$left(Fill)] && bload;

		//=== Pointer and Counter Management ================================
		always_ff @(posedge clk) begin
			if(rst) begin
				WrPtr  <= 0;
				RdPtr  <= 0;
				Credit <= -MEM_DEPTH;
				Fill   <= 0;
			end
			else begin
				if(we)     WrPtr <= WrPtr + 1;
				if(drain)  RdPtr <= RdPtr + 1;
				if(we != drain) begin
					Credit <= Credit + $signed(we?  1 : -1);
					Fill   <= Fill   + $signed(we? -1 :  1);
				end
			end
		end

		//=== Memory Array (asynchronous read) ==============================
		(* RAM_STYLE = "distributed" *)
		dat_t  Mem[MEM_SIZE];
		always_ff @(posedge clk) begin
			if(we)  Mem[WrPtr] <= idat;
		end
		uwire dat_t  rd_dat = Mem[RdPtr];

		//=== Output Register Load ==========================================
		always_ff @(posedge clk) begin
			if(rst) begin
				Vld <= 0;
				B   <= 'x;
			end
			else if(bload) begin
				Vld <= drain;
				B   <= rd_dat;
			end
		end

	end : genDist

	// Memory-Backed FIFO: BRAM (stallable pipeline) or URAM (free-running pipeline + output queue)
	"block", "ultra": begin : genMem
		// Geometry (IS_URAM, MIN_ABITS, QDEPTH, DEPTH_REQ, ABITS, MEM_IMPL,
		// PIPE_DEPTH) is declared at module scope, see above.

		// Pointer Type and Increment
		typedef logic [ABITS.lo + (ABITS.hi != 0) -1:0]  ptr_t;
		function ptr_t step(input ptr_t  ptr);
			// The default pointer increment is always +1.
			automatic ptr_t  inc = 1;
			// If there is a narrower high memory space, a big step in the bits beyond
			// its pointer range is made to facilitate the eventual pointer wrap-around.
			// cs | <- ABITS.lo
			//    |     | <- ABITS.hi
			// ---+-----+----------------
			//  0 | 000 | 00000
			//  0 |    ...
			//  0 | 111 | 11111
			//  1 | 000 | 00000  |  step triggered by "10" prefix
			//  1 | 111 | 00001  V
			//  1 | 111 |  ...
			//  1 | 111 | 11111
			if(ABITS.hi && ptr[ABITS.lo:ABITS.lo-1] == 2'b10)  inc[ABITS.lo-1:ABITS.hi] = '1;
			return  ptr + inc;
		endfunction : step

		ptr_t  WrPtr = 0;
		ptr_t  RdPtr = 0;

		// Credit Management
		typedef logic signed [$clog2(MEM_IMPL):0]  cap_t;
		cap_t  Credit = -MEM_IMPL;  // -MEM_IMPL (empty), ..., -1, 0 (full)
		cap_t  Fill   = 0;          // 0 (empty), -1, ..., -MEM_IMPL (full)

		assign	irdy = Credit[$left(Credit)];
		uwire  we = irdy && ivld;

		// Read enable: free-running for URAM, stallable for BRAM
		uwire  re = IS_URAM || (ordy || !ovld);

		//- Drain and Output Queue Credit -----------------------------------
		uwire  drain;
		if(!IS_URAM)  assign  drain = re && Fill[$left(Fill)];
		else begin : genQCredit
			typedef logic signed [$clog2(QDEPTH+1):0]  qcap_t;
			localparam qcap_t  QCREDIT_INIT = -QDEPTH-1;
			qcap_t  QCredit = QCREDIT_INIT;  // -QDEPTH-1 (empty), ..., -1, 0 (full)

			uwire  settle = ovld && ordy;
			always_ff @(posedge clk) begin
				if(rst)                   QCredit <= QCREDIT_INIT;
				else if(drain != settle)  QCredit <= QCredit + $signed(drain? 1 : -1);
			end
			assign	drain = QCredit[$left(QCredit)] && Fill[$left(Fill)];

		end : genQCredit

		//- Pointer and Credit Management -----------------------------------
		always_ff @(posedge clk) begin
			if(rst) begin
				WrPtr  <= 0;
				RdPtr  <= 0;
				Credit <= -MEM_IMPL;
				Fill   <= 0;
			end
			else begin
				if(we)     WrPtr <= step(WrPtr);
				if(drain)  RdPtr <= step(RdPtr);
				if(we != drain) begin
					Credit <= Credit + $signed(we?  1 : -1);
					Fill   <= Fill   + $signed(we? -1 :  1);
				end
			end
		end

		// Write Enables and Addresses
		uwire  wr_lo = we && (!ABITS.hi || !WrPtr[$left(WrPtr)]);
		uwire [ABITS.lo-1:0]  wr_ptr_lo = WrPtr[ABITS.lo-1:0];
		uwire [ABITS.lo-1:0]  rd_ptr_lo = RdPtr[ABITS.lo-1:0];

		// Lo Memory and Read Pipeline
		(* RAM_STYLE = RAM_STYLE_EFF *)
		dat_t  MemLo[2**ABITS.lo];
		dat_t  ODatLo[PIPE_DEPTH];
		always_ff @(posedge clk) begin
			if(wr_lo)  MemLo[wr_ptr_lo] <= idat;
			if(re)     ODatLo <= { ODatLo[1:PIPE_DEPTH-1], MemLo[rd_ptr_lo] };
		end

		// Hi Memory and Read Pipeline
		uwire dat_t  odat_hi;
		if(!ABITS.hi)  assign  odat_hi = 'x;
		else begin : genHi
			// Relax RAM_STYLE for shallow hi spaces
			localparam  RAM_STYLE_HI = (MIN_ABITS <= ABITS.hi)? RAM_STYLE_EFF : "auto";
			uwire  wr_hi = we && WrPtr[$left(WrPtr)];
			uwire [ABITS.hi-1:0]  wr_ptr_hi = WrPtr[ABITS.hi-1:0];
			uwire [ABITS.hi-1:0]  rd_ptr_hi = RdPtr[ABITS.hi-1:0];

			(* RAM_STYLE = RAM_STYLE_HI *)
			dat_t  MemHi[2**ABITS.hi];
			dat_t  ODatHi[PIPE_DEPTH];
			always_ff @(posedge clk) begin
				if(wr_hi)  MemHi[wr_ptr_hi] <= idat;
				if(re)     ODatHi <= { ODatHi[1:PIPE_DEPTH-1], MemHi[rd_ptr_hi] };
			end
			assign	odat_hi = ODatHi[0];
		end : genHi

		// Enable Pipeline for Output Selection
		uwire  osel_lo;
		if(!ABITS.hi)  assign  osel_lo = 1;
		else begin : genOutMux
			logic  OSelLo[PIPE_DEPTH] = '{ default: 'x };
			always_ff @(posedge clk) begin
				if(rst)      OSelLo <= '{ default: 'x };
				else if(re)  OSelLo <= { OSelLo[1:PIPE_DEPTH-1], !RdPtr[$left(RdPtr)] };
			end
			assign	osel_lo = OSelLo[0];
		end : genOutMux

		// Valid Pipeline
		logic  OVld[PIPE_DEPTH] = '{ default: 0 };
		always_ff @(posedge clk) begin
			if(rst)      OVld <= '{ default: 0 };
			else if(re)  OVld <= { OVld[1:PIPE_DEPTH-1], drain };
		end

		// Output Path
		uwire dat_t  mem_odat = osel_lo? ODatLo[0] : odat_hi;
		uwire  mem_ovld = OVld[0];
		if(!IS_URAM) begin : genOutDirect
			assign	odat = mem_odat;
			assign	ovld = mem_ovld;
		end : genOutDirect
		else begin : genOutQ
			uwire  irdy;
			fifo #(.DEPTH(QDEPTH+1), .DATA_WIDTH(DATA_WIDTH), .RAM_STYLE("shift")) outq (
				.clk, .rst,
				.idat(mem_odat), .ivld(mem_ovld), .irdy,
				.odat, .ovld, .ordy,
				.count(), .maxcount()
			);
			always_ff @(posedge clk) begin
				assert(irdy || !mem_ovld) else begin
					$error("%m: Overrunning output queue.");
					$stop;
				end
			end
		end : genOutQ

	end : genMem
	endcase

`default_nettype wire
endmodule : fifo

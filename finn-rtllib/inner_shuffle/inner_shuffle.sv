/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	A streaming 2D parallel transpose unit with SIMD parallelism.
 * @author	Shane T. Fleming <shane.fleming@amd.com>
 *
 * @description
 *
 * This unit can perform a streaming transpose (I,J) -> (J,I) with SIMD
 * parallelism.
 * It achieves this by using SIMD banks of memory and rotating write and reads
 * to the banks such that collisions are avoided and maximum throughput can be
 * maintained (II=1).
 *
 * Decisions about when to rotate writes and reads to the different banks are
 * made by a WR_ROT_PERIOD param, for writes, and a RD_PATTERN param matrix, for reads.
 * These two are computed at elaboration time and are constants at runtime.
 *
 * After WR_ROT_PERIOD writes to the banks the write bank allocation is shifted to
 * the right by one position.
 * The WR_ROT_PERIOD is determined by considering the GCD of SIMD
 * along with the inner input dimension J.
 *
 * The RD_PATTERN for the read side is a SIMDxSIMD matrix of banks that is a
 * periodic pattern of banks across the input matrix. This is computed by
 * evaluating what a SIMDxSIMD block of bank allocations will look like with
 * the current WR_ROT_PERIOD.
 *
 * On the write path of the hardware data is written into the banks according
 * to the initial write banks. A counter tracks how many writes have happened
 * and then after WR_ROT_PERIOD counts the banks are rotated. The write
 * address is incremented by one every write for every bank.
 *
 * The Read path has logic to generate the addresses for SIMD reads based on
 * the current index of the output loop:
 *
 *        	j : [0,J)
 *        	   i : [0,I)
 *        	     emit(i*J + j)
 *
 * SIMD addresses are generated and each is sent to the appropriate SIMD banks
 * based on the schedule in the relevant column of the RD_PATTERN matrix.
 * This column of the RD_PATTERN matrix is then forwarded to the output of the
 * banks, where a clock cycle later the relevant outputs appear at each bank
 * output. The output data is then rearranged again using the forwarded RD_PATTERN
 * column to assign the appropriate output signals.
 * Logic is used to track what column of the the RD_PATTERN to use based
 * on where the circuit current is in the output iteration space.
 *
 * Control flow for writing and reading the banks are managed by job
 * scheduling logic. This means that while a job is being
 * outputted on the read side, the next job can be written on the write side
 * enabling both the write path and the read path to be active simultaneously.
****************************************************************************/

// A memory bank in the inner_shuffle design. Pattern was kept as simple
// as possible to help with Vivado BRAM inference.
module mem_bank #(
	int unsigned WIDTH,
	int unsigned DEPTH,
	parameter RAM_STYLE = "auto"
)(
	input logic clk,

	input logic [WIDTH-1:0] d_in,
	input logic [$clog2(DEPTH)-1:0] wr_addr,
	input logic wr_en,

	output logic [WIDTH-1:0] d_out,
	input  logic [$clog2(DEPTH)-1:0] rd_addr,
	input  logic rd_en
);

	(* ram_style=RAM_STYLE *)
	logic [WIDTH-1:0] Mem [DEPTH-1:0]; // The Mem for this bank
	always_ff @(posedge clk) begin
		if(wr_en)  Mem[wr_addr] <= d_in;
		if(rd_en)  d_out <= Mem[rd_addr];
	end

endmodule : mem_bank


// ----------------------------------------
// Parallel Transpose Unit (InnerShuffle)
// ----------------------------------------
module inner_shuffle #(
	int unsigned BITS,   // Bitwidth of each element
	int unsigned I   ,   // Input dimension I
	int unsigned J   ,   // Input dimension J
	int unsigned SIMD,   // SIMD parallelism
	parameter RAM_STYLE = "auto"
)(
	input logic                       clk, // global control
	input logic                       rst,

	output logic                      irdy, // Input stream
	input  logic                      ivld,
	input  logic [SIMD-1:0][BITS-1:0] idat,

	input  logic                      ordy, // Output stream
	output logic                      ovld,
	output logic [SIMD-1:0][BITS-1:0] odat
);


	// assertion checks for ensuring that the constraints are satisfied
	initial begin
		if (I%SIMD != 0) begin
			$fatal(1, "Error! Assertion I%SIMD == 0 is not met for this circuit");
		end
	end

	function int unsigned gcd(input int unsigned  a, input int unsigned  b);
		return (b == 0) ? a : gcd(b, a%b);
	endfunction : gcd

	//=======================================================================
	// Generates the SIMD dual port memory banks
	localparam int unsigned  BANK_DEPTH  = 2* I*J / SIMD;
	localparam int unsigned  PAGE_OFFSET =    I*J / SIMD;
	uwire  wr_en = irdy && ivld;
	uwire [$clog2(BANK_DEPTH)-1:0]  wr_addr;
	uwire  rd_en;
	uwire [BITS-1:0]                d_in      [SIMD-1:0];
	uwire [BITS-1:0]                d_out     [SIMD-1:0];
	logic [$clog2(BANK_DEPTH)-1:0]  raddr     [SIMD-1:0];
	for(genvar i = 0; i<SIMD; i++) begin : gen_mem_banks
		mem_bank #(
			.WIDTH(BITS),
			.DEPTH(BANK_DEPTH),
			.RAM_STYLE(RAM_STYLE)
		) mem_bank_inst (
			.clk,
			.wr_en,
			.wr_addr,
			.d_in(d_in[i]),
			.rd_en,
			.rd_addr(raddr[i]),
			.d_out(d_out[i])
		);
	end : gen_mem_banks

	//=======================================================================
	// WRITE Control

	//-----------------------------------------------------------------------
	// Sequential write address increment (resets after completing second page)
	logic [$clog2(BANK_DEPTH)-1:0]  WrAddr = 0;	// 0, ..., 2*PAGE_OFFSET - 1
	always_ff @(posedge clk) begin : wr_addr_logic
		if(rst)  WrAddr <= 0;
		else if(wr_en) begin
			automatic logic  wrap = ((2*PAGE_OFFSET - 1) & ~WrAddr) == 0;
			WrAddr <= WrAddr + (wrap? 1-2*PAGE_OFFSET : 1);
		end
	end : wr_addr_logic
	assign	wr_addr = WrAddr;

	//-----------------------------------------------------------------------
	// Write Bank Schedule
	//	- The input is ingested in row-major order, SIMD elements at a time.
	//	- Each column in a block of SIMD consecutive rows is to be read out
	//	  in parallel and must hence distribute its elements across all banks.
	//	- A schedule for the J consecutive write operations is needed that
	//	  define the bank layout within these blocks of rows.
	//	- When maintaining the same bank assignment for the horizontal writes,
	//	  banks realign in a column after writing lcm(J, SIMD) values.
	//	- This corresponds to lcm(J , SIMD)/SIMD = J/gcd(J, SIMD) writes.
	//	- A stable bank assignment can hence be maintained for gcd(J, SIMD) = 1.
	//	- Otherwise, a unit rotation is used to desync the bank alignment.
	//	- Across the block of SIMD rows, gcd(J, SIMD) bank rotations will come
	//	  to be used.
	localparam int unsigned  WR_ROTATIONS = gcd(J, SIMD);	// 1 <= _ <= J

	if(WR_ROTATIONS == 1) begin : gen_no_write_rotation
		// Straight-thru Input Feed
		for(genvar  i = 0; i < SIMD; i++)  assign  d_in[i] = idat[i];
	end : gen_no_write_rotation
	else begin : gen_write_rotation

		localparam int unsigned  WR_ROT_PERIOD = J / WR_ROTATIONS;

		// Bank Rotation Requesting
		uwire  rotate;
		if(WR_ROT_PERIOD == 1)  assign  rotate = 1;
		else begin
			logic signed [$clog2(WR_ROT_PERIOD-1):0]  RepCnt = WR_ROT_PERIOD-2;	// WR_ROT_PERIOD-2, ..., 0, -1
			assign	rotate = RepCnt[$left(RepCnt)];
			always_ff @(posedge clk) begin
				if(rst)         RepCnt <= WR_ROT_PERIOD-2;
				else if(wr_en)  RepCnt <= RepCnt + (rotate? WR_ROT_PERIOD-1 : -1);
			end
		end

		// Bank Rotation Identification
		logic [$clog2(WR_ROTATIONS)-1:0]  RotCnt = 0;	// 0, ..., WR_ROTATIONS-1
		uwire  wrap = ((WR_ROTATIONS-1) & ~RotCnt) == 0;
		always_ff @(posedge clk) begin
			if(rst)  RotCnt <= 0;
			else if(wr_en && rotate) begin
				RotCnt <= RotCnt + (wrap? -WR_ROTATIONS+1 : 1);
			end
		end

		// Input Feed rotated by Offset given by RotCnt
		for(genvar  i = 0; i < SIMD; i++) begin
			// Wired re-indexing of sources starting from local position
			uwire [BITS-1:0]  srcs[WR_ROTATIONS];
			for(genvar  ofs = 0; ofs < WR_ROTATIONS; ofs++) begin
				assign	srcs[ofs] = idat[(SIMD+i-ofs)%SIMD];
			end
			assign	d_in[i] = srcs[RotCnt];
		end
	end : gen_write_rotation

	//=======================================================================
	// READ Control

	localparam int unsigned  RD_ROT_PERIOD = I / SIMD; // (I % SIMD == 0) is a constraint
	typedef logic [$clog2(SIMD)-1:0]  rotidx_vec_t[SIMD-1:0];
	typedef logic [$clog2(BANK_DEPTH)-1:0]  bank_addr_t [SIMD-1:0];

	// --------------------------------------------------------------------------
	// RD_INITIAL_PATTERN & RD_PERMUTATION_PATTERN

	function automatic rotidx_vec_t generate_initial_rd_pattern();
		rotidx_vec_t  rd_pat_0; // The RD Pattern for the first column
		foreach(rd_pat_0[i]) begin
			rd_pat_0[i] = (i*J + (i*WR_ROTATIONS)/SIMD) % SIMD;
		end
		return  rd_pat_0;
	endfunction : generate_initial_rd_pattern
	localparam rotidx_vec_t  RD_INIT_PAT = generate_initial_rd_pattern();

	function automatic rotidx_vec_t generate_initial_rev_rd_pattern();
		rotidx_vec_t rd_pat_0 = RD_INIT_PAT;
		rotidx_vec_t rev_rd_pat_0;
		for(int i = 0; i< SIMD; i++)
			for( int j = 0; j<SIMD; j++)
				if( rd_pat_0[j] == i) begin
					rev_rd_pat_0[i] = j;
					break;
				end
		return  rev_rd_pat_0;
	endfunction : generate_initial_rev_rd_pattern
	localparam rotidx_vec_t  REV_RD_INIT_PAT = generate_initial_rev_rd_pattern();

	function automatic rotidx_vec_t generate_rd_permutation_pattern(logic reverse);
		rotidx_vec_t perm_pattern;

		rotidx_vec_t rd_pat_0 = RD_INIT_PAT;
		rotidx_vec_t rd_pat_1; // The RD Pattern for the second column
		foreach(rd_pat_1[i])  rd_pat_1[i] = (rd_pat_0[i] + 1) % SIMD;

		// Calculate permutation indices
		foreach (rd_pat_0[i])
			foreach (rd_pat_1[j])
				if (rd_pat_0[i] == rd_pat_1[j]) begin
					if (reverse == 1)
						perm_pattern[j] = i;
					else
						perm_pattern[i] = j;
					break;
				end
		return perm_pattern;
	endfunction : generate_rd_permutation_pattern
	localparam rotidx_vec_t  RD_PERM_PAT = generate_rd_permutation_pattern(0);
	localparam rotidx_vec_t  REV_RD_PERM_PAT = generate_rd_permutation_pattern(1);

	// --------------------------------------------------------------------------
	//    Read Address generation

	// Job tracking and bank page locking
	logic [1:0] WrJobsDone = 2'b00; // Bit vector tracking when writes have been completed to pages
	logic CurrentPageRd = 0;     // 0 - reading from PAGE A, 1 - reading from PAGE B
	uwire [$clog2(BANK_DEPTH)-1:0] page_rd_offset;
	uwire osb_rdy; // output skid buffer ready signal
	uwire rd_guard = !CurrentPageRd && !WrJobsDone[0] && !WrJobsDone[1];
	uwire rd_inc = osb_rdy & !rd_guard;

	// Counts reads across the columns
	logic[$clog2(I)-1 : 0] RdICnt = 0; // 0, ..., I - 1
	uwire col_read = ((I-SIMD) & ~RdICnt) == 0;
	always_ff @(posedge clk) begin: rdidx_i_counters
		if(rst) RdICnt <= 0;
		else if (rd_inc) begin
			RdICnt <= RdICnt + (col_read ? -(I-SIMD) : SIMD);
		end
	end : rdidx_i_counters

	// Counts reads across the rows
	logic[$clog2(J)-1 : 0] RdJCnt = 0; // 0, ..., J - 1
	always_ff @(posedge clk) begin: rdidx_j_counters
		if(rst) RdJCnt <= 0;
		else if (rd_inc & col_read) begin
			automatic logic wrap = ((J-1) & ~RdJCnt) == 0;
			RdJCnt <= RdJCnt + (wrap ? -(J-1) : 1);
		end
	end : rdidx_j_counters

	// Read bank muxing
	uwire [$clog2(SIMD)-1:0] t_srcs[SIMD][SIMD];
	for (genvar i = 0; i < SIMD; i++) begin
    		assign t_srcs[i][0] = REV_RD_INIT_PAT[i];
    		for (genvar ofs = 1; ofs < SIMD; ofs++)
        		assign t_srcs[i][ofs] = REV_RD_INIT_PAT[(i + (SIMD - ofs)) % SIMD];
		assign raddr[i] = read_addr[t_srcs[i][RdJCnt%SIMD]];
	end

	// Initial read addresses
	function automatic bank_addr_t init_rdaddr();
		bank_addr_t a;
		foreach(a[i])
			a[i] = (i*J)/SIMD;
		return  a;
	endfunction : init_rdaddr
	localparam bank_addr_t  RD_ADDR_INIT = init_rdaddr();

	localparam int unsigned WR_ROT_RADDR = SIMD/WR_ROTATIONS;
	uwire page_boundary = (((J-1) & ~RdJCnt) == 0) & (((I-SIMD) & ~RdICnt) == 0);
	uwire rdaddr_hor_inc[SIMD]; // tracks the horizontal increments for the rdaddr
	uwire [$clog2(BANK_DEPTH)-1:0] rdaddr_vert_inc = (RdICnt != 0) ? J : 0;
	uwire [$clog2(BANK_DEPTH)-1:0] rdaddr_col_dec = (RdICnt == 0) && (RdJCnt !=0 )  ? ((I/SIMD) -1)*J : 0;
	uwire [$clog2(BANK_DEPTH)-1:0] read_addr [SIMD-1:0];

	// Read address update logic
	for(genvar i = 0; i < SIMD; i++)
		assign rdaddr_hor_inc[i] = ((RdJCnt != 0) && (RdICnt == 0) && ((RdJCnt + i*(J%SIMD))%SIMD)==0);

	for(genvar i=0; i<SIMD; i++)
		assign read_addr[i] = ReadAddrReg[i] + rdaddr_hor_inc[i] + rdaddr_vert_inc - rdaddr_col_dec;

	bank_addr_t ReadAddrReg = RD_ADDR_INIT;
	always_ff @(posedge clk)
		if(rst) ReadAddrReg <= RD_ADDR_INIT;
		else
			if (rd_inc) begin
				ReadAddrReg <= read_addr;
				if(page_boundary)
					for(int i=0; i<SIMD; i++)
						ReadAddrReg[i] <= RD_ADDR_INIT[i] + page_rd_offset;
			end
	// --------------------------------------------------------------------------


	// --------------------------------------------------------------------------
	//   Page management

	logic OsbVld  = 0; // output skidbuffer valid signal
	logic OsbVld_D = 0; // output skidbuffer valid signal
	assign rd_en  = osb_rdy;

	always_ff @(posedge clk) begin
		if (rst) begin
			WrJobsDone <= 2'b00;
			CurrentPageRd <= 0;
		end else begin
			// Track if we have completed a job
			if (wr_addr == PAGE_OFFSET   - 1) WrJobsDone[0] <= 1;
			if (wr_addr == 2*PAGE_OFFSET - 1) WrJobsDone[1] <= 1;

			// Clear the relevant job once it is read
			if (page_boundary && (osb_rdy && OsbVld)) begin
				WrJobsDone[CurrentPageRd] <= 0;
		       		CurrentPageRd <= !CurrentPageRd;
			end
		end
	end

	assign page_rd_offset = CurrentPageRd ? 0: PAGE_OFFSET;
	assign irdy = !WrJobsDone[0] || !WrJobsDone[1];

	// Forward the current RD_PATTERN row onto the next pipeline stage
	rotidx_vec_t RdPat = RD_INIT_PAT;
	rotidx_vec_t RdPat_D = RD_INIT_PAT; // The fowarded rotation pattern
	always_ff @(posedge clk) begin : rd_pattern_col_forwarding
		if (rst) begin
			OsbVld <= 0;
			RdPat_D <= RD_INIT_PAT;
		end
		else begin
			OsbVld <= !rd_guard;
			OsbVld_D <= OsbVld;
			if (rd_inc) RdPat_D <= RdPat;
			if(OsbVld & rd_guard & !osb_rdy) OsbVld <= 1;
		end
	end : rd_pattern_col_forwarding

	// Structural remapping using the output of the memory banks
	// and the Read rotation from the previous clock cycle that was
	// used to generate the read addresses.
    	uwire [SIMD-1:0][BITS-1:0] remapped_data; // remapped output
	for(genvar i=0; i<SIMD; i++) assign remapped_data[i] = d_out[RdPat_D[i]];

	// the next permutation of the rd pattern
	rotidx_vec_t rd_pat_next;
	for(genvar i=0; i<SIMD; i++) assign rd_pat_next[i] = RdPat[REV_RD_PERM_PAT[i]];

	//  Read Counter for Rotation tracking
	logic [$clog2(RD_ROT_PERIOD)-1:0] RdCnt = 0; // 0, ... , RD_ROT_PERIOD-1
	uwire rd_rotate = ((RD_ROT_PERIOD-1) & ~RdCnt) == 0;
	always_ff @(posedge clk) begin: rd_counter
		if(rst) RdCnt <= 0;
		else if (rd_inc) begin
			automatic logic wrap = rd_rotate | page_boundary;
			RdCnt <= RdCnt + (rd_rotate ? -(RD_ROT_PERIOD-1) : 1);
		end
	end : rd_counter

	// Read Col count : tracks how many columns have been read
	logic [$clog2(SIMD)-1:0] RdColCnt = 0; // 0, ... , SIMD-1
	uwire col_wrap = ((SIMD-1) & ~RdColCnt) == 0;
	always_ff @(posedge clk) begin : rd_col_counter
		if (rst) RdColCnt <= 0;
		else if (rd_inc & rd_rotate) begin
			RdColCnt <= (page_boundary | col_wrap) ? -(I-1) : 1;
		end
	end : rd_col_counter

	// Assign next pattern state
	always_ff @(posedge clk) begin : rd_pattern_assignment
		if(rst)
			RdPat <= RD_INIT_PAT;
		else begin
			if (rd_inc) begin
				if (rd_rotate)
					RdPat <= rd_pat_next;
				if (page_boundary)
					RdPat <= RD_INIT_PAT;
			end
		end
	end : rd_pattern_assignment
	// --------------------------------------------------------------------------

	//=======================================================================
	// Output SkidBuffer -- Used to decouple control signals for timing
	// improvements
	skid #(
		.DATA_WIDTH(SIMD*BITS)
	)
	oskidbf_inst (
		.clk(clk),
		.rst(rst),

		.idat(remapped_data),
		.ivld(OsbVld),
		.irdy(osb_rdy),

		.odat(odat),
		.ovld(ovld),
		.ordy(ordy)
	);

endmodule : inner_shuffle

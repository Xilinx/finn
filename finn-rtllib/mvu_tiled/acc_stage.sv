/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/


module acc_stage #(
	int unsigned  CHAINLEN,
	int unsigned  PE,
	int unsigned  ACCU_WIDTH,
	int unsigned  TH
)(
	input  logic  clk,
	input  logic  rst,
	input  logic  en,

	input  logic [PE-1:0][CHAINLEN-1:0][ACCU_WIDTH-1:0]  idat,
	input  logic  ival,
	input  logic  ilast,

	output logic [PE-1:0][ACCU_WIDTH-1:0]  odat,
	output logic  oval
);

	//=== Adder Tree + Accumulator Add ======================================
	localparam int unsigned  TREE_DEPTH = $clog2(CHAINLEN);
	localparam int unsigned  ADD_LAT    = TREE_DEPTH + 1;

	uwire [PE-1:0][ACCU_WIDTH-1:0]  Acc;
	logic [PE-1:0][ACCU_WIDTH-1:0]  DatInt;
	uwire [PE-1:0][ACCU_WIDTH-1:0]  Sum;

	for(genvar  i = 0; i < PE; i++) begin : genAdd
		// Tree reduction of CHAINLEN DSP partial products
		logic [ACCU_WIDTH-1:0]  add_arg[CHAINLEN];
		for(genvar  k = 0; k < CHAINLEN; k++)
			assign  add_arg[k] = idat[i][k];

		localparam int unsigned  SUM_WIDTH = $clog2(CHAINLEN) + ACCU_WIDTH;
		uwire [SUM_WIDTH-1:0]  tree_sum;
		add_multi #(.N(CHAINLEN), .DEPTH(TREE_DEPTH), .ARG_WIDTH(ACCU_WIDTH)) inst_add (
			.clk(clk), .rst(rst), .en(en),
			.arg(add_arg),
			.sum(tree_sum)
		);

		// Accumulator add (1 registered stage)
		assign  Sum[i] = tree_sum[ACCU_WIDTH-1:0] + Acc[i];
		always_ff @(posedge clk) begin
			if(rst)       DatInt[i] <= 'x;
			else if(en)   DatInt[i] <= Sum[i];
		end
	end : genAdd

	//=== Valid/Last Pipeline ===============================================
	logic [ADD_LAT:0]  Val;
	logic [ADD_LAT:0]  Last;

	assign  Val[0]  = ival;
	assign  Last[0] = ilast;

	always_ff @(posedge clk) begin
		if(rst) begin
			for(int  i = 1; i <= ADD_LAT; i++) begin
				Val [i] <= 0;
				Last[i] <= 'x;
			end
		end
		else if(en) begin
			for(int  i = 1; i <= ADD_LAT; i++) begin
				Val [i] <= Val [i-1];
				Last[i] <= Last[i-1];
			end
		end
	end

	uwire  val_out  = Val[ADD_LAT];
	uwire  last_out = Last[ADD_LAT];
	uwire  inc_acc  = Val[ADD_LAT-1];

	//=== Accumulation Delay Line ===========================================
	//	Rotates TH partial sums rather than queueing them. Advances on the
	//	accumulator read (inc_acc), one cycle ahead of the registered DatInt,
	//	so it shifts in the combinational Sum. Unreset, to infer SRLs: the
	//	first TH reads are forced to the zero the accumulation starts from.
	uwire  acc_shift = en && inc_acc;
	uwire [PE-1:0][ACCU_WIDTH-1:0]  acc_din = Last[ADD_LAT-1]? '0 : Sum;

	logic signed [$clog2(TH):0]  CntPrep = -TH;
	uwire  prep = CntPrep[$left(CntPrep)];
	always_ff @(posedge clk) begin
		if(rst)             CntPrep <= -TH;
		else if(acc_shift)  CntPrep <= CntPrep + prep;
	end

	(* SHREG_EXTRACT = "yes" *) logic [PE-1:0][ACCU_WIDTH-1:0]  AccLine[TH];
	always_ff @(posedge clk) begin
		if(acc_shift) begin
			AccLine[0] <= acc_din;
			for(int  i = 1; i < TH; i++)  AccLine[i] <= AccLine[i-1];
		end
	end
	assign  Acc = prep? '0 : AccLine[TH-1];

	//=== Output Stage ======================================================
	always_ff @(posedge clk) begin
		if(rst) begin
			odat <= 'x;
			oval <= 0;
		end
		else if(en) begin
			odat <= DatInt;
			oval <= val_out && last_out;
		end
	end

endmodule : acc_stage

module dwc #(
	type  T = logic[7:0],
	int unsigned  ICNT = 5,
	int unsigned  OCNT = 7
)(
	input	logic  clk,
	input	logic  rst,

	output	logic  irdy,
	input	logic  ivld,
	input	T [ICNT-1:0]  idat,

	input	logic  ordy,
	output	logic  ovld,
	output	T [OCNT-1:0]  odat
);

	function int unsigned gcd(int unsigned  a, int unsigned  b);
		while(b != 0) begin
			automatic int unsigned  t = b;
			b = a % b;
			a = t;
		end
		return  a;
	endfunction

	localparam int unsigned  GCD = gcd(ICNT, OCNT);
	typedef T [GCD-1:0]  T0;
	localparam int unsigned  ICNT0 = ICNT / GCD;
	localparam int unsigned  OCNT0 = OCNT / GCD;


	localparam int unsigned  BSIZE = ICNT0 + OCNT0 - 1;
	typedef logic signed [$clog2(BSIZE+1):0]  cnt_t;
	typedef logic        [$clog2(ICNT0)-1:0]  idx_t;

	T0 [ICNT0-1:0]  ADat = 'x;
	logic  ARdy = 1;

	T0 [BSIZE-1:0]  BDat = 'x;
	cnt_t  BCnt = OCNT0 - 1;

	uwire T0 [ICNT0-1:0]  idat_eff = ARdy? idat : ADat;
	uwire T0 [BSIZE-1:0]  idat_ext = { {((BSIZE-ICNT0)*$bits(T0)){1'bx}}, idat_eff };

	uwire  free = BCnt >= $signed(pop? ICNT0-BSIZE-1 : ICNT0+OCNT0-BSIZE-1);
	uwire  push = (ivld || !ARdy) && free;
	uwire  pop  = ovld && ordy;
	always_ff @(posedge clk) begin
		if(rst) begin
			ADat <= 'x;
			ARdy <=  1;
			BDat <= 'x;
			BCnt <= OCNT0 - 1;
		end
		else begin
			if(ARdy)  ADat <= idat;
			ARdy <= (ARdy && !ivld) || free;

			for(int  i = 0; i < BSIZE; i++) begin
				automatic idx_t  idx0 =  i+1          + BCnt;
				automatic idx_t  idx1 = (i+1 - OCNT0) + BCnt;
				BDat[i] <= pop?
							$signed(      -i-1) > BCnt? BDat[i + OCNT0] : idat_ext[idx0] :
							$signed(OCNT0 -i-1) > BCnt? BDat[i] : idat_ext[idx1];
			end
			BCnt <= BCnt + (push? (pop? OCNT0-ICNT0 : -ICNT0) : (pop? OCNT0 : 0));
		end
	end

	assign	irdy = ARdy;
	assign	ovld = BCnt[$left(BCnt)];
	assign	odat = BDat[OCNT0-1:0];

endmodule : dwc

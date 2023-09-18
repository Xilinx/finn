module mvu_lut #(
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  ACCU_WIDTH,
    int unsigned  ACTIVATION_WIDTH,
    int unsigned  WEIGHT_WIDTH,
    bit  SIGNED_ACTIVATIONS,
    bit  M_REG = 1,

    localparam unsigned MULT_WIDTH = ACTIVATION_WIDTH+WEIGHT_WIDTH
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	// Input
	input	logic  last,
	input	logic  zero,	// ignore current inputs and force this partial product to zero
	input	logic signed [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]      w,	// signed weights
	input	logic                [SIMD-1:0][ACTIVATION_WIDTH-1:0]  a,	// (un)signed activations

	// Ouput
	output	logic  vld,
	output	logic signed [PE-1:0][ACCU_WIDTH-1:0]  p
);

    typedef int unsigned  leave_load_t[2*SIMD-1];
    function leave_load_t init_leave_loads();
        automatic leave_load_t  res;
        for(int  i = 2*(SIMD-1); i >= int'(SIMD)-1; i--)  res[i] = 1;
        for(int  i = SIMD-2; i >= 0; i--)  res[i] = res[2*i+1] + res[2*i+2];
        return res;
    endfunction : init_leave_loads

    // Pipeline for last indicator flag
    uwire last_i;
    generate if (M_REG) begin
        logic [0:1] L = '0;
        always_ff @(posedge clk) begin
            if(rst)       L <= '0;
            else if (en)  L <= {last, L[0]};
        end
        assign  last_i = L[1];
    end
    else begin 
        logic L = '0;
        always_ff @(posedge clk) begin
            if(rst)       L <= '0;
            else if (en)  L <= last;
        end
        assign  last_i = L;
    end
    endgenerate

    // For each PE generate
    for (genvar  i = 0; i < PE; i++)  begin : genPE
        // Stage #1: SIMD multipliers in parallel
        uwire [MULT_WIDTH-1 : 0] m1 [SIMD];
        for (genvar j = 0; j < SIMD; j++) begin : genSIMD
            if (M_REG) begin : genMreg
                logic [MULT_WIDTH-1 : 0] M [SIMD];
                always_ff @(posedge clk) begin
                    if(rst)         M[j] = '{ default : 0 };
                    else if (en)    M[j] = zero ? 0 :
                                            SIGNED_ACTIVATIONS ? $signed(a[j]) * $signed(w[i][j]) :
                                                                 $signed({1'b0, a[j]}) * $signed(w[i][j]); 
                    // (SIGNED_ACTIVATIONS ? $signed(a[j]) : a[j]) * $signed(w[i][j]) isn't valid -- leads to unsigned multiplication
                end
                assign  m1[j] = M[j];
            end : genMreg
            else begin : genNoMreg 
                assign m1[j] = zero ? 0 :
                               SIGNED_ACTIVATIONS ? $signed(a[j]) * $signed(w[i][j]) :
                                                    $signed({1'b0, a[j]}) * $signed(w[i][j]);
            end : genNoMreg
        end : genSIMD

        // Stage #2: Adder tree to reduce SIMD products
        localparam leave_load_t  LEAVE_LOAD = SIMD > 1 ? init_leave_loads() : '{ default : 1 };
        localparam int unsigned  ROOT_WIDTH = $clog2(SIMD*(2**MULT_WIDTH-1));
        uwire signed [2*SIMD-2:0][ROOT_WIDTH-1:0]  tree;
        for(genvar s = 0; s < SIMD; s++)  assign  tree[SIMD-1+s] = $signed(m1[s]);
        for(genvar n = 0; n < SIMD-1; n++) begin
            // Sum truncated to actual maximum bit width at this node
            localparam int unsigned  NODE_WIDTH = $clog2(LEAVE_LOAD[n]*(2**MULT_WIDTH-1));
            uwire signed [NODE_WIDTH-1:0]  s = tree[2*n+1] + tree[2*n+2];
            assign tree[n] = s;
        end

        // Stage #3: Buffer output
        logic [ACCU_WIDTH-1:0] P2 [PE];
        always_ff @(posedge clk) begin
            if(rst)         P2[i] = '{ default : 0};
            else if (en)    P2[i] = (last_i ? 0 : $signed(P2[i])) + $signed(tree[0]);
        end

        assign  vld = last_i;
        assign  p[i] = P2[i];
    end : genPE

endmodule : mvu_lut

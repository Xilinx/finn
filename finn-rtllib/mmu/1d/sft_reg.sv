module sft_reg #(
    int N = 4,
    int DATA_BITS = 24
)(
    input  logic        clk,
    input  logic        ivld,
    input  logic [DATA_BITS-1:0] din,
    output logic [DATA_BITS-1:0] dout
);

    // A 2D array representing the shift stages
    logic [N-1:0][DATA_BITS-1:0] shift_pipe;

    always_ff @(posedge clk) begin
        if (ivld) begin
            // Shift the bits in
            shift_pipe <= {shift_pipe[N-2:0], din};
        end
    end

    // The tool sees this lack of reset and constant index 
    // and maps it to an SRL16 automatically.
    assign dout = shift_pipe[N-1];

endmodule
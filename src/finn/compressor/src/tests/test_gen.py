from ..utils.shape import Shape
from itertools import accumulate
from typing import List

def compressed_width(shape):
    max = sum([col * (1 << idx) for idx, col in enumerate(shape)])
    return max.bit_length()

def flatten_gates(gates: List[List[str]]) -> List[str]:
    return [el for col in gates for el in col]

def generate_test(shape: Shape, module_name: str, pipeline_stages: int, 
                  gates: List[List[str]], accumulation: bool, accumulator_width: int,
                  constant: int):
    assert(type(pipeline_stages) == int)

    if gates:
        gates = flatten_gates(gates)

    has_clk = bool(pipeline_stages) or accumulate

    accumulated_signature = list(accumulate(shape))
    addends = []
    for j, col in enumerate(accumulated_signature):
        for i in range(shape[j]):
            addends.append(f"\t\tin_reduced += arr_in[{col+i-shape[j]}] << {j};")
    addends = "\n".join(addends)

    if gates:
        preds = "".join([f"\tlocalparam pred_{idx} = 4'h{gate};\n" 
                         for idx, gate in enumerate(gates)])
        selects = "".join([f"\tlogic [3:0] sel_{idx};\n" 
                           for idx, _ in enumerate(gates)])
        arr_ins = "".join([
            f"\t\tsel_{i} = (arr_in_b[{i}]<<1) | arr_in_a[{i}];\n" + 
            f"\t\tarr_in[{i}] = pred_{i}[sel_{i}];\n"
            for i, _ in enumerate(gates)])
        gates_decl = (f"\tlogic [{sum(shape)-1}:0] arr_in_a;" + 
                      f"\tlogic [{sum(shape)-1}:0] arr_in_b;")
    accumulator_width = (accumulator_width if accumulator_width 
                         else compressed_width(shape))
    acc_decl = f"\tlogic [{accumulator_width-1}:0] acc_base;"

    acc_rst_block = """\t\t\tif (reset == 0) begin 
\t\t\t\tacc_base = 0;
\t\t\tend else begin 
\t\t\t\tacc_base = reference[0];
\t\t\tend"""

    return (
f"""module tb;
{gates_decl if gates else ""}
\tlogic [{sum(shape)-1}:0] arr_in;
\tlogic [{compressed_width(shape)-1}:0] in_reduced;
\tlogic [{accumulator_width-1}:0] out;
\tlogic [{accumulator_width-1}:0] reference [{pipeline_stages}:0]; 
{acc_decl if accumulation else ""}
\t{"logic [4:0] reset;" if accumulation else ""}
\t{"logic rst;" if accumulation else ""}
\t{"logic clk = 0;" if has_clk else ""}
\t{"logic en = 1;" if accumulation else ""}

{preds if gates else ""}
{selects if gates else ""}
\talways_comb begin;
{arr_ins if gates else ""}
\tend

\t{"always #10ns clk = !clk;" if has_clk else ""}

\talways_comb begin 
\t\t{"reference[0] = acc_base + in_reduced;" 
     if accumulation else "reference[0] = in_reduced;"}
\tend

\talways_comb begin 
\t\tin_reduced = 0;
\t\t{"if (en) begin" if accumulation else ""}
in_reduced += {constant};
{addends}
\t\t{"end" if accumulation else ""}
\tend
           
\tinitial begin
\t\t{"acc_base = 0;" if accumulation else ""}
\t\t{"arr_in_a = 0;" if gates else "arr_in = 0;"}
\t\t{"arr_in_b = 0;" if gates else ""}
      
\t\t{"assign rst = reset == 0;" if accumulation else ""}
\t\t{"reset = 0; #40ns;" if accumulation else ""}
        
\t\tfor (int i = 0; i < 16000; i += 1) begin
\t\t\t{"automatic type(reset) xx;" if accumulation else ""}
\t\t\t{"automatic type(en) zz;" if accumulation else ""}

\t\t\t{"automatic type(arr_in_a) yy;" if gates else "automatic type(arr_in) yy;"}
\t\t\t{"automatic type(arr_in_b) yz;" if gates else ""}

\t\t\t{"void'(std::randomize(xx));" if accumulation else ""}
\t\t\t{"reset = xx; " if accumulation else ""}
\t\t\t{"void'(std::randomize(zz));" if accumulation else ""}
\t\t\t{"en = zz;" if accumulation else ""}

\t\t\tif (i < 5) yy = 0;
\t\t\telse if (i < 10) yy = '1;
\t\t\telse void'(std::randomize(yy));
\t\t\t{"arr_in_a = yy;" if gates else "arr_in = yy;"}

\t\t\t{"if (i < 5) yz = 0;" if gates else ""}
\t\t\t{"else if (i < 10) yz = '1;" if gates else ""}
\t\t\t{"else void'(std::randomize(yz));" if gates else ""}
\t\t\t{"arr_in_b = yz;" if gates else ""}

\t\t\t@(posedge clk);
\t\t\tfor (int i = 1; i <= {pipeline_stages}; ++i) begin
\t\t\t\treference[i] <= reference[i-1];
\t\t\tend

{acc_rst_block if accumulation else ""}
\t\t\t#1ns;
\t\t\tif(^reference[{pipeline_stages}] !== 1'bX) begin
\t\t\t\tassert(reference[{pipeline_stages}] === out) else begin
\t\t\t\t\t$error("Mismatch: Ref[%0b] != Out[%0b]", reference[{pipeline_stages}], out);
\t\t\t\t\t#2ns;
\t\t\t\t\t$stop;
\t\t\t\tend 
\t\t\tend
\t\t#0.01ns;
        
\t\tend
\t\t$display("TEST PASSED");
\t\t$finish();
\tend

\t{module_name} dut(
    {".clk(clk)," if pipeline_stages or accumulation else ""}
    {".rst(rst)," if accumulation else ""}
    {".in(arr_in_a), .in_2(arr_in_b)," if gates else ".in(arr_in),"}
    {".en_neg(!en)," if accumulation else ""}
    .out(out));
endmodule
""").replace("\n\n", "\n")
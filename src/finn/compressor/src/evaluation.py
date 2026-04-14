from .target import Versal
from .utils.shape import Shape
from .main import generate_compressor
from .tests.test_gen import compressed_width
from concurrent.futures import ThreadPoolExecutor
import subprocess

def evaluation():
    examples = {
        "128": Shape([128]),
        "256": Shape([256]),
        "512": Shape([512]),
        "128,128": Shape([128,128]),
        "256,256": Shape([256,256]),
        "512,512": Shape([512,512]),
        "Int1": Shape([1,1,2,3,4,5,6,7,5,4,3,2,1]),
        "Int2": Shape([1,1,1,3,5,7,9,11,13,10,8,6,4,2,1]),
        "Int3": Shape([1,1,1,1,5,9,13,17,21,25,20,16,12,8,4]),
        "Int4": Shape([1,1,1,1,1,9,17,25,33,41,49,40,32,24,16,8]),
        "Int5": Shape([1,1,1,1,1,1,17,33,49,65,81,97,80,64,48,32,16]),
        "LPFP1": Shape([1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]),
        "LPFP2": Shape([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]),
        "LPFP3": Shape([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8]),
        "LPFP4": Shape([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,16]),
        "LPFP5": Shape([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,32]),
        "6x32": Shape(32*[6]),
        "10x32": Shape(32*[10]),
        "Mul16": Shape(list(range(1, 17)) + list(reversed(range(1, 16))))
    }

    filenames = []
    for example_name, example_shape in examples.items():
        print(example_name, example_shape)
        # combinatorial design
        filename = "../gen/"+example_name+"_comb.sv"
        generate_compressor(
            target=Versal(),
            shape=example_shape,
            name="comp",
            comb_depth=None,
            accumulate=False,
            accumulator_width=None,
            gates=None,
            constants=[],
            path=filename,
            test=True
        )
        generate_wrapper(shape=example_shape, pipelined=False, gates=False,
                         accumulation=False, filename=filename)
        filenames.append(filename)
        # accumulating design
        filename = "../gen/"+example_name+"_acc.sv"
        generate_compressor(
            target=Versal(),
            shape=example_shape,
            name="comp",
            comb_depth=1,
            accumulate=True,
            accumulator_width=None,
            gates=None,
            constants=[],
            path=filename,
            test=True
        )
        generate_wrapper(shape=example_shape, pipelined=True, gates=False,
                         accumulation=True, filename=filename)
        filenames.append(filename)
        # gate inlined design with accumulation
        filename = "../gen/"+example_name+"_gate.sv"
        generate_compressor(
            target=Versal(),
            shape=example_shape,
            name="comp",
            comb_depth=1,
            accumulate=True,
            accumulator_width=None,
            gates=[["8" for el in range(col)] for col in example_shape],
            constants=[],
            path=filename,
            test=True
        )
        generate_wrapper(shape=example_shape, pipelined=True, gates=True,
                         accumulation=True, filename=filename)
        filenames.append(filename)

    tclfiles = [emit_eval_tcl_script(el) for el in filenames]

    def call_vivado(filename):
        command = f"""cd ../gen/ &&
            ls && 
            source /proj/xbuilds/released/2023.1/2023.1_0508_1/installs/lin64/Vivado/2023.1/settings64.sh && 
            vivado -mode batch -source {filename.split("/")[-1]}"""
        return subprocess.run(command, shell=True, check=True, timeout=3600, 
                              text=True, executable="/bin/bash")

    print("Executing evaluation threads")
    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(call_vivado, tclfiles)
    print("Done executing evaluation threads")

def generate_wrapper(shape, pipelined, gates, accumulation, filename):
    iw = sum(shape)
    ow = compressed_width(shape)

    inputs = ["clk", "in"]
    if gates:
        inputs.append("in_2")

    if accumulation:
        inputs.append("en_neg")
        inputs.append("rst")

    input_str = "\tinput " + ", ".join(inputs) + ",\n"
    output_str = f"\toutput logic [{ow-1}:0] outReg"

    wrapper_str =  (
    "module sandwich(\n" +
    input_str + 
    output_str +
    '\n);\n' + 
    f"""
\t{"logic en_negReg, rstReg;" if accumulation else ""}
\tlogic [{iw-1}:0] inReg{", in_2Reg;" if gates else ";"}
\twire [{ow-1}:0] out;
\t
\talways_ff @ (posedge clk) begin
\t\t{"rstReg <= rst;" if accumulation else ""}
\t\t{"en_negReg <= en_neg;" if accumulation else ""}
\t\tinReg <= {{inReg, in}};
\t\t{"in_2Reg <= {in_2Reg, in_2};" if gates else ""}
\t\toutReg <= out;
\tend
\t
\t(* keep_hierarchy = "yes" *)
\tcomp c(.in(inReg), .clk(clk),{" .in_2(in_2Reg)," if gates else ""
                                }{" .en_neg(en_negReg), .rst(rstReg)," 
                                  if accumulation else ""} .out(out));

endmodule"""
    )
    with open(filename, 'a') as f:
        f.writelines(wrapper_str)

def emit_eval_tcl_script(compressor_path):
    comps = "set comps { " + str(compressor_path.split("/")[-1])  + " }"
    script = comps + """
set PART xcvc1902-vsva2197-2MP-e-S ; # From VCK190 Evaluation Board

foreach comp $comps {
    read_verilog $comp

    # -----------------------------------------------------------------------------
    # Open new file for current module
    set filename_prefix RESULT_
    set filename_suffix ".json"
    set filename $filename_prefix$comp$filename_suffix
    puts $filename
    set outfile [open $filename w]
    puts $outfile "\{"

    set tm 0.7 ; # Minimum possible ime
    set tt 10.0 ; # Time to Test
    set ts 100.0 ; # Successful Time
    set lc 100000 ; # LUT utilization

    # -----------------------------------------------------------------------------
    # Run synthesis
    synth_design -top sandwich -part $PART

    # -----------------------------------------------------------------------------
    # while loop, updating clock 
    while {[expr $ts - $tm] > 0.1} {
        puts "NEW SYNTHESIS RUN WITH FREQ $tt"
        create_clock -name CLK -period $tt [get_port clk]

        # -----------------------------------------------------------------------------
        # Place and route
        opt_design -retarget -propconst -sweep ;
        place_design -directive Explore
        report_utilization -file util_$comp.twrA
        route_design -directive Explore
        report_drc
        report_utilization -hierarchical
        report_timing -setup -hold -max_paths 3 -nworst 3 -input_pins -sort_by group -file $comp.twrA
        report_timing_summary -delay_type min_max -path_type full_clock_expanded -report_unconstrained -check_timing_verbose -max_paths 3 -nworst 3 -significant_digits 3 -input_pins -file $comp.twrA

        # -----------------------------------------------------------------------------
        # Find maximum data path delay and slack
        set f [open $comp.twrA r]
        set file_data [read $f]
        close $f
        if {[regexp { +Data Path Delay: +(\d+\.\d+)} $file_data -> value]} {
            set tr $value
        } {
            error "DATA PATH DELAY NOT FOUND"
        }

        # -----------------------------------------------------------------------------
        # Find LUT and Slice utilization 
        set f [open util_$comp.twrA r]
        set file_data [read $f]
        close $f
        if {[regexp {CLB LUTs +\| +(\d+)} $file_data -> value]} {
            set lc $value
        } {
            error "LUT UTILIZATION NOT FOUND"
        }

        if {[regexp {SLICE +\| +(\d+)} $file_data -> value]} {
            set sc $value
        } {
            error "SLICE UTILIZATION NOT FOUND"
        }

        # -----------------------------------------------------------------------------
        # Check if timing was met
        if { $tt < $tr } {
            puts {Timing $tr was NOT met!}
            set tm $tt
            if { $tr < $ts } {
                set ts $tr
            } 
        } else {
            set ts $tr
        }
        set tt [expr { ($ts + $tm)/2}]
    }

    puts -nonewline $outfile "\\"Delay\\": $ts,"
    puts -nonewline $outfile "\\"Slice\\": $sc,"
    puts -nonewline $outfile "\\"LUTS\\": $lc" ;

    puts $outfile "\}"
    close $outfile
    remove_files {$comp}
}
q
"""
    tclpath = compressor_path.replace(".sv", ".tcl")
    with open(tclpath, "w") as f:
        f.writelines(script)
    return tclpath

if __name__=="__main__":
    evaluation()
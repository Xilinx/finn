# Python Compressor Generator
This tool can generate compressor trees for 7-Series, UltraScale(+) and Versal for arbitrary input shapes.

# Getting started
1. Clone this repository.
2. _No_ further dependencies needed!

## Usage
Generate a compressor of shape `(12,12,12)` called `comp` and save it under `/gen/comp12_12_12.sv`:

```python3 src/main.py -s 12,12,12 -n comp -o gen/comp12_12_12.sv```

See `python3 src/main.py -h` for details.

## Features
### Custom Input Shape
The tool can generate compressors for any input shape. A shape is passed as a comma-separated list. Each digit indicates a column's height. *LSB* is *left*, *MSB* is *right*.

### Accumulation
By passing `-a`, the tool generates an accumulator instead of just an adder. The accumulators width can be specified by `-w`.
### Gate Absorption
If desired, every input to the compressor can be preceded by a two-input gate. These gates can be integrated into the first compression stage. Each gate is specified as a HEX digit. The encoding is the same is Vivado's LUT2 primitive: 
| Secondary Input | Primary Input | Output
|-----------------|---------------|----------------
|0	              |0	          |(DIGIT << 0) & 1
|0	              |1	          |(DIGIT << 1) & 1
|1	              |0	          |(DIGIT << 2) & 1
|1	              |1	          |(DIGIT << 3) & 1

For example, `8` maps to an AND gate and `6` maps to an XOR gate.

In CLI, gates can be specified as a flat string like `-g 883ABC`. The *LSB* is *left* and *MSB* is *right*. The leftmost specified gate corresponds to the LSB input in the generated compressor input vector.

### Target
Generate compressors for either Versal, 7-Series or UltraScale fabrics using `-t \{Versal,7-Series,UltraScale\}̀ .

### Automated Testing
The tool can automatically generate a SystemVerilog testbench to fuzzy-test the generated compressors by passing `--test`. For testing, the `xvlog`, `xelab` and `xsim` commands have to be available.

### Custom Pipeline Depth
Specify the maximum combinational delay for the compressor using `-p MAX_DEPTH`. Note that the final adder, which has at least one single routing delay, cannot be pipelined. 

### Constant Input
Aside to the regular, variable compressor inputs, the tool also supports an additional constant input. It can be specified as a binary number by `-c NUMBER`.

# Implementation Details - How the Code is Structured
The compressor is internally represented as a graph. Its nodes are defined in `src/graph/nodes.py`. 
Compressor construction is done in several passes:
1. Create a graph with all scheduled counters and a final adder (in `src/passes/compressor_constructor.py`).
    1. (Optional) Generate a gate absorption stage.
    2. Generate regular compression stages until the compression goal is reached.
    3. Insert pipeline registers between compressor stages.
    4. Build either a final adder or an accumulator as the final stage.
2. Annotate LUT6CY instances with placement constraints so that the LUT Cascade will be utilized (in `src/passes/lut_placer.py`).
3. Replace inexpressible connections: Place wires between connected instantiated modules (in `src/passes/wire_inserter.py`). 
4. Annotate input and output signals in the compressor (in `src/passes/io_annotator.py`).
5. Emit generated SystemVerilog source (in `src/passes/emitter.py`)

## Extending the Tool
### Adding new Counters
Counters without gate absorption are defined in `graph/counters/counter_candidates.py`. 
Counters with gate absorption are defined in `graph/counters/absorption_counter_candidates.py`. 

### Adding new Passes
Before adding new passes over the compressor graph, check out if the simple iterator defined in `node_iterator.py` can be inherited to save boilerplate code.

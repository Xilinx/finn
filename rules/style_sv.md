---
paths:
  - "**/*.{sv,svh}"
---
# SystemVerilog Style Guidelines

General rules (file headers, comments, deviations, abbreviations) are in `style_common.md`.

Every module body is enclosed by `` `default_nettype none`` (immediately after the port-list closing `);`) and `` `default_nettype wire`` (immediately before `endmodule`). This catches implicit net declarations inside the module without affecting port declarations or inter-module scope. Files with multiple modules repeat the pair for each module.

## Naming

Signals that represent registers are named in capital camel case. They are always initialized. The initialization value must be identical to the value assigned in the reset branch of the corresponding always_ff block. If the reset value is semantically irrelevant, `'x` is used in both places, granting the synthesis tool full freedom in control set assignment.

Signals that represent combinatorial compute are named in lower snake case. Their pure declarations have no initializer. For simple combinatorial computations by a single expression, continuous assigns are preferred over always_comb blocks. If all inputs to such a computation are available at the point of declaration, a uwire declaration fused with an immediate continuous assignment is used. The declaration as a uwire net is preferred over a logic variable if allowed by the use of the signal, i.e. when it is not the target of a procedural assignment.

Generic parameters and localparams are named in all-capital snake case. Don't use the parameter keyword for typed generics. Use unsigned types whenever semantically sensible. Width, precision, count, offset, and version generics are typically typed as int unsigned. Only localparams that are referenced by port declarations belong in the parameter list; all others are declared in the module body.

Type definitions use a `_t` suffix. Enum types may use `_e` instead. Types should be declared in the smallest meaningful scope, i.e. within a module if no other module relies on them.

Signedness follows semantics: quantities that can represent negative numeric values are declared signed consistently at ports and typedefs, while control data and widths stay unsigned.

## Ports

Modules use ANSI port declarations. Ports are grouped logically, typically starting with the control set (clock and reset), and keeping related signals together regardless of direction (e.g. all signals associated with a stream).

Non-AXI modules use a simplified port naming scheme: `idat`/`ivld`/`irdy` for input streams and `odat`/`ovld`/`ordy` for output streams, where `irdy` is an output (this module can accept input) and `ordy` is an input (downstream can accept output). The clock input is named `clk`, the reset `rst`.

Modules designated for instantiation by the Vivado IP Integrator use standard AXI-Stream port naming: `s_axis_<name>_tdata`, `s_axis_<name>_tvalid`, `s_axis_<name>_tready`, etc. for slave interfaces, and `m_axis_<name>_` prefixes for master interfaces. These modules use `aclk` and `aresetn` as clock and reset.

## Resets

Resets are always synchronous. Non-AXI modules use active-high `rst`. AXI IP Integrator modules use active-low `aresetn`.

## Formatting

Tabs are used for indentation, one tab per indentation level. Tab-indented assign keywords are separated from their assignment expressions by another tab. Assign keywords that are the single-line body of a block statement such as an if-statement or a for-loop are followed by two spaces instead. Tabs may not be used for visual alignment if such alignment would depend on a specific tab size.

Separate parameter or variable names from their type by at least two spaces in a declaration. If equivalent elements in declarations on subsequent lines line up closely already, align them completely. In these cases, also align the elements of index range specifications. Right-align bounds if padding is needed.

The ternary operator is preferred over if-else constructs. The `?` follows the condition without a preceding space (`cond? a : b`). A space before `?` is only used for vertical alignment in multi-decision expressions where several ternaries share a column. Non-trivial conditions (compound expressions with operators) are parenthesized: `(a && b)? c : d`.

If-else branches and loop bodies do not enclose single-line statements that are not particularly long in begin-end blocks.

## Numeric Literals

Numeric literals avoid size specifications unless required by context. Literals of small values whose purpose is not a bit pattern should be decimal. A decimal base is not explicitly identified. A zero vector of a known size is produced by a sized literal rather than by a repetition of a single-bit zero. In concatenations/aggregations, retain explicit element widths when needed to avoid ambiguity.

Examples:
- `3` — not `32'd3` (small value, not a bit pattern)
- `8'd0` — not `{8{1'b0}}` (zero vector of known size via sized literal)
- `4'b1010` — bit pattern, explicit size and base appropriate

## Procedural Blocks

Use `always_ff` for sequential logic and `always_comb` for combinatorial logic. Plain `always` blocks are banned with one exception: clock generation in testbenches (e.g. `always #5ns clk = !clk;`).

## Generate Blocks

Generate blocks of any kind should be labeled if they extend over more than a few lines.

## Assertions

Assertions are used for simulation-time checks. Violations should typically be reported as `$error`.

Static parameter validation that must also be caught by synthesis tools uses `$error`/`$finish` in `initial` blocks rather than `assert` since synthesis tools may ignore assert statements.

## Code Organization

A module body is organized into sections separated by `//===` banner comments. Each section implements a self-contained piece of the dataflow, e.g. a pipeline stage.

The code within each section is organized as follows:
(a) Declaration of all signals representing outputs computed by the section for the consumption by others.
(b) Derivation of combinatorial refinements of processed inputs if needed. Single-line uwire declarations with a fused continuous assignment are preferred.
(c) Description of the sequential logic updating the state of the section.
(d) If needed, a refining combinatorial derivation of the outputs of the section.
If there are signals that only facilitate the internal communication between these steps, an `if(1)` block is used to scope them locally. This construction is preferred over unmotivated begin-end blocks, which are not supported by Vivado. Signal declarations are not batched at the beginning of this block but happen as late as possible. Between sections, give a preference to input rather than output refinements.

## Beware of Self-Determined Contexts as in Range Specifiers

Arithmetic inside a packed range specifier is evaluated in a self-determined context (IEEE 1800-2017 §11.6). A sum of boolean subexpressions like `(A > 0) + (B > 0)` is computed in 1-bit arithmetic, silently overflowing: `1'b1 + 1'b1 = 1'b0`. Always hoist such computations into a `localparam` or use `$clog2` to obtain the correct integer-width result.

```systemverilog
// WRONG: 1-bit overflow, produces [0:0] instead of [2:0]
logic [(A > 0)+(B > 0):0]  Phase;

// OK: localparam provides int assignment context
localparam int unsigned  NUM_PHASES = 1 + (A > 0) + (B > 0);
logic [$clog2(NUM_PHASES)-1:0]  Phase;
```

## Operator Ordering: MUX before ADD

When a conditional selects between advancing a value or keeping it, prefer MUX-then-ADD over ADD-then-MUX. That is, mux the addend and feed a single adder rather than computing both sums and muxing the results. An input MUX can typically be merged into the same LUT level that implements the addition. A terminal MUX always costs an extra LUT level.

Preferred: `result = base + (sel? inc : 0)`
Avoid: `result = sel? (base + inc) : base`

The same principle applies to any associative operator (subtraction, OR-combine, etc.) where one branch is the identity element.

## Design Patterns

### Free-Running Pipeline

A free-running pipeline implements a continuously progressing pipelined dataflow computation without means to stall the pipeline due to backpressure or by an enable signal. A data item is typically accompanied by a valid indication flowing through the pipeline in sync with it. The input is fed by the signal pair `idat` carrying the data and `ivld` indicating a valid input. The respective outputs are `odat` and `ovld`.

Unlike a free-running HLS kernel (`ap_ctrl_none`, see `style_hls.md`), which merely lacks start/stop control but still honours stream backpressure, this pipeline cannot stall at all.

Each pipeline stage is a section following the general code organization described above.

### Pipelined Binary Reduction Tree

A generically-sized binary reduction tree reduces `N` inputs to a single output using `N-1` pipelined binary operator nodes. The tree is represented as a flat array of `2*N-1` edges using heap-style indexing: the root is at index 0, and the children of node `i` are at `2*i+1` and `2*i+2`. Leaves occupy indices `N-1` through `2*N-2`.

The structure is scoped within an `if(1)` block and organized in three sections:

**(1) Tree array and leaf feed.** Declare the tree as `uwire edge_t tree[2*N-1]`. Feed each leaf from the input:

```
for(genvar  i = 0; i < N; i++) begin : genLeaves
    assign  tree[N-1+i] = in[i];
end : genLeaves
```

**(2) Edge delay balancing.** When `N` is not a power of two, the tree has an incomplete leaf level. Leaves at shallower levels have shorter paths to the root and require delay compensation. This is computed by an elaboration-time function:
- Mark each leaf that is not on the deepest level for one cycle of delay.
- Propagate shared delays upward: if both children of a node carry a delay, remove them and assign the delay to the parent instead.

```
typedef bit edge_delays_t[2*N-1];
function edge_delays_t INIT_EDGE_DELAYS();
    localparam int unsigned  LEVELS = 1+$clog2(N);
    automatic edge_delays_t  d = '{ default: 0 };
    for(int unsigned  i = N-1; i < 2*N-1; i++) begin
        if($clog2(i+2) == LEVELS)  break;
        d[i] = 1;
    end
    for(int unsigned  i = N-2; i > 0; i--) begin
        if(d[2*i+1]) begin
            d[2*i+1] = 0;
            d[2*i+2] = 0;
            d[i] = 1;
        end
    end
    return  d;
endfunction
localparam edge_delays_t  EDGE_DELAYS = INIT_EDGE_DELAYS();
```

**(3) Inner nodes.** Generate `N-1` nodes, each computing `tree[i] = f(tree[2*i+1], tree[2*i+2])` for an abstract combinatorial function `f()`. Where `EDGE_DELAYS` indicates a balancing delay, a pipeline register is inserted on the corresponding input:

```
for(genvar  i = 0; i < N-1; i++) begin : genNodes
    uwire edge_t  a;
    if(EDGE_DELAYS[2*i+2]) begin : genDelay
        edge_t  Del = 'x;
        always_ff @(posedge clk)  Del <= tree[2*i+2];
        assign  a = Del;
    end : genDelay
    else begin : genDirect
        assign  a = tree[2*i+2];
    end : genDirect
    assign  tree[i] = f(tree[2*i+1], a);
end : genNodes
```

Since delays are assigned to a contiguous run of leaves and a pair of delays is always lifted to the parent, only right edges (`2*i+2`) retain delays after propagation. The node generation therefore only checks `EDGE_DELAYS[2*i+2]`.

The root result is taken from `tree[0]`. The total tree latency is `$clog2(N)` cycles, assuming `f()` is registered at its output.

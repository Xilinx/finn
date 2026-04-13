# FINN Style Guide

## Purpose and Scope

This guide covers coding standards for all languages used in the FINN project:
- **Python** (FINN compiler, transformations, analysis)
- **C++/HLS** (finn-hlslib operators)
- **SystemVerilog** (finn-rtllib components)

Following these standards ensures consistency across the codebase and makes code easier to read, review, and maintain.

---

## Python Style

### General Principles

- Follow **PEP 8** ([https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/))
- Follow **Google Python Style Guide** for docstrings ([https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html))
- Use **pre-commit hooks** (already configured in `.pre-commit-config.yaml`)

### Naming Conventions

#### Classes
**Pattern**: PascalCase with descriptive names

**Examples**:
```python
StreamingConcat      # HWCustomOp subclass
StreamingFIFO        # HWCustomOp subclass
Pool_hls             # HLS implementation (note _hls suffix)
FMPadding_rtl        # RTL implementation (note _rtl suffix)
MakeZYNQProject      # Transformation class
InsertDWC            # Transformation class
AnnotateResources    # Analysis class
```

**Convention**:
- `*_hls` suffix indicates HLS backend implementation
- `*_rtl` suffix indicates RTL backend implementation
- Transformation classes use imperative names (verbs)

#### Functions and Methods
**Pattern**: snake_case with prefix-based grouping

**Common prefixes**:
- `get_*`: Getters - `get_nodeattr_types()`, `get_input_datatype()`, `get_folded_input_shape()`, `get_instream_width()`
- `set_*`: Setters - `set_nodeattr()`, `set_tensor_datatype()`, `set_tensor_shape()`
- `infer_*`: Inference methods - `infer_node_datatype()`, `infer_shapes()`
- `execute_*`: Execution methods - `execute_node()`, `execute_onnx()`
- `_private_method`: Underscore prefix for private/internal methods

**Examples**:
```python
def get_nodeattr_types(self):
    """Define node attribute schema."""
    ...

def get_folded_input_shape(self, ind=0):
    """Return folded input shape with explicit PE dimension."""
    ...

def _suitable_node(node):
    """Internal helper to check if node is suitable for transformation."""
    ...
```

#### Variables
**Pattern**: snake_case with domain-specific abbreviations

**Standard abbreviations** (use consistently):
- `idt` - Input datatype
- `odt` - Output datatype
- `wdt` - Weight datatype
- `pe` - Processing Elements
- `simd` - SIMD parallelism factor
- `node` - ONNX node
- `graph` - ONNX graph
- `model` - ModelWrapper instance
- `ind` - Index
- `ishape` - Input shape
- `oshape` - Output shape
- `vecs` - Vectors
- `cf` - Channel fold
- `fxn` - Function

**Example**:
```python
idt = self.get_input_datatype()
odt = self.get_output_datatype()
pe = self.get_nodeattr("PE")
simd = self.get_nodeattr("SIMD")
fold = num_channels // pe
```

### Import Organization

**Use isort** (configured in pre-commit):

1. Standard library imports
2. Third-party imports (numpy, onnx, qonnx, brevitas)
3. Local finn imports

**Example**:
```python
import math
import warnings
from copy import deepcopy

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.base import Transformation
```

### Docstring Style

**Required**: All public classes, methods, and functions

**Format**: Google Python Style Guide format

**Full docstring example**:
```python
def execute_onnx(model, input_dict, return_full_exec_context=False):
    """Executes given ONNX ModelWrapper with given named inputs.

    Args:
        model: ONNX ModelWrapper instance to execute
        input_dict: Dictionary mapping input names to numpy arrays
        return_full_exec_context: If True, return all intermediate tensors
            in addition to final outputs

    Returns:
        Dictionary of output tensors if return_full_exec_context is False,
        otherwise dictionary of all tensors including intermediates

    Raises:
        ValueError: If input_dict is missing required inputs
    """
```

**One-liner for simple methods**:
```python
def get_n_inputs(self):
    """Returns number of input streams."""
    return len(self.get_nodeattr("ChannelsPerStream"))

def calc_tmem(self):
    """Calculates and returns TMEM (NumChannels / PE)."""
    return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
```

### Type Hints

**Adoption encouraged** as part of ongoing refactoring:
- Add type hints to **new code** and functions being modified
- Focus on public API functions, transformations, and CustomOp methods
- Use for ModelWrapper parameters and complex return types
- Improves IDE support, documentation clarity, and early error detection

**Example**:
```python
from typing import Dict, Tuple, Optional
from qonnx.core.modelwrapper import ModelWrapper

def get_driver_shapes(model: ModelWrapper) -> Dict[str, Tuple]:
    """Extract driver tensor shapes from model."""
    ...

def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
    """Apply transformation to model.

    Returns:
        Tuple of (modified model, transformation_applied_flag)
    """
    ...
```

**Guidelines**:
- Use `typing` module for complex types (Dict, List, Tuple, Optional, Union)
- Don't add type hints just for the sake of it; focus on clarity
- Balance between helpful type information and readability

### Error Handling

- Use **assertions with descriptive messages** for invariant checks
- Use **exceptions with clear error messages** for runtime errors
- Include context (node names, values) in error messages

**Good examples**:
```python
assert len(consumers) == 1, (
    f"{n.name}: HW node with fan-out higher than 1 cannot be stitched"
)

assert num_channels % pe == 0, (
    f"PE ({pe}) must divide NumChannels ({num_channels})"
)

raise Exception(f"Unrecognized mode '{mode}' for AnnotateResources")

raise ValueError(f"Expected positive SIMD value, got {simd}")
```

---

## HLS/C++ Style (finn-hlslib)

### General Principles

- Follow **Xilinx Vitis HLS Coding Guidelines** ([UG1399](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls))
- Use **Doxygen-style documentation**
- Prioritize **synthesizability** over software engineering patterns
- Avoid dynamic memory allocation, recursion, and unbounded loops

### Naming Conventions

#### Template Parameters
**Pattern**: ALL_CAPS or CamelCase with semantic meaning

**Examples**:
```cpp
template<
    unsigned int ConvKernelDim,    // Clear semantic names
    unsigned int IFMChannels,      // IFM = Input Feature Maps
    unsigned int OFMChannels,      // OFM = Output Feature Maps
    unsigned int SIMD,             // Hardware parallelism
    unsigned int PE,               // Processing Elements
    typename TSrcI = Identity,     // Type params: T prefix
    typename TDstI,
    typename TWeightI,
    typename TW,                   // Weights
    typename TA                    // Activation
>
```

**Standard abbreviations**:
- `IFM` - Input Feature Map
- `OFM` - Output Feature Map
- `SIMD` - SIMD parallelism
- `PE` - Processing Elements
- `T*` - Type parameters (prefix with T)

#### Functions
**Pattern**: CamelCase

**Examples**:
```cpp
StreamLimiter                      // Describes operation
StreamLimiter_Batch                // Variants use underscore
Mem2Stream                         // DMA operations
Stream2Mem
Pool_batch
Matrix_Vector_Activate_Batch
Thresholding_Stream_Batch
```

#### Types
**Pattern**: Standard HLS types

**Examples**:
```cpp
ap_int<N>              // Signed fixed-width integer
ap_uint<N>             // Unsigned fixed-width integer
hls::stream<T>         // Streaming interface
hls::vector<T, N>      // SIMD vector operations
```

### Documentation

**Required**: Doxygen comments for all public functions and templates

**Example**:
```cpp
/**
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector,
 * and generating output based on the comparison results
 *
 * \tparam ImgDim         Total spatial size of input feature map
 * \tparam NumChannels    Number of channels in input feature map
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the parameters
 * \tparam TO             DataType of the output stream - safely deducible from the parameters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the parameters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDim, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        TA const &activation,
                        int const reps)
{
    constexpr unsigned NF = NumChannels / PE;
    // Implementation with pipelined loop
    for (unsigned i = 0; i < reps * ImgDim * NF; i++) {
#pragma HLS pipeline style=flp II=1
        // Process activation thresholding
    }
}
```

### Pragma Usage

**Common HLS pragmas**:
```cpp
#pragma HLS pipeline style=flp II=1        // Pipeline with flushable pipeline, II=1
#pragma HLS INLINE                         // Inline function
#pragma HLS dataflow disable_start_propagation
#pragma HLS aggregate variable=in0_V compact=bit
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
```

**Guidelines**:
- Use `#pragma HLS pipeline` for loops to improve throughput
- Use `#pragma HLS INLINE` judiciously (can increase area)
- Use `#pragma HLS dataflow` for task-level pipelining
- Use `#pragma HLS ARRAY_PARTITION` to enable parallel access

### File Organization

**Header guards required**:
```cpp
#ifndef STREAM_TOOLS_HPP
#define STREAM_TOOLS_HPP

// Includes
#include <hls_stream.h>
#include <ap_int.h>

// Local includes
#include "weights.hpp"
#include "streamtools.h"

// Code

#endif // STREAM_TOOLS_HPP
```

**Include order**:
1. System HLS headers (`<hls_stream.h>`, `<ap_int.h>`)
2. Standard library headers if needed
3. Local finn-hlslib headers

---

## SystemVerilog Style (finn-rtllib)

### General Principles

- Follow **IEEE 1800-2017 SystemVerilog** standard
- Use **SystemVerilog features** (logic, typedef, struct, enum) over Verilog-2001
- Prioritize **synthesizability** and **timing closure**

### 1. Indentation

- **Structural indentation** uses **tabulators (tabs)**
- **Alignment within lines** uses **spaces**
- Tab size is implementation-defined but irrelevant (tabs only for indentation)

### 2. Time

- No explicit settings for time unit or scale
- Timed delays only used for simulation (not synthesis)
- Each delay specified with explicit unit

**Example**:
```systemverilog
logic clk = 0;
always #5ns clk = !clk;
```

### 3. Parameters

- Names use **ALL_CAPS snake_case**
- Parameters are **generally typed** (exception: auto-sizing string literals)
- Module parameters only given **natural defaults**; avoid arbitrary pre-imposed choices

**Example**:
```systemverilog
module memstream #(
    int unsigned  DEPTH,
    int unsigned  WIDTH,
    parameter     INIT_FILE = "",
    parameter     RAM_STYLE = "auto"
)(
    // Ports
);
    localparam int unsigned  WRAP_INC = 2**$clog2(DEPTH) - DEPTH + 1;
    ...
endmodule : memstream
```

### 4. State (Registers)

- Names use **initial-capital camel case** (e.g., `MyRegister`, `PipelineStage`)
- Registers declared as **variables** (keyword `reg` never used; `var` keyword discouraged)
- Registers **generally initialized**, making don't-care explicit by assigning `'x`
- Registers **reset explicitly** to the same value they are initialized to
- **Exception**: State in primitives incapable of reset (SRL, BRAM contents) - must be justified by comment
- **Names introduced as module ports are never used for storing state**

**Example**:
```systemverilog
typedef struct packed {
    logic [7:0] op;
    logic [31:0] val;
} pipe_t;

pipe_t P = '{ op: NOP, default: 'x };
logic  Reval = 0;

always_ff @(posedge clk) begin
    if(rst) begin
        P     <= '{ op: NOP, default: 'x };
        Reval <= 0;
    end
    else begin
        P     <= p;
        Reval <= (p.op ==? RB) && cs;
    end
end
```

### 5. Combinatorial Logic

- Names use **lower-case snake_case** (e.g., `mag_eq`, `cmp`, `ready_signal`)
- Combinational signals declared as **unresolved `uwire`** wherever possible
- Immediate continuous assignment at declaration preferred
- Combinational signals computed in `always` blocks declared as **variables**
- Combinational variables are **not initialized**

**Example**:
```systemverilog
logic cmp;  // Variable for always block
uwire mag_eq = Thresh[K-2:0] == P.val[K-2:0];
uwire mag_le = Thresh[K-2:0] <= P.val[K-2:0];

always_comb begin
    unique case({Thresh[K-1], P.val[K-1]})
        2'b00:   cmp = mag_le;
        2'b01:   cmp = 0;
        2'b10:   cmp = 1;
        2'b11:   cmp = !mag_le || mag_eq;
        default: cmp = 'x;
    endcase
end
```

### 6. Block Labels

- Closings of block entities (`endmodule`, `endfunction`, `endtask`) are **labeled**
- Non-trivial `begin ... end` blocks **should be labeled**
- Generate blocks use **camel case** starting with prefix **"gen"**
- Scoping blocks use **camel case** starting with prefix **"blk"**

**Example**:
```systemverilog
module thresholding #(...)(...);
    ...
    if(1) begin : blkFeed
        ...
    end : blkFeed

    for(genvar stage = 0; stage < N; stage++) begin : genStages
        for(genvar pe = 0; pe < PE; pe++) begin : genPE
            ...
        end : genPE
    end : genStages
endmodule : thresholding
```

### 7. Scoping

- Identifiers introduced in **smallest scope** of their use
- Introduction of scoping blocks (`if(1) begin ... end`) encouraged for names of sole immediate local relevance

### 8. Always Blocks

- Always blocks **must be** `always_ff` or `always_comb`
- **Exception**: Raw `always` blocks only for special needs (e.g., clock generation in testbenches)

**Example**:
```systemverilog
always_ff @(posedge clk) begin
    // Sequential logic
end

always_comb begin
    // Combinatorial logic
end
```

### 9. Case Statements

- Case statements **must be explicitly** implemented by most restrictive applicable alternative
- **Single-choice** case statements identified by `unique` keyword
- **Wildcard matches** implemented by `casez` or `casex`

**Example**:
```systemverilog
always_comb begin
    unique case(state)
        IDLE:   next_state = ACTIVE;
        ACTIVE: next_state = DONE;
        DONE:   next_state = IDLE;
        default: next_state = 'x;
    endcase
end
```

### 10. Data Types

- **Hardware signals** use **4-valued logic types** (`logic`, not `bit`)
- **Parameters without hardware association** use **2-valued data types** (`int`, `bit`)
- Use **role-specific typedefs**, **structs**, and **enums** (encouraged)
- When domain knowledge available, use **most restrictive basic type** (e.g., `byte` over `int`)

**Example**:
```systemverilog
typedef logic [31:0]  fp32;
typedef fp32 [SIMD-1:0]  vfp32;

typedef struct packed {
    fp32   dat;
    logic  vld;
} edge_t;

typedef enum logic [1:0] {
    IDLE   = 2'b00,
    ACTIVE = 2'b01,
    DONE   = 2'b10
} state_t;
```

### Port Naming Conventions

**Clock/Reset**:
- `clk` or `aclk` - Clock signal
- `rst` or `aresetn` - Reset (active-low uses 'n' suffix)

**Handshake Signals**:
- `*vld` or `*valid` - Data validity
- `*rdy` or `*ready` - Backpressure/ready
- `*dat` or `*data` - Data payload

**AXI Stream Example**:
```systemverilog
// Input stream
input  logic [SIMD-1:0][31:0]  xdat,
input  logic                   xvld,
output logic                   xrdy,

// Output stream
output logic [SIMD-1:0][31:0]  ydat,
output logic                   yvld,
input  logic                   yrdy
```

**AXI-Lite Interface** (standard names):
- Write Address: `awready`, `awvalid`, `awprot`, `awaddr`
- Write Data: `wready`, `wvalid`, `wstrb`, `wdata`
- Write Response: `bready`, `bvalid`, `bresp`
- Read Address: `arready`, `arvalid`, `arprot`, `araddr`
- Read Data: `rready`, `rvalid`, `rresp`, `rdata`

### Xilinx-Specific Attributes

```systemverilog
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input ap_clk;
```

### Design Patterns

For detailed design patterns (counters, binary reduction trees, elastic buffers), refer to the full AMD FinnLib SystemVerilog Style Guide document.

**Key pattern**: Use sign bit of counters for completion signaling:
```systemverilog
logic signed [$clog2(N-1):0] Cnt = N-2;  // N-2, ..., 1, 0, -1
uwire complete = Cnt[$left(Cnt)];  // Sign bit signals completion
```

---

## Common Patterns Across All Languages

### File Headers

All source files should include the copyright header with SPDX identifier:

**Python**:
```python
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

**HLS/C++**:
```cpp
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
```

**SystemVerilog**:
```systemverilog
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
```

### Naming Domain-Specific Concepts

Use these abbreviations **consistently across Python, HLS, and SystemVerilog**:

- **PE** - Processing Elements
- **SIMD** - Single Instruction Multiple Data
- **MVAU** - Matrix Vector Activation Unit
- **VVAU** - Vector Vector Activation Unit
- **DWC** - Data Width Converter
- **SWG** - Sliding Window Generator
- **IFM** - Input Feature Map
- **OFM** - Output Feature Map
- **TMEM** - Threshold Memory (NumChannels / PE)

### Comments

- Use comments to explain **why**, not **what**
- Complex algorithms should have block comments explaining approach
- Avoid obvious comments

**Bad**:
```python
simd = 8  # Set SIMD to 8
```

**Good**:
```python
simd = 8  # Limit parallelism to match BRAM port constraints
```

---

## References

### Python
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### HLS
- [Xilinx Vitis HLS User Guide (UG1399)](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls)

### SystemVerilog
- [IEEE 1800-2017 SystemVerilog Standard](https://ieeexplore.ieee.org/document/8299595)

---

## Enforcement

- **Python**: Pre-commit hooks enforce black, flake8, isort (see `.pre-commit-config.yaml`)
- **HLS**: Manual code review
- **SystemVerilog**: Manual code review (consider future linter integration)

When in doubt, follow existing patterns in the codebase and consult with maintainers during PR review.

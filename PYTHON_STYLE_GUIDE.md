# FINN Python Style Guide

## Purpose and Scope

This guide covers coding standards for Python code in the FINN project:
- **Python conventions** (FINN compiler, transformations, analysis)
- **FINN-specific architectural patterns** (CustomOps, transformations, node attributes)

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

### File Headers

All Python source files should include the copyright header with SPDX identifier:

```python
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

---

## FINN-Specific Patterns and Conventions

This section covers architectural patterns and design principles specific to the FINN compiler.

### Node Attributes

Node attributes persist layer configuration in the ONNX graph between compiler steps.

**DO add node attributes for**:
- Information that **cannot be easily computed** from other attributes or graph structure
- Layer-specific configuration (PE, SIMD, NumChannels, DataTypes)
- Parameters that **vary between layer instances**

**DON'T add node attributes for**:
- Values **computable from other attributes** (e.g., `TMEM = NumChannels / PE`)
- **Global/model-wide settings** (FPGA part, clock period) → use `DataflowBuildConfig` instead
- Temporary calculation results
- Information already in the ONNX graph (tensor shapes, initializers)

**Example** (from CustomOp `get_nodeattr_types`):
```python
# ✓ GOOD - required configuration that varies per layer
my_attrs = {
    "Channels": ("i", True, 0),           # Cannot be inferred
    "PE": ("i", True, 1),                 # Parallelism factor
    "InputDataType": ("s", True, ""),     # FINN DataType string
    "preferred_impl_style": ("s", False, ""),  # Optional HLS vs RTL preference
}

# ✗ BAD - can compute as Channels / PE
"TMEM": ("i", False, 0)

# ✗ BAD - global setting, belongs in DataflowBuildConfig
"fpga_part": ("s", False, "")

# ✗ BAD - already in ONNX tensor metadata
"weight_initializer": ("ints", False, [])
```

**See**: GitHub discussion [#1026](https://github.com/Xilinx/finn/discussions/1026) for detailed CustomOp system explanation

### CustomOp Three-Tier Hierarchy

FINN uses a **three-layer class hierarchy** for hardware operators:

**1. Base Layer** (`src/finn/custom_op/fpgadataflow/<layer>.py`)
- Backend-agnostic functionality
- Defines layer semantics and node attributes via `get_nodeattr_types()`
- Implements shape calculations, stream width calculations
- Provides Python golden reference execution (`execute_node()`)
- **Naming**: PascalCase (e.g., `Pool`, `MatrixVectorActivation`, `FMPadding`)

**2. HLS Variant** (`src/finn/custom_op/fpgadataflow/hls/<layer>_hls.py`)
- Inherits from base layer + `HLSBackend`
- Implements HLS-specific code generation methods
- Uses finn-hlslib C++ templates
- **Naming**: Base name + `_hls` suffix (e.g., `Pool_hls`, `MVAU_hls`)

**3. RTL Variant** (`src/finn/custom_op/fpgadataflow/rtl/<layer>_rtl.py`)
- Inherits from base layer + `RTLBackend`
- Implements RTL-specific HDL generation methods
- Uses finn-rtllib SystemVerilog modules
- **Naming**: Base name + `_rtl` suffix (e.g., `FMPadding_rtl`, `MVAU_rtl`)

**Design principle**: Shared logic goes in **base class**, backend-specific code in **HLS/RTL variants**.

**Example structure**:
```python
# Base layer: src/finn/custom_op/fpgadataflow/pool.py
class Pool(HWCustomOp):
    def get_nodeattr_types(self):
        # Define attributes for all backends
        ...

    def get_folded_output_shape(self):
        # Shared shape calculation logic
        ...

# HLS variant: src/finn/custom_op/fpgadataflow/hls/pool_hls.py
class Pool_hls(Pool, HLSBackend):
    def generate_params(self, model, path):
        # HLS-specific parameter generation
        ...

    def docompute(self):
        # HLS compute template call
        return "Pool_batch<...>(...)"

# RTL variant: src/finn/custom_op/fpgadataflow/rtl/pool_rtl.py
class Pool_rtl(Pool, RTLBackend):
    def generate_hdl(self, model, fpgapart, clk):
        # RTL HDL generation
        ...
```

### Transformation Pass Structure

All transformation passes follow a consistent structure:

**Required**:
- Inherit from `Transformation` base class
- Implement `apply(model: ModelWrapper) -> Tuple[ModelWrapper, bool]`
- Return tuple of `(modified_model, model_was_changed)`

**Naming**: Imperative verbs (e.g., `InsertDWC`, `InferShapes`, `AbsorbAddIntoMultiThreshold`)

**Example**:
```python
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.base import Transformation

class MyTransformation(Transformation):
    """Brief description of what this transformation does."""

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        """Apply transformation to model.

        Returns:
            Tuple of (modified model, transformation_applied_flag)
        """
        graph = model.graph
        model_was_changed = False

        for node in graph.node:
            # Check if transformation applies
            if self._should_transform(node):
                # Modify graph
                self._apply_to_node(model, node)
                model_was_changed = True

        return (model, model_was_changed)
```

### Analysis Pass Structure

Analysis passes extract information from the model without modifying it.

**Required**:
- Return a **dictionary** of extracted properties
- Do not modify the model
- Use descriptive dictionary keys

**Naming**: Descriptive nouns (e.g., `exp_cycles_per_layer`, `res_estimation`)

**Example**:
```python
def my_analysis(model: ModelWrapper) -> Dict[str, Any]:
    """Extract properties from model.

    Returns:
        Dictionary mapping property names to values
    """
    results = {}

    for node in model.graph.node:
        # Extract information
        results[node.name + "_property"] = compute_property(node)

    return results
```

### Testing Organization

**Test file naming**:
- `test_<module_or_feature>.py`
- Group related tests in same file

**Test locations**:
- Unit tests for transformations: `tests/transformation/<category>/test_*.py`
- HW layer tests: `tests/fpgadataflow/test_*.py`
- End-to-end flows: `tests/end2end/test_*.py`

**Test function naming**:
- `test_<specific_behavior>()`
- Use descriptive names explaining what is tested

**Pytest markers**:

See `setup.cfg` under `[tool:pytest]` markers section for the complete list of available markers.

**Example**:
```python
import pytest
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC

def test_insert_dwc_basic():
    """Test DWC insertion for basic stream width mismatch."""
    model = build_test_model()
    model = model.transform(InsertDWC())
    # Assertions
    assert check_expected_behavior(model)

@pytest.mark.vivado
def test_pool_hls_synthesis():
    """Test HLS synthesis of Pool layer."""
    # Test requiring Vivado
    ...
```

---

## Domain-Specific Abbreviations

Use these abbreviations **consistently** across FINN:

- **PE** - Processing Elements
- **SIMD** - Single Instruction Multiple Data
- **MVAU** - Matrix Vector Activation Unit
- **VVAU** - Vector Vector Activation Unit
- **DWC** - Data Width Converter
- **SWG** - Sliding Window Generator
- **IFM** - Input Feature Map
- **OFM** - Output Feature Map
- **TMEM** - Threshold Memory (NumChannels / PE)

---

## Comments

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

- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [FINN GitHub Discussion #1026 – CustomOp System](https://github.com/Xilinx/finn/discussions/1026)

---

## Enforcement

- **Pre-commit hooks** enforce black, flake8, isort (see `.pre-commit-config.yaml`)
- **Manual code review** for FINN-specific patterns

When in doubt, follow existing patterns in the codebase and consult with maintainers during PR review.

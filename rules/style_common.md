---
paths:
  - "**/*.{sv,svh,v,cpp,hpp,h,py}"
---
# Common Style Guidelines

These apply to all FINN source code: SystemVerilog, C++/HLS and Python. The
language-specific files in this directory add to them.

The guidelines are strict but not religious. Deviations are possible but must be
rare and always accompanied by an explanatory comment in the sources.

## File Headers

Every source file starts with a copyright notice and an SPDX license identifier.
The license is that of the repository the file lives in: BSD-3-Clause for FINN,
finn-hlslib and finn-rtllib. Keep the identifier of an existing file; do not
change the license of code you merely edit.

SystemVerilog and C++:

```cpp
/****************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************/
```

Python:

```python
# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
```

## Comments

Comments explain **why**, not **what**: design decisions, algorithms, non-obvious
constraints, and hardware considerations such as timing or resource use. Do not
repeat what the code already says. Complex algorithms get a block comment
describing the approach.

```python
simd = 8  # limit parallelism to the BRAM port width
```

rather than

```python
simd = 8  # set SIMD to 8
```

## Domain Abbreviations

Use these consistently, in code and in comments:

| Abbreviation | Meaning |
|---|---|
| PE | processing elements, output-channel parallelism |
| SIMD | input-channel (lane) parallelism |
| NF, SF | neuron fold (`MH/PE`), synapse fold (`MW/SIMD`) |
| CF | channel fold (`C/PE` or `C/SIMD`) |
| TH | tiling height of the tiled MVU |
| IFM, OFM | input, output feature map |
| MVAU, VVAU | matrix-vector, vector-vector activation unit |
| DWC | data width converter |
| SWG | sliding window generator |
| TMEM | threshold memory depth (`NumChannels/PE`) |

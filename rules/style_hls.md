---
paths:
  - "**/*.{cpp,hpp,h}"
---
# HLS Architecture

General rules are in `style_common.md`, C++ formatting and naming in
`style_cpp.md`. Reference: Vitis HLS User Guide (UG1399).

## General Principles

- Synthesizability comes first: no dynamic memory allocation, no recursion, no
  loops without a static bound.
- New kernels are **free-running** (`ap_ctrl_none`) with `hls::stream<>`
  interfaces and `#pragma HLS pipeline II=1 style=flp`. Data flows through
  streams; there is no software-controlled start/stop. Unlike a free-running RTL
  pipeline (see `style_sv.md`), a kernel still honours stream backpressure.
- One invocation of such a kernel handles at most one item per stream; progress
  through a frame is kept in `static` state. (Legacy `_Batch` kernels instead
  loop over whole frames per call.)

## State and Constants

- `static` kernel state carries `#pragma HLS reset variable=…`, so that it
  starts from a defined value after reset.
- Constants that differ per kernel instance but are passed as function
  arguments (e.g. a padding value) get `#pragma HLS function_instantiate
  variable=…`, so that each instance is specialized for its constant.

```cpp
template<unsigned  H, unsigned  W, unsigned  CF, typename  T>
void pad(hls::stream<T> &src, hls::stream<T> &dst, T const &pad_val) {
#pragma HLS function_instantiate variable=pad_val
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(H)>  ho = 0;
#pragma HLS reset variable=ho
	...
}
```

## Pragmas

Pragma keywords are lowercase (see `style_cpp.md`).

- `pipeline II=1 style=flp` on every free-running kernel; `style=frp` where a
  free-running pipeline suffices.
- `dataflow` for task-level pipelining of composed kernels.
- `array_partition` where parallel access to an array is needed.
- `inline` deliberately: inlining enables optimization across the call but can
  increase area.

## Design Patterns

- **Parallelism**: `tensor<T, PE, SIMD>`, inner SIMD lane parallelism, outer PE
  parallelism. Do not use `hls::vector<>` in new code.
- **Framing**: `flit_t<T>` (`utils.hpp`) with `last` and `data` fields marks
  end-of-accumulation boundaries between kernels.
- **Tree reduction**: a fully unrolled binary tree over a heap-indexed array
  (root at 0, children of `i` at `2*i+1` and `2*i+2`), the HLS counterpart of
  the pipelined reduction tree in `style_sv.md`.

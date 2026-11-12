---
paths:
  - "**/finn/**/*.py"
  - "**/tests/**/*.py"
---
# FINN Compiler Patterns

Architectural conventions of the FINN compiler. Python style is in
`style_python.md`. See also `docs/finn/implementation/customop-pattern.rst`,
`specialization-rules.rst` and `docs/finn/reference/`.

## Node Attributes

Node attributes persist layer configuration in the ONNX graph between compiler
steps.

Add a node attribute for:
- information that cannot be computed from other attributes or the graph,
- configuration that varies between layer instances (`PE`, `SIMD`,
  `NumChannels`, datatypes, `preferred_impl_style`).

Do not add one for:
- values computable from other attributes (e.g. `TMEM = NumChannels / PE`),
- global settings such as FPGA part or clock period, which belong in
  `DataflowBuildConfig`,
- temporary results,
- information already in the graph (tensor shapes, initializers).

## Custom Op Hierarchy

Hardware operators use four levels:

1. `HWCustomOp`: infrastructure common to all hardware layers.
2. Backend mixins `HLSBackend` and `RTLBackend`: code generation interfaces
   (shared finn-hlslib templates for HLS, per-layer finn-rtllib wrappers for
   RTL).
3. The base layer, `src/finn/custom_op/fpgadataflow/<layer>.py`, PascalCase
   (`Pool`, `LayerNorm`): backend-agnostic semantics, node attributes
   (`get_nodeattr_types()`), shapes, stream widths and the Python reference
   execution (`execute_node()`).
4. The implementations `hls/<layer>_hls.py` (base + `HLSBackend`) and
   `rtl/<layer>_rtl.py` (base + `RTLBackend`), named with an `_hls` or `_rtl`
   suffix.

Shared logic goes into the base layer; only backend-specific code goes into
the implementations.

## Transformations

- Subclass `Transformation` and implement
  `apply(model) -> (model, model_was_changed)`.
- Name them imperatively: `InsertDWC`, `InferShapes`, `AbsorbAddIntoMultiThreshold`.
- Return `model_was_changed=True` only if the graph changed; the transform loop
  repeats until it is `False`.

## Analysis Passes

- A function taking the model and returning a dictionary of results.
- Never modify the model.
- Name them descriptively: `exp_cycles_per_layer`, `res_estimation`.

## Tests

- Files `test_<module_or_feature>.py`: transformations in
  `tests/transformation/<category>/`, hardware layers in `tests/fpgadataflow/`,
  end-to-end flows in `tests/end2end/`.
- Test names describe the behaviour under test: `test_<behaviour>()`.
- Mark tests with the markers defined in `setup.cfg` (`[tool:pytest]`), notably
  `vivado` and `slow` for tests that need Vivado or take long.
- Create scratch directories with `make_build_dir()` and remove them with
  `robust_rmtree()`.

---
paths:
  - "**/*.py"
---
# Python Style Guidelines

General rules (file headers, comments, deviations, abbreviations) are in
`style_common.md`. FINN compiler architecture (custom ops, transformations,
analysis passes, tests) is in `patterns_finn.md`.

## Formatting and Linting

Code follows PEP 8, enforced by the pre-commit hooks in
`.pre-commit-config.yaml`: black (line length 100), isort and ruff. Run
`pre-commit run --files <changed files>` before committing; the hooks rewrite
formatting and import order themselves.

Imports are grouped by isort: standard library, third party (numpy, onnx,
qonnx, brevitas), then `finn`.

## Naming

- Classes: PascalCase (`StreamingFIFO`, `InsertDWC`). Backend implementations
  carry an `_hls` or `_rtl` suffix (`Pool_hls`, `FMPadding_rtl`).
- Functions, methods, variables: snake_case. Common prefixes group methods:
  `get_*`, `set_*`, `infer_*`, `execute_*`.
- A leading underscore marks module-internal helpers and private methods.
- Use the standard short names consistently: `idt`, `odt`, `wdt` (input,
  output, weight datatype), `pe`, `simd`, `node`, `graph`, `model`
  (`ModelWrapper`), `ind` (stream index), `ishape`, `oshape`.

## Docstrings

- Every public module, class and function has a docstring.
- Docstrings are prose: what the function does, what it returns, and anything a
  caller must know (side effects, failure modes, assumptions). A one-line
  docstring is enough for simple methods.
- Add `Args:` / `Returns:` / `Raises:` sections (Google style) where the
  parameters are not self-explanatory from their names and types, typically for
  public API functions with several parameters.

```python
def get_folded_input_shape(self, ind=0):
    """The input shape with the SIMD dimension split off as the innermost axis."""
```

## Type Hints

Add type hints to new code and to public functions you modify, notably
`ModelWrapper` parameters and non-trivial return types. Do not add them only
for the sake of it.

```python
def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
```

## Errors

- Assertions with a message check internal invariants.
- Exceptions with a clear message report invalid input or configuration.
- Messages name the node and the offending values:

```python
assert num_channels % pe == 0, f"{node.name}: PE ({pe}) must divide NumChannels ({num_channels})"
raise ValueError(f"{node.name}: unsupported mem_mode '{mem_mode}'")
```

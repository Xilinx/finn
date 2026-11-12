# FINN Coding Rules

Coding rules for FINN and its HDL libraries (finn-hlslib, finn-rtllib). They
replace the former `HDL_STYLE_GUIDE.md` and `PYTHON_STYLE_GUIDE.md`.

| File | Scope | Content |
|---|---|---|
| [`style_common.md`](style_common.md) | all sources | file headers and licenses, comments, deviations, domain abbreviations |
| [`style_sv.md`](style_sv.md) | `*.sv`, `*.svh` | SystemVerilog naming, ports, resets, formatting, literals, code organization; free-running pipeline and pipelined reduction tree |
| [`style_cpp.md`](style_cpp.md) | `*.cpp`, `*.hpp`, `*.h` | C++ naming, documentation, formatting, declarations, const, casts, includes, scoping |
| [`style_hls.md`](style_hls.md) | `*.cpp`, `*.hpp`, `*.h` | HLS kernel architecture: free-running streams, kernel state, `tensor<>` parallelism, framing, tree reduction |
| [`style_python.md`](style_python.md) | `*.py` | Python formatting and linting, naming, docstrings, type hints, errors |
| [`patterns_finn.md`](patterns_finn.md) | FINN sources and tests | node attributes, custom op hierarchy, transformations, analysis passes, tests |

Each file declares its scope in a `paths:` frontmatter block. Tools that apply
rules by path, such as Claude Code, can use this directory directly, e.g. by
linking it as `.claude/rules`.

Formatting of Python code is enforced by the pre-commit hooks
(`.pre-commit-config.yaml`); everything else is checked in code review. When in
doubt, follow the patterns of the surrounding code and ask during review.

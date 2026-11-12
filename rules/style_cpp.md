---
paths:
  - "**/*.{cpp,hpp,h}"
---
# C++ Style Guidelines

General rules (file headers, comments, deviations, abbreviations) are in `style_common.md`. HLS-specific rules are in `style_hls.md`.

Header files use include guards: `#ifndef FILENAME_HPP` / `#define` / `#endif`.

## Naming

- Kernel functions are lower snake case free functions, named after what they compute: `pad<>()`, `input_gen<>()`, `pool<F>()`, `topk<>()`. CamelCase names such as `Matrix_Vector_Activate_Batch` or `StreamingDataWidthConverter_Batch` belong to the legacy API; keep them where they exist, but do not introduce new ones.
- Template value parameters are named in all-capital snake case (`H`, `W`, `CF`, `PAD_N`, `PE`, `SIMD`). Type parameters are `T` or `T`-prefixed (`TI`, `TO`, `TW`).
- Type definitions use a `_t` suffix (`flit_t`, `dat_t`).
- Local variables are lower snake case; `static` kernel state may use capital camel case like registers in RTL.

## Documentation

- Every public kernel has a block comment above it describing what it computes, its streaming behaviour (what it reads and writes per invocation, and when), and any constraints on its parameters.
- Template parameters are documented by trailing comments in the template parameter list:

```cpp
template<
	unsigned  H,	// height of input feature map
	unsigned  W,	// width of input feature map
	unsigned  CF,	// channel fold
	typename  T
>
void pad(hls::stream<T> &src, hls::stream<T> &dst, T const &pad_val = T{0,});
```

- Doxygen blocks (`\brief`, `\tparam`, `\param`) are accepted, notably in existing legacy code, but not required.

## Indentation

Tabs are used for indentation, one tab per indentation level. Tabs may not be used for visual alignment if such alignment would depend on a specific tab size.

## Preprocessor Directives

All preprocessor directives have their `#` as the first character of the line, regardless of the surrounding code's nesting level. Conditional directives are indented by one space per nesting level between the `#` and the directive name. Include guards do not open a nesting level:

```cpp
#ifndef UTIL_HPP
#define UTIL_HPP

#include <ap_int.h>
#if VERSION >= 20242
# include <ap_float.h>
# if VERSION >= 20252
#  include <ap_extra.h>
# endif
#endif

void foo(hls::stream<T> &src) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS pipeline II=1 style=flp
	if(!src.empty())  dst.write(src.read());
}
```

HLS pragma keywords are lowercase. Pipeline pragmas should typically choose a non-blocking style (`style=flp` or `style=frp`) to ensure live data is flushed, avoiding tail data getting stuck in a non-trivial pipeline.

## Includes

Includes are organized into groups separated by blank lines, ordered from general to specific:

1. Standard library headers (`<iostream>`, `<vector>`)
2. Vendor/tool headers (`<ap_int.h>`, `<hls_stream.h>`)
3. Project headers (`"cnpy.h"`)

## Declarations

Separate variable names from their complete type by at least two spaces in declarations. The complete type includes any `const` qualifiers but excludes pointer/reference symbols:

```cpp
size_t const  inner_dim_elems = 42;
std::vector<NpyT>  data_to_save;
cnpy::NpyArray const  arr = cnpy::npy_load(path);
```

## Return and Throw

Separate `return` and `throw` from the following expression by two spaces when letters meet:

```cpp
return  header;
throw  std::runtime_error("something went wrong");
```

A single space is sufficient when a non-letter follows the keyword:

```cpp
return -1;
throw "message";
```

## Const

Use east const exclusively: the `const` keyword follows the type it qualifies.

```cpp
size_t const  n = 1;          // const value
NpyT const *loaded_data;      // pointer to const
NpyT const *const  base;      // const pointer to const
std::string const &fname;      // reference to const
```

Use `const` aggressively for any value that is not modified after initialization.

## Pointers and References

Pointer (`*`) and reference (`&`) symbols are preceded by a space and side with the name — no space between the symbol and the name:

```cpp
NpyT const *loaded_data = arr.data<NpyT>();
hls::stream<PackedT> &out_stream;
std::vector<size_t> const &shape;
```

In template arguments and casts, where there is no name, the symbol attaches to the type:

```cpp
reinterpret_cast<ap_uint<ElemBits>*>(&loaded_elem);
reinterpret_cast<T const*>(data_holder_->data());
```

## Ternary Operator

Simple binary decisions fit on one line with `?` attached to the condition (no space before `?`):

```cpp
return  x < 2.0? 0 : 1 + f(x / 2.0);
```

Multi-decision chains use one line per case with aligned `?` and `:`, the final case as a `/* else */` comment:

```cpp
return
	x == 0?    1.0 :
	x  > 0?    f(x - 1) * 2.0 :
	/* else */ f(x + 1) / 2.0;
```

## Commas and Semicolons

Commas and semicolons are always followed by a space:

```cpp
return  fp_maxval(E, M) * fp_maxval(E2, M2);
template<typename T, size_t N>
hls::vector<T, N>  v;
minifloat<E, M>  x;
for(unsigned  i = 0; i < N; i++) {
```

## Type Conversions

Do not use C-style casts. When a conversion is needed, use constructor syntax. Type literals properly to avoid unnecessary conversions:

```cpp
pow(2.0, double(exp));  // constructor syntax, literal already typed
-int(M)                 // constructor syntax for sign conversion
```

Implicit conversions that do not create overload ambiguities are entirely acceptable and preferred over explicit casts.

## Visibility and Linkage

Implementation details that are not part of the public interface are declared `static` in the `.cpp` file. Only the minimal public API is exposed in the header.

Template functions that require a non-template backing implementation are split: the non-template core lives in the `.cpp` file, and thin template wrappers in the header forward to it.

## Scoping

Use block scopes `{ }` with a short comment to limit the lifetime of helper variables and to structure larger functions into logical sections:

```cpp
{ // preamble
	char  preamble[10];
	...
}
{ // shape
	auto const  loc1 = header.find("(");
	...
}
```

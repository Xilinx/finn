# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the minimal qonnx/finn helper modules generated for the PYNQ
driver. It checks two things quickly, without building or running any hardware:

1. The generated ``basic.py`` and ``data_packing.py`` contain no heavy imports
   (``onnx`` / ``bitstring``) and still import + work when those packages are
   made unavailable -- i.e. the deployment board does not need them.
2. The extracted subset of functions still works (a pack/unpack round-trip and
   ``gen_finn_dt_tensor``).
"""

import importlib.util
import numpy as np
import qonnx.core.datatype
import qonnx.util.basic
import sys

import finn.util.data_packing
from finn.transformation.fpgadataflow.make_driver import _generate_minimal_module

BASIC_FUNCS = [
    "roundup_to_integer_multiple",
    "gen_finn_dt_tensor",
]
BASIC_IMPORTS = (
    "import numpy as np\n"
    "from typing import cast\n\n"
    "from qonnx.core.datatype import BaseDataType, DataType, FixedPointType"
)
DATA_PACKING_FUNCS = [
    "finnpy_to_packed_bytearray",
    "_pack_whole_byte_container",
    "_pack_bit_double_reverse",
    "_pack_general",
    "finnpy_to_int_array",
    "int_array_to_packed_bytearray",
    "packed_bytearray_to_finnpy",
    "prepare_values",
    "unsiged_array_to_signed",
    "packed_bytearray_to_finnpy_fast",
    "data_prepared_to_finnpy_bipolar",
    "data_prepared_to_finnpy_ternary",
    "data_prepared_to_finnpy_fixed",
    "data_prepared_to_finnpy_int",
    "packed_bytearray_to_finnpy_float",
]
DATA_PACKING_IMPORTS = (
    "import numpy as np\n\n"
    "from qonnx.core.datatype import DataType\n"
    "from qonnx.util.basic import roundup_to_integer_multiple"
)


def _write_minimal_tree(root):
    """Emit the trimmed modules the same way the driver generator does, plus a
    verbatim (dependency-free) copy of datatype.py."""
    basic_py = root / "basic.py"
    data_packing_py = root / "data_packing.py"
    _generate_minimal_module(
        str(basic_py),
        qonnx.util.basic,
        [getattr(qonnx.util.basic, n) for n in BASIC_FUNCS],
        BASIC_IMPORTS,
    )
    _generate_minimal_module(
        str(data_packing_py),
        finn.util.data_packing,
        [getattr(finn.util.data_packing, n) for n in DATA_PACKING_FUNCS],
        DATA_PACKING_IMPORTS,
    )
    return basic_py, data_packing_py


def _load_module(monkeypatch, name, path):
    """Load a generated file under its real dotted name so downstream absolute
    imports resolve to it, and register it via monkeypatch for auto-cleanup."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _imported_top_level_modules(text):
    """Return the set of top-level module names imported by the given source."""
    modules = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("import "):
            spec = line[len("import ") :]
        elif line.startswith("from "):
            spec = line[len("from ") :].split(" import ", 1)[0]
        else:
            continue
        modules.add(spec.split()[0].split(".")[0].split(",")[0])
    return modules


def test_generated_modules_have_no_heavy_imports(tmp_path):
    basic_py, data_packing_py = _write_minimal_tree(tmp_path)
    for path in (basic_py, data_packing_py):
        text = path.read_text()
        imported = _imported_top_level_modules(text)
        assert "onnx" not in imported, f"{path.name} unexpectedly imports onnx"
        assert "bitstring" not in imported, f"{path.name} unexpectedly imports bitstring"
        # the license header and "minimal version" note should be preserved
        assert "minimal version of" in text
        assert "Copyright" in text


def test_generated_modules_import_and_roundtrip(tmp_path, monkeypatch):
    basic_py, data_packing_py = _write_minimal_tree(tmp_path)
    # make the heavy deps unavailable: importing them now raises ImportError
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "bitstring", None)
    # load the trimmed modules; data_packing's "from qonnx.util.basic import ..."
    # must resolve to the trimmed basic we just registered
    gen_basic = _load_module(monkeypatch, "qonnx.util.basic", basic_py)
    gen_dp = _load_module(monkeypatch, "finn.util.data_packing", data_packing_py)

    dt = qonnx.core.datatype.DataType["INT4"]
    x = gen_basic.gen_finn_dt_tensor(dt, (2, 8))
    packed = gen_dp.finnpy_to_packed_bytearray(x, dt, reverse_inner=True, reverse_endian=True)
    y = gen_dp.packed_bytearray_to_finnpy(
        packed, dt, x.shape, reverse_inner=True, reverse_endian=True
    )
    assert np.array_equal(x, y)

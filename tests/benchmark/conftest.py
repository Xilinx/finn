# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Put this directory on sys.path so the benchmark tests in per-model subfolders
# can import the shared benchmark_helpers module by bare name.
import os
import sys

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

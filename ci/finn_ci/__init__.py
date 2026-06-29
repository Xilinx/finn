# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN CI helpers.

A small package, importable without the finn package installed, that backs the
FINN Jenkins CI pipeline and the pytest sharding plugin.

Submodules:
  config   - the CI board and stage tables and the pure helpers over them
  sharding - deterministic weight-balanced group-to-shard assignment
  jsonio   - the JSON read helper shared across the package
  plugin   - the pytest plugin that selects a shard and captures timings
"""

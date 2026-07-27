# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN CI helpers.

A small package, importable without the finn package installed, that backs the
FINN Jenkins CI pipeline and the pytest sharding plugin. The build pipeline
drives the CLI with ``PYTHONPATH=ci python3 -m finn_ci <subcommand>``.

Submodules:
  config    - the CI board and stage tables and the pure helpers over them
  sharding  - deterministic weight-balanced group-to-shard assignment
  jsonio    - the JSON read/write helpers shared across the package
  plugin    - the pytest plugin that selects a shard and captures timings
  timing    - the self-maintaining per-group timing master and summaries
  retention - image/artifact/snapshot tree rotation and pip-cache pruning
  lsf       - bjobs orphan-job parsing for the build reaper
  failures  - the stdlib JUnit failure printer
  __main__  - the CLI dispatched by python3 -m finn_ci
"""

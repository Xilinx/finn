#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
skip_xsi=0
skip_resources=0
for arg in "$@"; do
    case "$arg" in
        --ci) ;;
        --skip-xsi) skip_xsi=1 ;;
        --skip-resources|--skip-deps) skip_resources=1 ;;
        --help|-h)
            echo 'Usage: setup-local.sh [--ci] [--skip-xsi] [--skip-resources]'
            echo 'Requires Python 3.12, pip, git and a C++ compiler; bootstraps pinned uv locally if needed.'
            echo 'FINN_VENV selects the environment; FINN_PYTHON selects the interpreter.'
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
for command in git g++; do
    command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done
. "$root/scripts/ensure-uv.sh"
cd "$root"
export UV_PYTHON_DOWNLOADS=${UV_PYTHON_DOWNLOADS:-never}
export UV_PROJECT_ENVIRONMENT=${FINN_VENV:-$root/.venv}
[ "$skip_resources" = 0 ] || export FINN_RESOURCES_OFFLINE=1
uv sync --locked --python "${FINN_PYTHON:-python3.12}"
if [ "$skip_resources" = 0 ]; then
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m finn.resources fetch --kind hls-include --kind vivado-boards
fi
# Activation performs no installation; --skip-resources requires prepared caches
# or local resource overrides (FINN_RESOURCES_OFFLINE=1 enforces this).
. "$root/scripts/activate.sh"
python -c 'import finn, qonnx, brevitas, finnexperimental, torch, onnxruntime'
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
if [ "$skip_xsi" = 0 ] && [ -n "${XILINX_VIVADO:-}" ]; then
    python -m finn.xsi.setup --quiet
fi
if [ "${FINN_VCK190_GOLDEN:-0}" = 1 ]; then
    python -m finn.resources.prepare golden
fi
printf 'FINN ready. Activate with: source %q/scripts/activate.sh\n' "$root"

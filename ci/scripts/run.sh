#!/usr/bin/env bash
# Compiler tests use the full upstream runtime profile. Host-side finn_ci and
# board-side PYNQ/XRT environments remain separate.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
export FINN_RUNTIMES=${FINN_RUNTIMES:-xrt}
exec "$root/docker/run" --fpga -- "$@"

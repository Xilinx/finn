#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
. "$REPO/docker/lib.sh"
export FINN_RUNTIMES=${FINN_RUNTIMES:-xrt}
finn_prepare_runtime_inputs
finn_set_provenance
cd "$REPO"
tag=$(finn_bake_tag "$(finn_bake_target "$FINN_RUNTIMES")")
python3 "$REPO/docker/image.py" metadata "$REPO/image-metadata" "$tag"

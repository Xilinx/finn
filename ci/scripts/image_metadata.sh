#!/usr/bin/env bash
# Record the provenance of the locally built image in the Jenkins workspace.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export FINN_RUNTIMES=${FINN_RUNTIMES:-xrt}
exec python3 "$REPO/docker/launcher.py" image metadata "$REPO/image-metadata"

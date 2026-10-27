#!/usr/bin/env bash
# Publish the prepared image and archive its provenance in the Jenkins workspace.
set -euo pipefail
[ "$#" -eq 3 ] || { echo 'Usage: publish_docker_image.sh IMAGE_DIR IMAGE BUILD' >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
. "$REPO/docker/lib.sh"
export FINN_RUNTIMES=${FINN_RUNTIMES:-xrt}
finn_set_provenance
exec python3 "$REPO/docker/image.py" publish "$1" "$2" --workspace "$REPO/image-metadata"

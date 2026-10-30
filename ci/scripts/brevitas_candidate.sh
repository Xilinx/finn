#!/usr/bin/env bash
# A uv overlay changes this command only; the base uv.lock stays untouched.
set -euo pipefail
cd "${FINN_ROOT:?}"
export FINN_BREVITAS_PROVENANCE=${FINN_BREVITAS_PROVENANCE:-brevitas-provenance}
rm -f -- brevitas_suite.xml brevitas_suite.html
rm -rf -- "$FINN_BREVITAS_PROVENANCE" coverage_brevitas_suite
if [[ ! ${BREVITAS_COMMIT:-} =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo 'BREVITAS_COMMIT must be a full Git commit SHA' >&2
    exit 2
fi
export FINN_BREVITAS_EXPECTED_COMMIT=${BREVITAS_COMMIT,,}
# Keep the rest of FINN's baseline selected. Without this generated uv export,
# the overlay independently upgrades Torch and can break the base torchvision.
# Candidate-only new dependencies remain free to resolve and are recorded.
mkdir -p "$FINN_BREVITAS_PROVENANCE"
uv export --locked --no-emit-project --no-emit-local --no-emit-package brevitas \
    --no-hashes --output-file "$FINN_BREVITAS_PROVENANCE/base-requirements.txt" >/dev/null
printf '%s\n' '-c base-requirements.txt' > "$FINN_BREVITAS_PROVENANCE/overlay-constraints.txt"
sha256sum uv.lock > "$FINN_BREVITAS_PROVENANCE/base-lock.sha256"
status=0
uv run --locked --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --with-requirements "$FINN_BREVITAS_PROVENANCE/overlay-constraints.txt" \
    --with "brevitas @ git+https://github.com/Xilinx/brevitas.git@${FINN_BREVITAS_EXPECTED_COMMIT}" \
    python -m pytest -p finn_ci.brevitas_candidate -m brevitas_export "$@" || status=$?
sha256sum --check "$FINN_BREVITAS_PROVENANCE/base-lock.sha256"
exit "$status"

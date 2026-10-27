# shellcheck shell=bash
# Build identity and image preparation shared by docker/build and docker/run.
gecho() { echo "$*" >&2; }
recho() { echo "ERROR: $*" >&2; }
yecho() { echo "WARNING: $*" >&2; }

finn_normalize_runtimes () {
    printf '%s' "${1:-}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

# The image identity: a hash of the files listed in docker/image-inputs.txt plus
# the build arguments that change image content. FINN's sources are not inputs;
# containers install the mounted checkout. Image IDs still identify the exact
# build.
finn_image_revision () (
    set -o pipefail
    if [ -n "${FINN_IMAGE_REVISION:-}" ]; then
        case "$FINN_IMAGE_REVISION" in
            *[!A-Za-z0-9_.-]*|"")
                recho "FINN_IMAGE_REVISION contains characters invalid in a Docker tag"
                return 2
                ;;
        esac
        printf '%s' "$FINN_IMAGE_REVISION"
        return 0
    fi

    local repo manifest pattern optional path found bits mode checksum result patterns runtime_reader
    repo="${FINN_IMAGE_INPUT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    manifest="$repo/docker/image-inputs.txt"
    [ -f "$manifest" ] || { recho "image input manifest not found: $manifest"; return 2; }
    runtime_reader="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime_inputs.py"
    patterns=$(cat "$manifest" && python3 "$runtime_reader" "${FINN_RUNTIMES:-}" \
        --packages --root "$repo") || return 2

    result=$(
    (
        cd "$repo" || exit 1
        shopt -s globstar
        printf 'finn-image-inputs-v5\n'
        while IFS= read -r pattern || [ -n "$pattern" ]; do
            case "$pattern" in ""|\#*) continue ;; esac
            optional=0
            case "$pattern" in \?*) optional=1; pattern=${pattern#?} ;; esac
            found=0
            while IFS= read -r path; do
                [ -f "$path" ] || continue
                case "$path" in */__pycache__/*|*.pyc|*.so|*.egg-info/*) continue ;; esac
                found=1
                # Git records the owner's executable bit, not agent umasks.
                # Canonical modes keep identity stable across Jenkins nodes.
                bits=$(stat -c '%a' "$path")
                mode=100644
                (( (8#$bits & 8#100) == 0 )) || mode=100755
                printf 'path=%s mode=%s sha256=' "$path" "$mode"
                checksum=$(sha256sum "$path") || exit 2
                printf '%s\n' "${checksum%% *}"
            done < <(compgen -G "$pattern" | LC_ALL=C sort || true)
            if [ "$found" = 0 ] && [ "$optional" = 0 ]; then
                recho "image input pattern matched no files: $pattern"
                exit 2
            fi
        done <<< "$patterns"
        # Build arguments that change image contents without changing a file.
        # The runtime set is part of the tag suffix instead.
        printf 'arg=UBUNTU_TAG=%s\n' "${UBUNTU_TAG:-noble-20240605}"
    ) | sha256sum | awk '{print "img-" substr($1, 1, 16)}'
    ) || return 2
    printf '%s' "$result"
)

finn_source_revision () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    git -C "$repo" rev-parse HEAD 2>/dev/null || printf '%s' unknown
}

finn_source_describe () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    git -C "$repo" describe --always --tags --abbrev=12 --dirty 2>/dev/null || printf '%s' unknown
}

finn_source_dirty () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local state
    if ! state=$(git -C "$repo" status --porcelain --untracked-files=normal 2>/dev/null); then
        printf '%s' unknown
    elif [ -z "$state" ]; then
        printf '0'
    else
        printf '1'
    fi
}

finn_set_provenance () {
    FINN_IMAGE_REVISION=$(finn_image_revision) || return
    FINN_SOURCE_REVISION=$(finn_source_revision)
    FINN_SOURCE_DESCRIBE=$(finn_source_describe)
    FINN_SOURCE_DIRTY=$(finn_source_dirty)
    export FINN_IMAGE_REVISION FINN_SOURCE_REVISION FINN_SOURCE_DESCRIBE FINN_SOURCE_DIRTY
}

# The bake target for a runtime set and an optional release variant.
#
#   finn_bake_target ""              -> finn
#   finn_bake_target "xrt"           -> finn-xrt
#   finn_bake_target "xrt,slash"     -> finn-runtime        (parameterized)
#   finn_bake_target "xrt" release   -> finn-release        (parameterized)
finn_bake_target () {
    local runtimes variant="${2:-}"
    runtimes=$(finn_normalize_runtimes "${1:-}")
    case "$variant:$runtimes" in
        :)          printf '%s' finn ;;
        :xrt)       printf '%s' finn-xrt ;;
        release:*)  printf '%s' finn-release ;;
        *)          printf '%s' finn-runtime ;;
    esac
}

# Ask bake for a target's tag. The ONE place that parses bake's output.
#
# Two things here are load-bearing and were each missing somewhere:
#
#   -f docker-bake.hcl   without it bake also auto-loads compose.yaml and fails
#                        to interpolate ${FINN_XILINX_PATH:?} on a machine with
#                        no Xilinx, refusing every target including ones that
#                        have nothing to do with the toolchain.
#   sed -n '/^{/,$p'     bake writes progress lines to stdout before the JSON.
#                        Without this, json.load sees "#1 reading ..." and dies.
finn_bake_tag () {
    local target="$1" request
    request=$(docker buildx bake -f docker-bake.hcl --print "$target") || {
        recho "Unable to resolve $target. Docker Buildx with Bake is required; install docker-buildx-plugin and check 'docker buildx bake --help'."
        return 1
    }
    printf '%s\n' "$request" | sed -n '/^{/,$p' \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['target']['$target']['tags'][0])"
}

finn_prepare_runtime_inputs () {
    FINN_RUNTIMES=$(python3 "$REPO/docker/runtime_inputs.py" "${FINN_RUNTIMES:-}") || return
    export FINN_RUNTIMES
}

# Build one Bake target. Image references are resolved separately.
finn_bake_build () {
    local target="$1"; shift
    python3 "$REPO/docker/runtime_inputs.py" "${FINN_RUNTIMES:-}" --cache-xrt || return
    gecho "Building $target"
    # shellcheck disable=SC2086
    docker buildx bake -f docker-bake.hcl --load \
        --set "$target.output=type=docker,compression=gzip,compression-level=1" "$@" "$target" \
        || { recho "docker buildx bake $target failed; Docker Buildx with Bake is required"; return 1; }
}

# Prepare a Docker-daemon image for any consumer. Sets FINN_IMAGE. Explicit
# builds refresh cached layers; runs reuse the selected environment-input tag.
finn_prepare_image () {
    local target="$1" mode="${2:-ensure}"
    local build_args=()
    if [ "$mode" != build ] && [ -n "${FINN_DOCKER_SHARED_IMAGE_DIR:-}" ]; then
        if FINN_IMAGE=$(python3 "$REPO/docker/image.py" load "$FINN_DOCKER_SHARED_IMAGE_DIR"); then
            export FINN_IMAGE
            return 0
        elif [ "${FINN_DOCKER_PREBUILT:-0}" = 1 ]; then
            return 1
        fi
        yecho "Shared image unavailable; falling back to the local environment image"
    fi
    if [ "$mode" = build ]; then
        FINN_IMAGE=$(finn_bake_tag "$target") || return 1
    else
        FINN_IMAGE=${FINN_IMAGE:-$(finn_bake_tag "$target")} || return 1
    fi
    [ -n "$FINN_IMAGE" ] || { recho "No image reference for $target"; return 1; }
    export FINN_IMAGE
    if [ "$mode" != build ] && { [ "${FINN_CONTAINER_NO_BUILD:-0}" = 1 ] || [ "${FINN_DOCKER_PREBUILT:-0}" = 1 ]; }; then
        docker image inspect "$FINN_IMAGE" >/dev/null 2>&1 || {
            recho "No prepared Docker image for $FINN_IMAGE; run ./docker/build first."
            return 1
        }
        FINN_IMAGE=$(python3 "$REPO/docker/image.py" inspect "$FINN_IMAGE") || return
        export FINN_IMAGE
        return 0
    fi
    if [ "$mode" = ensure ] && [ "${FINN_CONTAINER_REBUILD:-0}" != 1 ] \
       && [ -z "${FINN_DOCKER_BUILD_EXTRA:-}" ] \
       && docker image inspect "$FINN_IMAGE" >/dev/null 2>&1; then
        FINN_IMAGE=$(python3 "$REPO/docker/image.py" inspect "$FINN_IMAGE") || return
        export FINN_IMAGE
        return 0
    fi
    [ "${FINN_CONTAINER_REBUILD:-0}" != 1 ] || build_args+=(--no-cache)
    # Legacy build flags are a shell word list, never evaluated as shell code.
    # shellcheck disable=SC2206
    build_args+=(${FINN_DOCKER_BUILD_EXTRA:-})
    finn_bake_build "$target" "${build_args[@]}" || return
    # Execute by immutable ID, including local runs.
    FINN_IMAGE=$(python3 "$REPO/docker/image.py" inspect "$FINN_IMAGE") || return
    export FINN_IMAGE
}

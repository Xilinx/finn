#!/bin/sh
# Install the selected runtime targets. Runs as root in the `runtime` stage of
# docker/Dockerfile.finn.
#
#   FINN_RUNTIMES=xrt,slash sh install-runtimes.sh
#
# Reads one manifest per name from /tmp/runtimes/<name>.env and installs the
# packages it declares. See docker/runtimes/README.md for the manifest format
# and for what is allowed to be a runtime target.
#
# Shared installation logic for all selected runtime packages.
#
# POSIX sh. This runs before anything guarantees bash is the shell.

set -eu

RUNTIME_DIR=/tmp/runtimes
PACKAGE_DIR=/tmp/packages

: "${FINN_RUNTIMES:=}"

if [ -z "$FINN_RUNTIMES" ]; then
    echo "No runtime targets selected; this is the base image."
    exit 0
fi

run_os=$(. /etc/os-release; echo "$VERSION_ID")
debs=""
# Fetched packages are written into /tmp inside this same RUN, so they land in
# the layer unless we delete them here. Supplied ones arrive through a bind
# mount and never do, which is why the two are tracked apart.
fetched=""

# ---------------------------------------------------------------------------
# Resolve every manifest FIRST, install nothing.
#
# A supply target whose file is missing must fail before any apt-get runs.
# Installing XRT and then discovering slash.deb is absent wastes a long
# download and leaves a half-built layer that looks like a network fault.
# ---------------------------------------------------------------------------

for name in $(echo "$FINN_RUNTIMES" | tr ',' ' '); do
    manifest="$RUNTIME_DIR/$name.env"
    if [ ! -f "$manifest" ]; then
        echo "ERROR: no runtime target named '$name'." >&2
        echo "       Expected $manifest (from docker/runtimes/$name.env)." >&2
        echo "       Available: $(find "$RUNTIME_DIR" -maxdepth 1 -type f -name '*.env' -printf '%f\n' 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')" >&2
        exit 1
    fi

    # Clear first. The keys are deliberately short and generic (SOURCE, DEB,
    # URL), and every manifest is sourced into this one shell, so without the
    # unset a second manifest silently inherits the first's values -- a
    # supply target after a fetch target would install the fetched deb again.
    unset SOURCE DEB URL FALLBACK_URL SHA256 FILES EXPECT_OS
    # shellcheck disable=SC1090
    . "$manifest"

    : "${SOURCE:?$manifest does not set SOURCE (fetch or supply)}"

    if [ -n "${EXPECT_OS:-}" ] && [ "$EXPECT_OS" != "$run_os" ]; then
        echo "ERROR: runtime '$name' targets Ubuntu $EXPECT_OS but this image is Ubuntu $run_os." >&2
        echo "       apt will usually resolve a mismatched package rather than refuse it," >&2
        echo "       which gives a subtly wrong runtime instead of a clean failure." >&2
        echo "       Fix docker/runtimes/$name.env or the image base, not both." >&2
        exit 1
    fi

    case "$SOURCE" in
    fetch)
        : "${DEB:?$manifest is SOURCE=fetch but does not set DEB}"
        : "${URL:?$manifest is SOURCE=fetch but does not set URL}"
        out="/tmp/$DEB.deb"
        echo "runtime '$name': fetching $DEB"
        if ! wget -q "$URL" -O "$out"; then
            if [ -n "${FALLBACK_URL:-}" ]; then
                echo "  primary URL failed, trying the fallback" >&2
                rm -f "$out"
                wget -q "$FALLBACK_URL" -O "$out"
            else
                echo "ERROR: could not download $name from $URL" >&2
                exit 1
            fi
        fi
        if [ -n "${SHA256:-}" ]; then
            echo "$SHA256  $out" | sha256sum -c -
        else
            # Not fatal: a user pinning an unpublished build has nothing to
            # check against. Loud, because this is an unauthenticated fetch
            # installed as root.
            echo "WARNING: $name downloaded with no checksum. Set SHA256 in" >&2
            echo "         docker/runtimes/$name.env to pin it. Actual sum:" >&2
            sha256sum "$out" >&2
        fi
        debs="$debs $out"
        fetched="$fetched $out"
        ;;
    supply)
        : "${FILES:?$manifest is SOURCE=supply but does not set FILES}"
        for f in $FILES; do
            if [ ! -f "$PACKAGE_DIR/$f" ]; then
                echo "ERROR: runtime '$name' needs a package you have not supplied." >&2
                echo "       Put it at: docker/packages/$f" >&2
                echo "       See docker/runtimes/$name.env for where to get it." >&2
                exit 1
            fi
            echo "runtime '$name': using supplied $f"
            debs="$debs $PACKAGE_DIR/$f"
        done
        ;;
    *)
        echo "ERROR: $manifest has SOURCE=$SOURCE; expected 'fetch' or 'supply'." >&2
        exit 1
        ;;
    esac
done

# ---------------------------------------------------------------------------
# Install. One apt-get, so packages that depend on each other resolve together
# regardless of the order the user listed them.
# ---------------------------------------------------------------------------

echo "Installing:$debs"
apt-get update
# shellcheck disable=SC2086
apt-get install -y --no-install-recommends $debs
rm -rf /var/lib/apt/lists/*
# An `if`, not `[ -n ] && rm`. Under `set -e` a false test as the last command
# in a script exits 1, and this is one line away from being last.
if [ -n "$fetched" ]; then
    # shellcheck disable=SC2086
    rm -f $fetched
fi
# Supplied packages need no cleanup: they arrive through RUN --mount=type=bind
# and are never written into a layer.
echo "Runtime targets installed: $FINN_RUNTIMES"

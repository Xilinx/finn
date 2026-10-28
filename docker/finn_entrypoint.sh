#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
set -eo pipefail
# Arbitrary caller IDs have no passwd entry or writable image home.
if [ -z "${HOME:-}" ] || [ "$HOME" = / ] || ! [ -w "$HOME" ]; then
    export HOME="/tmp/finn-home-$(id -u)"
fi
mkdir -p "$HOME"
if [ -n "${FINN_CONTAINER_SSH_DIR:-}" ] && [ ! -e "$HOME/.ssh" ]; then
    ln -s "$FINN_CONTAINER_SSH_DIR" "$HOME/.ssh"
fi
export USER=${USER:-finn}
export LOGNAME=${LOGNAME:-$USER}
# docker/run mounts the host's FINN_HOME; elsewhere (e.g. Apptainer) it is ~/.finn.
export FINN_BUILD_DIR=${FINN_BUILD_DIR:-/tmp/finn_build_$(id -u)}
export FINN_HOME=${FINN_HOME:-$HOME/.finn}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FINN_BUILD_DIR/.uv-cache}
mkdir -p "$FINN_HOME" "$UV_CACHE_DIR"
# libnss_wrapper is disabled until these variables name real files. Preserve
# all image service users and add the caller, including on a read-only SIF.
_finn_identity=$(python /usr/local/lib/finn/identity.py \
    --directory "${FINN_IDENTITY_DIR:-$FINN_HOME/identity/$(id -u)}" --username "$USER" --home "$HOME")
eval "$_finn_identity"
unset _finn_identity
if [ "${FINN_SYNC:-1}" != 0 ]; then
    : "${FINN_ROOT:?FINN_ROOT must name the checkout to install}"
    test -f "$FINN_ROOT/uv.lock" || { echo "No uv.lock at $FINN_ROOT" >&2; exit 1; }
    cmp "$FINN_ROOT/uv.lock" /opt/finn/uv.lock || {
        echo 'FINN lock differs from the prepared image; run docker/build.' >&2; exit 1;
    }
    # Only FINN changes at startup. Dependencies were checked during the build.
    uv pip install --no-deps --no-build-isolation -e "$FINN_ROOT"
fi
. /etc/finn-toolchain.sh
if [ -n "${FINN_TOOL_DIR_OVERRIDE:-}" ] && [ -z "${FINN_RESOURCES_HLSLIB:-}" ]; then
    # Farm jobs see the shared build tree, not the image's /opt resource cache.
    python -m finn.resources fetch hlslib >/dev/null || exit 1
fi
FINN_HLSLIB_PATH=$(python -m finn.resources path hlslib) || exit 1
export FINN_HLSLIB_PATH
if [ -n "${XILINX_VIVADO:-}${FINN_BOARD_FILES_PATH:-}" ]; then
    FINN_BOARD_FILES_PATH=$(python -m finn.resources.prepare boards) || exit 1
    export FINN_BOARD_FILES_PATH
fi
# Restore upstream's opt-in device initialization scripts in the caller's HOME.
if [ -d "$FINN_ROOT/.Xilinx" ]; then
    mkdir -p "$HOME/.Xilinx"
    cp -a "$FINN_ROOT/.Xilinx/." "$HOME/.Xilinx/"
fi
if [ "${FINN_PREPARE_XSI:-1}" = 1 ] && [ -n "${XILINX_VIVADO:-}" ]; then
    python -m finn.xsi.setup --quiet
fi
if [ "${FINN_VCK190_GOLDEN:-0}" = 1 ]; then
    python -m finn.resources.prepare golden
fi
# This marker means preparation succeeded. Failed setup never starts the command.
touch /tmp/finn-ready
exec "$@"

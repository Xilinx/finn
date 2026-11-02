#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Source this helper. Bootstrap only the pinned tool into the user's cache;
# compiler dependencies are installed separately and exclusively from uv.lock.
_finn_ensure_uv() {
    local root version current tool_dir binary
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || return 1
    version=$(cat "$root/.uv-version") || return 1
    current=$(uv --version 2>/dev/null | cut -d' ' -f2) || current=""
    [ "$current" != "$version" ] || return 0
    tool_dir="${XDG_CACHE_HOME:-$HOME/.cache}/finn/uv-tool/$version"
    binary="$tool_dir/bin/uv"
    if [ ! -x "$binary" ]; then
        "${FINN_PYTHON:-python3.12}" -m pip install --disable-pip-version-check \
            --no-deps --upgrade --target "$tool_dir" "uv==$version" >&2 || return 1
    fi
    [ -x "$binary" ] || binary="$tool_dir/uv/uv"
    current=$("$binary" --version 2>/dev/null | cut -d' ' -f2) || return 1
    [ "$current" = "$version" ] || { echo "Failed to provision uv $version" >&2; return 1; }
    export PATH="$(dirname "$binary"):$PATH"
}
if _finn_ensure_uv; then
    unset -f _finn_ensure_uv
else
    unset -f _finn_ensure_uv
    return 1
fi

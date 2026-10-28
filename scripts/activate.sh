#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Source this file after setup-local.sh. It does not install packages.
_finn_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FINN_ROOT=$_finn_root
_finn_venv=${FINN_VENV:-$FINN_ROOT/.venv}
if [ ! -f "$_finn_venv/bin/activate" ]; then
    echo "FINN environment missing: run $FINN_ROOT/setup-local.sh" >&2
    return 1
fi
. "$_finn_venv/bin/activate"
# FINN_HOME (default ~/.finn), FINN_BUILD_DIR (default /tmp/finn_build_<uid>), the
# selected tools and licence. Values are shell-quoted by the stdlib launcher.
_finn_settings=$(python3 "$FINN_ROOT/docker/launcher.py" inspect --tier auto --workspace-policy mirror --format sh) || return
eval "$(printf '%s\n' "$_finn_settings" | sed 's/^/export /')"
unset FINN_ENV_APPLIED
. "$FINN_ROOT/docker/finn-toolchain.sh"
if [ -n "${FINN_TOOL_DIR_OVERRIDE:-}" ] && [ -z "${FINN_RESOURCES_HLSLIB:-}" ]; then
    python3 -m finn.resources fetch hlslib >/dev/null || return
fi
FINN_HLSLIB_PATH=$(python3 -m finn.resources path hlslib) || return
export FINN_HLSLIB_PATH
if [ -n "${XILINX_VIVADO:-}" ]; then
    FINN_BOARD_FILES_PATH=$(python3 -m finn.resources.prepare boards) || return
    export FINN_BOARD_FILES_PATH
fi
export XILINX_LOCAL_USER_DATA=no
export VIVADO_IP_CACHE=${VIVADO_IP_CACHE:-$FINN_BUILD_DIR/vivado_ip_cache}
unset _finn_settings _finn_root _finn_venv

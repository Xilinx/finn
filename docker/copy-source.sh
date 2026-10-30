#!/bin/sh
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Embed source assets without unselected or redundant supplied runtime archives.
set -eu
[ "$#" -eq 2 ] || { echo 'Usage: copy-source.sh TRACKED_SOURCE_TAR DESTINATION' >&2; exit 2; }
archive=$1
destination=$2
mkdir -p "$destination"
tar -C "$destination" -xf "$archive"

# A restrictive checkout umask must not make embedded sources unreadable to callers.
chmod -R a+rX "$destination"

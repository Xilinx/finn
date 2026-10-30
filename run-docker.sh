#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# run-docker.sh was replaced by docker/build and docker/run. This stub prints
# the equivalent command for the requested mode and fails, so scripts that
# still call it stop with a pointer instead of running something else.
root=$(cd "$(dirname "$0")" && pwd)
quote() { local out="" word; for word in "$@"; do out+=" $(printf '%q' "$word")"; done; printf '%s' "$out"; }
absolute() { (cd "$1" 2>/dev/null && pwd) || printf '%s' "$1"; }
mode=${1:-}
case "$mode" in
    "") replacement="$root/docker/run" ;;
    test) replacement="$root/docker/run --fpga -- python -m pytest" ;;
    quicktest) shift; replacement="$root/docker/run -- bash docker/quicktest.sh$(quote "$@")" ;;
    notebook) replacement="$root/docker/run --notebook" ;;
    print-tag) replacement="$root/docker/build --print-tag" ;;
    build_dataflow)
        dir=$(absolute "${2:?build_dataflow needs a folder}")
        replacement="$root/docker/run --fpga --volume$(quote "$dir:$dir") -- build_dataflow$(quote "$dir")" ;;
    build_custom)
        dir=$(absolute "${2:?build_custom needs a folder}")
        flow=${3:-build}
        shift; shift; [ "$#" -eq 0 ] || shift
        replacement="$root/docker/run --fpga --volume$(quote "$dir:$dir") --workdir$(quote "$dir") -- python$(quote "$flow.py" "$@")" ;;
    *) replacement="$root/docker/run --$(quote "$@")" ;;
esac
{
    echo "run-docker.sh has been replaced by docker/build and docker/run."
    echo "The equivalent of this command is:"
    echo
    echo "    $replacement"
    echo
    echo "Environment variables have changed too; see $root/docker/README.md (Interface migration)."
} >&2
exit 2

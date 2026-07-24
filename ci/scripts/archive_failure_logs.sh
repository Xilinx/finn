#!/bin/bash
# archive_failure_logs.sh <build_dir> <tarball_path> [start_marker]
#
# One tarball of tool logs per failed shard. LSF staging logs live outside
# the build dir under FINN_LSF_NFS_STAGING. They are scoped to files newer
# than the start_marker if provided.
# Best-effort: failures are logged but never abort the pipeline (the real
# test result is owned by pytest).
set +e

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: archive_failure_logs.sh <build_dir> <tarball_path> [start_marker]" >&2
  exit 2
fi

bd=$1
tarball=$2
start_marker=${3:-}
lsf_staging="${FINN_LSF_NFS_STAGING:-}"

mkdir -p "$(dirname "$tarball")"
if [ ! -d "$bd" ]; then
  exit 0
fi

# Use absolute paths so tar can stat them from its own cwd.
abs_bd=$(cd "$bd" && pwd)
# The LSF staging scan is only useful when scoped to files newer than the
# real per-shard start. Without a marker the find would walk the entire
# shared staging dir, so we skip the LSF block entirely instead.
newer_ref=
if [ -n "$start_marker" ] && [ -e "$start_marker" ]; then
  newer_ref=$start_marker
fi

# Collect to a temp file so we can both count and tar, and so a healthy
# "no candidates" run does not produce an empty tarball indistinguishable
# from a tar failure.
list=$(mktemp)
trap 'rm -f "$list"' EXIT

# Two-pass capture. First pass: basenames that are unambiguously FINN/Vitis
# artefacts anywhere in the build dir. Second pass: generic names (config.txt
# etc.) gated by grep to the build-subdir families, so they only match inside
# those subtrees. The leading-slash anchor stops 'myvitis_proj' matching
# 'vitis_proj'; trailing '/' marks a fixed dir name, trailing '_' a
# '<name>_<hash>' prefix.
build_subdirs='/(project_|finn_zynqbuild_|vitis_proj/|vivado_stitch_proj_|vitis_link_proj_)'
{
  find "$abs_bd" -type f \( \
      -name 'vitis_hls.log' -o \
      -name 'build_dataflow.log' -o \
      -name 'vivado.log' -o \
      -name 'v++_a.log' -o \
      -name 'v++.link_summary' -o \
      -name 'link.steps.log' -o \
      -name '*runme.log' \
    \) -print0 2>/dev/null
  find "$abs_bd" -type f \( \
      -name 'config.txt' -o \
      -name 'run_vitis_link.sh' -o \
      -name '*.fcnmap.xml' -o \
      -name 'xd_ip_index.xml' \
    \) -print0 2>/dev/null | grep -zE "$build_subdirs"
  if [ -n "$newer_ref" ] && [ -d "$lsf_staging" ]; then
    find "$lsf_staging" -mindepth 2 -maxdepth 3 -type f -newer "$newer_ref" \( \
        -name 'lsf.stdout' -o \
        -name 'lsf.stderr' -o \
        -name 'remote_runner.sh' \
      \) -print0 2>/dev/null
  fi
} > "$list"

n=$(tr -cd '\0' < "$list" | wc -c)
echo "[archive-failure-logs] ${n} candidate file(s) for ${tarball}"
if [ "$n" = "0" ]; then
  exit 0
fi
tar --null --create --gzip --file "$tarball" --files-from "$list" 2>/dev/null || true

#!/bin/bash
# find_copy_zip.sh <test_type> <board> <find_dir> <stage_dir>
#
# Walks <find_dir> for per-shard hw_deployment_*/<board>/* directories and
# stages each model dir under <stage_dir>/<board>/. The Jenkinsfile calls
# this once per (testType, board) shard from runShardBody.
# NOT best-effort. A half-staged deployment tree would silently lose models
# at aggregate time, so strict mode is on and any error aborts the step.
# See also: publish_board_zip_stage.sh, which performs the matching
# aggregate-time walk over the staged trees this script produced.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: find_copy_zip.sh <test_type> <board> <find_dir> <stage_dir>" >&2
  exit 2
fi

test_type=$1
board=$2
find_dir=$3
stage_dir=$4
tag="findCopyZip(${test_type}/${board})"

if [ ! -d "$find_dir" ]; then
  exit 0
fi

mkdir -p "$stage_dir/$board"
# u+w so a previous run's read-only residue can be removed. ignore failures
# on a freshly-created dir.
chmod -R u+w "$stage_dir/$board" 2>/dev/null || true
# find -mindepth 1 catches dotfiles that glob */* would miss.
find "${stage_dir:?}/${board:?}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

found=0
while IFS= read -r -d '' board_dir; do
  found=1
  for model_dir in "$board_dir"/*; do
    [ -d "$model_dir" ] || continue
    name=$(basename "$model_dir")
    if [ -e "$stage_dir/$board/$name" ]; then
      echo "$tag: duplicate deployment $name across hw_deployment dirs" >&2
      exit 1
    fi
    # cp not mv: the per-shard build dir is kept for failure triage. Trees
    # are small (per-board deployment dirs).
    cp -a "$model_dir" "$stage_dir/$board/"
  done
done < <(find "$find_dir" -maxdepth 2 -type d -name "$board" -path '*/hw_deployment_*/*' -print0)

if [ "$found" = "0" ]; then
  exit 0
fi

echo "$tag: staged deployments under $stage_dir"

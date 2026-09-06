#!/bin/bash
# publish_board_zip_stage.sh <src_root> <work_board> <board> <test_type>
#
# Stages per-shard board deployments into work_board ahead of zipping.
# Hard-fails on duplicate model names across shards (a real conflict).
# Touches <work_board>/.NO_DEPLOYMENTS when nothing was found so the caller
# can short-circuit cleanly.
# NOT best-effort. A half-staged board zip is worse than no zip, so strict
# mode is on and any error aborts the publish step.
# See also: find_copy_zip.sh, which produced the per-shard staged trees
# this script aggregates.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: publish_board_zip_stage.sh <src_root> <work_board> <board> <test_type>" >&2
  exit 2
fi

src_root=$1
work_board=$2
board=$3
test_type=$4
tag="publishBoardZip(${test_type}/${board})"

found=0
while IFS= read -r -d '' board_dir; do
  found=1
  while IFS= read -r -d '' model_dir; do
    name=$(basename "$model_dir")
    if [ -e "$work_board/$name" ]; then
      echo "$tag: duplicate deployment $name under $src_root"
      exit 1
    fi
    cp -a "$model_dir" "$work_board/"
  done < <(find "$board_dir" -mindepth 1 -maxdepth 1 -type d -print0)
done < <(find "$src_root" -mindepth 2 -maxdepth 2 -type d -name "$board" -print0)

if [ "$found" = "0" ] || [ -z "$(find "$work_board" -mindepth 1 -maxdepth 1 -type d -print -quit)" ]; then
  echo "$tag: no deployment directories found under $src_root"
  touch "$work_board/.NO_DEPLOYMENTS"
  exit 0
fi

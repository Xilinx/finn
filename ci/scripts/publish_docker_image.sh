#!/bin/bash
# publish_docker_image.sh <image_dir> <tag> <build_number>
#
# Save the named Docker image to <image_dir>/finn-docker-image.tar.gz and
# write its tag to a sibling file. image_dir is per-build, so the only
# concurrency is a same-build retry whose bytes are identical. Unique
# per-invocation temp files plus the atomic rename are the serialisation
# point (no NFS lock, which flock cannot provide cross-host anyway). A
# concurrent retry may redo the docker save, which is rare and harmless.
set -eo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: publish_docker_image.sh <image_dir> <tag> <build_number>" >&2
  exit 2
fi

image_dir=$1
tag=$2
build=$3

final_img="${image_dir}/finn-docker-image.tar.gz"
final_tag="${image_dir}/finn-docker-tag.txt"
# Unique per invocation (pid + host) so a concurrent same-build retry cannot
# clobber our half-written temp; the trap removes it on any early exit.
uniq="${build}.$$.$(hostname -s 2>/dev/null || echo host)"
tmp_img="${final_img}.tmp-${uniq}"
tmp_tag="${final_tag}.tmp-${uniq}"
trap 'rm -f "$tmp_img" "$tmp_tag"' EXIT

if command -v pigz >/dev/null 2>&1; then
  docker save "$tag" | pigz -p "$(nproc)" > "$tmp_img"
else
  docker save "$tag" | gzip > "$tmp_img"
fi
printf '%s\n' "$tag" > "$tmp_tag"
sync "$tmp_img" "$tmp_tag"
# image last so a reader that gates on both files never sees the image without
# its tag (the loader keys off finn-docker-image.tar.gz being present).
mv -f "$tmp_tag" "$final_tag"
mv -f "$tmp_img" "$final_img"

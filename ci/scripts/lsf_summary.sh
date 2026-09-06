#!/bin/bash
# lsf_summary.sh <build_dir> <agent> <stash>
#
# Per-tool run counts and hosts so Blue Ocean shows LSF fan-out.
# Best-effort: failures are logged but never abort the pipeline (the real
# test result is owned by pytest).
set +e

if [ "$#" -ne 3 ]; then
  echo "Usage: lsf_summary.sh <build_dir> <agent> <stash>" >&2
  exit 2
fi

bd=$1
agent=$2
stash=$3
tag="[lsf-summary ${stash}]"

if [ ! -d "$bd" ]; then
  echo "$tag no build dir"
  exit 0
fi

# vitis_hls log line: "INFO: [HLS 200-10] ... on host '<host>'"
# vivado log header:  "# Running On: <host>,"
hls_hosts=$(find "$bd" -name vitis_hls.log -exec grep -h "INFO: \[HLS 200-10\] .* on host '" {} + 2>/dev/null \
  | awk -F"on host '" '{print $2}' | awk -F"'" '{print $1}')
viv_hosts=$(find "$bd" -name vivado.log -exec grep -h '^# Running On:' {} + 2>/dev/null \
  | awk -F'Running On: *' '{print $2}' | awk -F',' '{print $1}')

all_hosts=$(printf '%s\n%s\n' "$hls_hosts" "$viv_hosts" | grep -v '^$')
n_runs=$(echo "$all_hosts" | grep -c .)
if [ "$n_runs" = "0" ]; then
  # warn loudly when logs exist but the host-line format has drifted; a
  # silent "no tool runs" otherwise masks a parsing regression after a
  # Vivado/Vitis version bump.
  n_logs=$(find "$bd" \( -name vitis_hls.log -o -name vivado.log \) 2>/dev/null | wc -l)
  if [ "$n_logs" -gt 0 ]; then
    echo "$tag no parseable host found in ${n_logs} log(s) (format change?)"
  else
    echo "$tag no tool runs"
  fi
  exit 0
fi
n_remote=$(echo "$all_hosts" | grep -vcx "$agent")
n_local=$(echo "$all_hosts" | grep -cx "$agent")
buckets=$(echo "$all_hosts" | sort | uniq -c | sort -rn)
n_hosts=$(echo "$buckets" | grep -c .)

if [ "$n_remote" = "0" ]; then
  echo "$tag $n_runs tool run(s) on this agent (no LSF dispatch)"
else
  echo "$tag $n_runs tool run(s) across $n_hosts host(s) ($n_remote remote / $n_local local)"
fi
top=$(echo "$buckets" | head -5 | awk '{printf "%s(%d) ", $2, $1}')
echo "                          top: $top"
